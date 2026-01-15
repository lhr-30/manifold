#!/usr/bin/env python3
import argparse
import math
import os
import statistics as stats

import torch
import torch.distributed as dist


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int32": torch.int32,
}


def bytes_per_elem(dtype: torch.dtype) -> int:
    return torch.tensor([], dtype=dtype).element_size()


def gbps(bytes_moved: float, seconds: float) -> float:
    return (bytes_moved / 1e9) / seconds if seconds > 0 else float("inf")


def percentile(xs, p):
    xs = sorted(xs)
    idx = max(0, min(len(xs) - 1, int(math.ceil(p * len(xs))) - 1))
    return xs[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=8.0, help="Total H2D bytes per iter across BOTH GPUs (GB).")
    ap.add_argument("--chunk-mb", type=float, default=256.0, help="Chunk size per copy (MB).")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--dtype", type=str, default="float16", choices=DTYPE_MAP.keys())
    ap.add_argument("--reduce", action="store_true", default=True, help="Do NCCL reduce to rank0 (sum).")
    ap.add_argument("--allgather", action="store_true", help="Instead of reduce, gather halves onto rank0 (concatenate).")
    args = ap.parse_args()

    if args.allgather:
        args.reduce = False

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    # torchrun env
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if world_size != 2:
        raise SystemExit(f"This script expects WORLD_SIZE=2, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dtype = DTYPE_MAP[args.dtype]

    dist.init_process_group(backend="nccl")
    dist.barrier()

    # Total bytes per iter across 2 GPUs; per GPU half
    total_bytes = int(args.gb * 1e9)
    per_gpu_bytes = total_bytes // 2
    if per_gpu_bytes <= 0:
        raise SystemExit("--gb too small")

    chunk_bytes = int(args.chunk_mb * 1e6)
    if chunk_bytes <= 0:
        raise SystemExit("--chunk-mb must be positive")

    elem_size = bytes_per_elem(dtype)
    chunk_elems = max(1, chunk_bytes // elem_size)
    chunk_bytes = chunk_elems * elem_size

    num_chunks = math.ceil(per_gpu_bytes / chunk_bytes)
    moved_bytes_per_gpu = num_chunks * chunk_bytes  # aligned
    moved_bytes_total = moved_bytes_per_gpu * 2

    if rank == 0:
        print("=== 2-GPU H2D + (NVLink) NCCL Reduce Benchmark ===")
        print(f"World size: {world_size}")
        print(f"Total target per-iter: {total_bytes/1e9:.3f} GB (split -> {per_gpu_bytes/1e9:.3f} GB per GPU)")
        print(f"Chunk: {chunk_bytes/1e6:.1f} MB, chunks/iter per GPU: {num_chunks}")
        print(f"Aligned moved per-iter: {moved_bytes_total/1e9:.3f} GB (total), {moved_bytes_per_gpu/1e9:.3f} GB (per GPU)")
        print(f"Pin: {args.pin}, dtype: {dtype}, iters: {args.iters}, warmup: {args.warmup}")
        print(f"Mode: {'ALL_GATHER (concat on rank0)' if args.allgather else ('REDUCE to rank0 (sum)' if args.reduce else 'H2D only')}")
        print()

    # Allocate host/device buffers (per rank)
    host = torch.empty((chunk_elems,), dtype=dtype, device="cpu", pin_memory=args.pin)
    if dtype in (torch.int8, torch.uint8, torch.int32):
        host.random_(0, 127)
    else:
        host.uniform_(0, 1)

    dev = torch.empty((chunk_elems,), dtype=dtype, device=device)

    # A fixed per-rank output tensor: the "half" tensor we want to combine on GPU0
    # Size equals per-GPU total moved (aligned) but stored as 1D tensor of elements.
    half_elems = moved_bytes_per_gpu // elem_size
    dev_half = torch.empty((half_elems,), dtype=dtype, device=device)

    stream = torch.cuda.Stream(device=device)

    # Events: time H2D copies that fill dev_half, then time collective
    copy_start = torch.cuda.Event(enable_timing=True)
    copy_end = torch.cuda.Event(enable_timing=True)
    coll_start = torch.cuda.Event(enable_timing=True)
    coll_end = torch.cuda.Event(enable_timing=True)
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)

    def one_iter():
        # Fill dev_half by chunk copies
        with torch.cuda.stream(stream):
            total_start.record(stream)
            copy_start.record(stream)

            offset = 0
            for _ in range(num_chunks):
                # copy chunk to dev (staging)
                dev.copy_(host, non_blocking=args.pin)
                # then write into the big dev_half region
                dev_half[offset : offset + chunk_elems].copy_(dev, non_blocking=True)
                offset += chunk_elems

            copy_end.record(stream)

            if args.reduce or args.allgather:
                # Make sure H2D on this stream is visible to NCCL
                # Record event on our stream; NCCL uses current stream by default in PyTorch collectives.
                # We'll switch to current stream to run collective but wait on our stream work first.
                pass

        # Synchronize stream work before collective timing starts on default stream
        stream.synchronize()

        # Collective on current (default) stream, timed by events on default stream
        if args.reduce:
            coll_start.record()
            # reduce dev_half to rank0 (sum) -> data transferred over NVLink/PCIe via NCCL
            dist.reduce(dev_half, dst=0, op=dist.ReduceOp.SUM)
            coll_end.record()
        elif args.allgather:
            coll_start.record()
            # Gather halves onto rank0: result is concatenation [rank0_half, rank1_half] on rank0
            if rank == 0:
                full = torch.empty((half_elems * 2,), dtype=dtype, device=device)
            else:
                full = torch.empty((half_elems * 2,), dtype=dtype, device=device)  # unused but required by API
            dist.all_gather_into_tensor(full, dev_half)
            coll_end.record()
        else:
            # no collective
            coll_start.record()
            coll_end.record()

        total_end.record()
        torch.cuda.synchronize()

        copy_ms = copy_start.elapsed_time(copy_end)
        coll_ms = coll_start.elapsed_time(coll_end)
        total_ms = total_start.elapsed_time(total_end)
        return copy_ms, coll_ms, total_ms

    # Warmup
    for _ in range(args.warmup):
        one_iter()
    dist.barrier()

    # Measure
    copy_ms_list, coll_ms_list, total_ms_list = [], [], []
    for _ in range(args.iters):
        c, r, t = one_iter()
        copy_ms_list.append(c)
        coll_ms_list.append(r)
        total_ms_list.append(t)

    # Aggregate on rank0
    # We'll gather per-rank averages to rank0 for reporting
    copy_avg = stats.mean(copy_ms_list) / 1e3
    coll_avg = stats.mean(coll_ms_list) / 1e3
    total_avg = stats.mean(total_ms_list) / 1e3

    # For bandwidth: H2D is total moved across both GPUs
    h2d_bw = gbps(moved_bytes_total, copy_avg)  # GB/s aggregate (both GPUs together)
    # For collective: bytes transferred depends on NCCL algorithm; for reduce, a rough lower bound is ~half tensor size.
    # We'll report "effective" bandwidth as tensor_bytes / time, which is common for collective benchmarks.
    tensor_bytes = moved_bytes_per_gpu  # size of dev_half per rank
    coll_bw_eff = gbps(tensor_bytes, coll_avg)

    # Send stats to rank0
    stats_tensor = torch.tensor(
        [h2d_bw, coll_bw_eff, total_avg, stats.mean(copy_ms_list), stats.mean(coll_ms_list), stats.mean(total_ms_list)],
        device=device,
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(stats_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, stats_tensor)

    if rank == 0:
        # gathered[0] and gathered[1] correspond to rank0/rank1
        def fmt_row(r):
            return {
                "rank": int(r),
                "H2D_GBps_agg": float(gathered[r][0].item()),
                "Collective_eff_GBps": float(gathered[r][1].item()),
                "avg_total_s": float(gathered[r][2].item()),
                "avg_copy_ms": float(gathered[r][3].item()),
                "avg_coll_ms": float(gathered[r][4].item()),
                "avg_total_ms": float(gathered[r][5].item()),
            }

        print("=== Per-rank summary (note: H2D_GBps_agg is computed as total(2GPU)/rank_copy_time) ===")
        for r in range(world_size):
            row = fmt_row(r)
            print(
                f"rank{row['rank']}: copy {row['avg_copy_ms']:.3f} ms | "
                f"coll {row['avg_coll_ms']:.3f} ms | total {row['avg_total_ms']:.3f} ms | "
                f"H2D~ {row['H2D_GBps_agg']:.2f} GB/s | coll_eff~ {row['Collective_eff_GBps']:.2f} GB/s"
            )

        # Also print p50/p95 on rank0 (for local variability)
        p50_copy = stats.median(copy_ms_list)
        p95_copy = percentile(copy_ms_list, 0.95)
        p50_coll = stats.median(coll_ms_list)
        p95_coll = percentile(coll_ms_list, 0.95)
        p50_total = stats.median(total_ms_list)
        p95_total = percentile(total_ms_list, 0.95)

        print("\n=== Rank0 latency percentiles (ms) ===")
        print(f"copy  p50 {p50_copy:.3f} | p95 {p95_copy:.3f}")
        print(f"coll  p50 {p50_coll:.3f} | p95 {p95_coll:.3f}")
        print(f"total p50 {p50_total:.3f} | p95 {p95_total:.3f}")

        print("\nNotes:")
        print("- reduce/all_gather uses NCCL; whether it rides NVLink depends on your GPU topology.")
        print("- If you need 'two halves make one full tensor on GPU0', use --allgather (concat) not reduce (sum).")
        print("- For true overlap (H2D and NCCL concurrently), we can put NCCL on a separate stream and use stream-waits.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# torchrun --standalone --nproc_per_node=2 two_gpu_h2d_reduce.py --gb 8 --chunk-mb 256 --iters 50 --warmup 10 --pin --dtype float16
