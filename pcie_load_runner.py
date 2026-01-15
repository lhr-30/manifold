import random
import statistics as stats
import time
import os
import torch
import torch.distributed as dist

import argparse
from pcie_load_client import PcieLoadClient
from kvcache_pcie_sim import BaselineLoader, OptimizedLoader

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


def setup_distributed():
    if "RANK" not in os.environ:
        raise SystemExit("Use torchrun --nproc_per_node=2 to launch this script.")

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size != 2:
        raise SystemExit(f"This script expects WORLD_SIZE=2, got {world_size}.")

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank

def allocate_cache(num_blocks, block_elems, dtype, pin, device):
    host_cache = torch.empty((num_blocks, block_elems), dtype=dtype, device="cpu", pin_memory=pin)
    if dtype in (torch.int8, torch.uint8, torch.int32):
        host_cache.random_(0, 127)
    else:
        host_cache.uniform_(0, 1)
    gpu_cache = torch.empty((num_blocks, block_elems), dtype=dtype, device=device)
    return host_cache, gpu_cache


def compute_sleep_seconds(h2d_seconds, util_ratio, base_sleep_ms):
    if util_ratio <= 0:
        return base_sleep_ms / 1e3
    target_total = h2d_seconds / util_ratio
    extra_sleep = max(0.0, target_total - h2d_seconds)
    return (base_sleep_ms / 1e3) + extra_sleep


def run_baseline(
    client,
    loader,
    block_bytes,
    util_ratio,
    warmup,
    iters,
    sleep_min_ms,
    sleep_max_ms,
):
    per_iter_gbps = []
    for it in range(warmup + iters):
        request = client.build_request()
        if request.num_blocks == 0:
            if it >= warmup:
                per_iter_gbps.append(0.0)
            _maybe_sleep(0.0, util_ratio, sleep_min_ms, sleep_max_ms)
            continue

        bytes_moved = request.num_blocks * block_bytes
        h2d_seconds = loader.load(request.indices_cpu)
        if it >= warmup:
            per_iter_gbps.append(_gbps(bytes_moved, h2d_seconds))
        _maybe_sleep(h2d_seconds, util_ratio, sleep_min_ms, sleep_max_ms)
    return per_iter_gbps


def run_optimized(
    client,
    loader,
    block_bytes,
    util_ratio,
    warmup,
    iters,
    sleep_min_ms,
    sleep_max_ms,
    world_size,
    rank,
    indices_buffer,
    count_buffer,
):
    per_iter_gbps = []
    per_iter_reduce_ms = []

    for it in range(warmup + iters):
        request = client.build_request()
        indices_buffer.fill_(-1)
        if request.num_blocks > 0:
            indices_buffer[: request.num_blocks] = request.indices_cpu
        count_buffer[0] = request.num_blocks

        gathered_counts = [torch.zeros_like(count_buffer) for _ in range(world_size)]
        gathered_indices = [torch.empty_like(indices_buffer) for _ in range(world_size)]
        dist.all_gather(gathered_counts, count_buffer)
        dist.all_gather(gathered_indices, indices_buffer)

        owner_rank = it % world_size
        owner_count = int(gathered_counts[owner_rank].item())
        if owner_count == 0:
            if it >= warmup:
                per_iter_gbps.append(0.0)
                per_iter_reduce_ms.append(0.0)
            _maybe_sleep(0.0, util_ratio, sleep_min_ms, sleep_max_ms)
            continue

        indices = gathered_indices[owner_rank][:owner_count].clone()
        indices_rank = indices[rank::world_size]
        bytes_moved = indices_rank.numel() * block_bytes
        h2d_seconds, reduce_ms = loader.load(indices, rank)

        if it >= warmup:
            per_iter_gbps.append(_gbps(bytes_moved, h2d_seconds))
            per_iter_reduce_ms.append(reduce_ms)

        _maybe_sleep(h2d_seconds, util_ratio, sleep_min_ms, sleep_max_ms)

    return per_iter_gbps, per_iter_reduce_ms


def summarize_bandwidth(per_iter_gbps, device, world_size):
    stats_tensor = torch.tensor(
        [stats.mean(per_iter_gbps), stats.median(per_iter_gbps)], dtype=torch.float64, device=device
    )
    gathered = [torch.empty_like(stats_tensor) for _ in range(world_size)]
    dist.all_gather(gathered, stats_tensor)
    return gathered


def _gbps(bytes_moved, seconds):
    return (bytes_moved / 1e9) / seconds if seconds > 0 else float("inf")


def _maybe_sleep(h2d_seconds, util_ratio, sleep_min_ms, sleep_max_ms):
    if sleep_max_ms < sleep_min_ms:
        base_sleep_ms = sleep_min_ms
    else:
        base_sleep_ms = random.uniform(sleep_min_ms, sleep_max_ms)
    sleep_seconds = compute_sleep_seconds(h2d_seconds, util_ratio, base_sleep_ms)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "optimized"], default="baseline")
    ap.add_argument("--num-blocks", type=int, default=2048, help="Number of blocks in the KV cache per GPU.")
    ap.add_argument("--block-mb", type=float, default=2.0, help="Block size (MB) for each KV cache block.")
    ap.add_argument("--read-min-blocks", type=int, default=4)
    ap.add_argument("--read-max-blocks", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pcie-util", type=float, default=70.0, help="Target PCIe utilization (0-100%).")
    ap.add_argument("--sleep-min-ms", type=float, default=1.0, help="Random compute sleep min (ms).")
    ap.add_argument("--sleep-max-ms", type=float, default=5.0, help="Random compute sleep max (ms).")
    ap.add_argument("--pin", action="store_true", help="Use pinned host memory.")
    ap.add_argument("--dtype", type=str, default="float16", choices=DTYPE_MAP.keys())
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = DTYPE_MAP[args.dtype]

    if args.read_min_blocks <= 0 or args.read_max_blocks <= 0:
        raise SystemExit("read-min-blocks and read-max-blocks must be positive.")
    if args.read_min_blocks > args.read_max_blocks:
        raise SystemExit("read-min-blocks must be <= read-max-blocks.")
    if args.read_max_blocks > args.num_blocks:
        raise SystemExit("read-max-blocks must be <= num-blocks.")
    if args.block_mb <= 0:
        raise SystemExit("block-mb must be positive.")

    elem_size = bytes_per_elem(dtype)
    block_bytes = int(args.block_mb * 1e6)
    block_elems = max(1, block_bytes // elem_size)
    block_bytes = block_elems * elem_size

    host_cache, gpu_cache = allocate_cache(args.num_blocks, block_elems, dtype, args.pin, device)
    stream = torch.cuda.Stream(device=device)

    util_ratio = max(0.0, min(1.0, args.pcie_util / 100.0))
    client = PcieLoadClient(
        rank=rank,
        num_blocks=args.num_blocks,
        read_min_blocks=args.read_min_blocks,
        read_max_blocks=args.read_max_blocks,
        util_ratio=util_ratio,
        seed=args.seed,
    )

    if rank == 0:
        print("=== KV Cache PCIe Simulation ===")
        print(f"Mode: {args.mode}")
        print(f"World size: {world_size}, dtype: {dtype}, pin: {args.pin}")
        print(
            f"Blocks: {args.num_blocks}, block: {block_bytes/1e6:.2f} MB, "
            f"read range: [{args.read_min_blocks}, {args.read_max_blocks}] blocks"
        )
        print(f"PCIe utilization target: {args.pcie_util:.1f}%")
        print(f"Sleep jitter: {args.sleep_min_ms:.2f}~{args.sleep_max_ms:.2f} ms")
        print()

    max_blocks = args.read_max_blocks
    indices_buffer = torch.full((max_blocks,), -1, dtype=torch.int64, device="cpu")
    count_buffer = torch.zeros((1,), dtype=torch.int64, device="cpu")

    baseline_loader = BaselineLoader(host_cache, gpu_cache, stream, args.pin, device)
    optimized_loader = OptimizedLoader(
        host_cache, gpu_cache, block_elems, stream, args.pin, device, world_size
    )

    if args.mode == "baseline":
        per_iter_gbps = run_baseline(
            client,
            baseline_loader,
            block_bytes,
            util_ratio,
            args.warmup,
            args.iters,
            args.sleep_min_ms,
            args.sleep_max_ms,
        )
        gathered = summarize_bandwidth(per_iter_gbps, device, world_size)
        if rank == 0:
            for r, row in enumerate(gathered):
                print(
                    f"Rank{r} PCIe H2D bandwidth (GB/s): "
                    f"avg {row[0].item():.2f}, p50 {row[1].item():.2f}"
                )
            print("Notes: bandwidth is per-rank based on H2D bytes and copy time.")
    else:
        per_iter_gbps, per_iter_reduce_ms = run_optimized(
            client,
            optimized_loader,
            block_bytes,
            util_ratio,
            args.warmup,
            args.iters,
            args.sleep_min_ms,
            args.sleep_max_ms,
            world_size,
            rank,
            indices_buffer,
            count_buffer,
        )
        gathered = summarize_bandwidth(per_iter_gbps, device, world_size)
        if rank == 0:
            for r, row in enumerate(gathered):
                print(
                    f"Rank{r} PCIe H2D bandwidth (GB/s): "
                    f"avg {row[0].item():.2f}, p50 {row[1].item():.2f}"
                )
            if per_iter_reduce_ms:
                print(
                    "Reduce latency (ms) on rank0: "
                    f"avg {stats.mean(per_iter_reduce_ms):.3f}, p50 {stats.median(per_iter_reduce_ms):.3f}"
                )
            print("Notes: reduce combines per-rank buffers to rank0 (sum) after H2D.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# Example:
# torchrun --standalone --nproc_per_node=2 kvcache_pcie_sim.py --mode baseline --pin
# torchrun --standalone --nproc_per_node=2 kvcache_pcie_sim.py --mode optimized --pin
