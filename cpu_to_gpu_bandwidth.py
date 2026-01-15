import argparse
import math
import time
import statistics as stats

import torch


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


def format_gbps(bytes_moved: float, seconds: float) -> float:
    # GB/s using 1e9 bytes
    return (bytes_moved / 1e9) / seconds if seconds > 0 else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=4.0, help="Total data moved per iteration (GB).")
    ap.add_argument("--chunk-mb", type=float, default=256.0, help="Chunk size per copy (MB).")
    ap.add_argument("--iters", type=int, default=50, help="Measured iterations.")
    ap.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    ap.add_argument("--pin", action="store_true", help="Use pinned host memory.")
    ap.add_argument("--dtype", type=str, default="float16", choices=DTYPE_MAP.keys(), help="Tensor dtype.")
    ap.add_argument("--device", type=str, default="cuda:0", help="CUDA device, e.g., cuda:0")
    ap.add_argument("--stream", type=str, default="default", choices=["default", "new"], help="Use default or a new CUDA stream.")
    ap.add_argument("--verify", action="store_true", help="Do a small verification to prevent dead-code elimination concerns.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    total_bytes = int(args.gb * 1e9)
    chunk_bytes = int(args.chunk_mb * 1e6)
    if chunk_bytes <= 0 or total_bytes <= 0:
        raise SystemExit("gb and chunk-mb must be positive.")
    if chunk_bytes > total_bytes:
        chunk_bytes = total_bytes

    elem_size = bytes_per_elem(dtype)
    chunk_elems = max(1, chunk_bytes // elem_size)
    # Recompute chunk_bytes to be multiple of elem_size
    chunk_bytes = chunk_elems * elem_size

    num_chunks = math.ceil(total_bytes / chunk_bytes)
    # Actual moved bytes per iter (last chunk padded to chunk_bytes)
    moved_bytes_per_iter = num_chunks * chunk_bytes

    print("=== CPU -> GPU Bandwidth Test (H2D) ===")
    print(f"Device: {device}, dtype: {dtype} ({elem_size} B/elem)")
    print(f"Target per-iter: {total_bytes/1e9:.3f} GB, chunk: {chunk_bytes/1e6:.1f} MB, chunks/iter: {num_chunks}")
    print(f"Actual moved per-iter (aligned): {moved_bytes_per_iter/1e9:.3f} GB")
    print(f"Pin memory: {args.pin}, iters: {args.iters}, warmup: {args.warmup}, stream: {args.stream}")
    print()

    # Allocate host chunk buffer (reused)
    # Use random data to avoid special cases; doesn't matter much for memcpy bandwidth.
    host = torch.empty((chunk_elems,), dtype=dtype, device="cpu", pin_memory=args.pin)
    host.random_(0, 127) if dtype in (torch.int8, torch.uint8, torch.int32) else host.uniform_(0, 1)

    # Allocate device chunk buffer (reused)
    dev = torch.empty((chunk_elems,), dtype=dtype, device=device)

    # Choose stream
    stream = torch.cuda.default_stream(device) if args.stream == "default" else torch.cuda.Stream(device=device)

    # Events for timing on GPU
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    def one_iter_timed() -> float:
        # returns milliseconds measured on GPU timeline
        with torch.cuda.stream(stream):
            start_evt.record(stream)
            for _ in range(num_chunks):
                dev.copy_(host, non_blocking=args.pin)  # non_blocking only effective if host is pinned
            end_evt.record(stream)
        end_evt.synchronize()
        ms = start_evt.elapsed_time(end_evt)
        return ms

    # Warmup
    for _ in range(args.warmup):
        _ = one_iter_timed()

    # Measure
    times_ms = []
    for _ in range(args.iters):
        ms = one_iter_timed()
        times_ms.append(ms)

    # Optional verification
    if args.verify:
        # quick checksum on GPU to force use
        s = float(dev[:1024].float().sum().item())
        print(f"Verify checksum (dev[:1024] sum): {s:.4f}")

    # Stats
    avg_ms = stats.mean(times_ms)
    p50_ms = stats.median(times_ms)
    p95_ms = sorted(times_ms)[max(0, int(0.95 * len(times_ms)) - 1)]
    p99_ms = sorted(times_ms)[max(0, int(0.99 * len(times_ms)) - 1)]
    min_ms = min(times_ms)
    max_ms = max(times_ms)

    avg_s = avg_ms / 1e3
    bw_avg = format_gbps(moved_bytes_per_iter, avg_s)

    print("=== Results ===")
    print(f"Time per iter (ms): avg {avg_ms:.3f}, p50 {p50_ms:.3f}, p95 {p95_ms:.3f}, p99 {p99_ms:.3f}, min {min_ms:.3f}, max {max_ms:.3f}")
    print(f"Bandwidth (GB/s, decimal): {bw_avg:.2f}")
    print()
    print("Tips:")
    print("- If you want to measure pageable memory behavior, run without --pin.")
    print("- If you want steadier results, increase --iters and keep the node otherwise idle.")
    print("- For PCIe systems, bandwidth typically saturates with larger --chunk-mb (e.g., 64~512MB).")


if __name__ == "__main__":
    main()
    
# python cpu_to_gpu_bandwidth.py --gb 8 --chunk-mb 256 --iters 50 --warmup 10 --pin --dtype float16
