#!/usr/bin/env python3
import argparse
import time
import torch

import torch.utils.cpp_extension

cpp_source = open('custom_pinner.cpp', 'r').read()
module_name = 'custom_pinner_fixed'
try:
    custom_ops = torch.utils.cpp_extension.load_inline(
        name=module_name,
        cpp_sources=cpp_source,
        verbose=True,
        with_cuda=True
    )
except Exception as e:
    print("\nCompilation failed. Please try: rm -rf ~/.cache/torch_extensions to clear cache and retry.")
    raise e

print("finish jit compile custom_pinner_fixed")

def make_tensor(size_gib: float, dtype: torch.dtype, shared: bool, pin: bool) -> torch.Tensor:
    elem_size = torch.tensor([], dtype=dtype).element_size()
    num_bytes = int(size_gib * (1024**3))
    numel = max(1, num_bytes // elem_size)

    x = torch.empty(numel, dtype=dtype, device="cpu", pin_memory=pin)
    
    # touch pages
    if dtype.is_floating_point:
        x.uniform_(0, 1)
    else:
        x.random_(0, 127)

    if shared:
        x.share_memory_()
        custom_ops.pin_buffer(x)
    return x


def run_read_bw(x: torch.Tensor, iters: int) -> tuple[float, float]:
    """CPU read+reduce bandwidth via sum()."""
    view = x.view(-1)

    # warmup
    _ = float(view.sum().item())

    t0 = time.perf_counter()
    for _ in range(iters):
        _ = float(view.sum().item())
    t1 = time.perf_counter()

    bytes_read = view.numel() * view.element_size() * iters
    gib = bytes_read / (1024**3)
    secs = t1 - t0
    bw = gib / secs
    return bw, secs


def run_h2d_bw(x: torch.Tensor, iters: int, device: str, non_blocking: bool) -> tuple[float, float]:
    """Host->GPU copy bandwidth."""
    dev = torch.device(device)
    y = torch.empty_like(x, device=dev)

    # warmup (also warms up cuda context)
    
    y.copy_(x, non_blocking=non_blocking)
    torch.cuda.synchronize(dev)
    # timing with CUDA events (more accurate than perf_counter for GPU ops)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    print(f"x.is_pinned(): {x.is_pinned()}, {x.data_ptr()}")
    start_evt.record()
    for _ in range(iters):
        y.copy_(x, non_blocking=non_blocking)
    end_evt.record()
    torch.cuda.synchronize(dev)
    print(f"x.is_pinned(): {x.is_pinned()}, {x.data_ptr()}")

    ms = start_evt.elapsed_time(end_evt)
    secs = ms / 1e3

    bytes_xfer = x.numel() * x.element_size() * iters
    gib = bytes_xfer / (1024**3)
    bw = gib / secs
    return bw, secs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-gib", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=10)

    ap.add_argument("--threads", type=int, default=1, help="torch intra-op threads")
    ap.add_argument("--interop", type=int, default=1, help="torch interop threads")

    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "int8"])
    ap.add_argument("--mode", type=str, default="both", choices=["normal", "shared", "both"])

    ap.add_argument("--op", type=str, default="both", choices=["read", "h2d", "both"])
    ap.add_argument("--gpu", type=str, default="cuda:0")
    ap.add_argument("--pin", action="store_true", help="allocate pinned host memory")
    ap.add_argument("--non-blocking", action="store_true", help="use non_blocking copy_")
    args = ap.parse_args()

    # must set once before parallel work begins
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.interop)

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "int8": torch.int8}
    dtype = dtype_map[args.dtype]
    elem_size = torch.tensor([], dtype=dtype).element_size()

    if args.op in ("h2d", "both") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available, cannot run --op h2d/both")

    def banner(tag: str, x: torch.Tensor):
        size_gib = x.numel() * x.element_size() / (1024**3)
        print(
            f"=== {tag} === dtype={dtype} elem_size={elem_size}B "
            f"numel={x.numel():,} size={size_gib:.2f} GiB "
            f"is_shared={x.is_shared()} pinned={x.is_pinned()} "
            f"threads={args.threads} interop={args.interop} iters={args.iters}",
            flush=True,
        )

    def run_case(tag: str, shared: bool):
        print(f"args: shared:{shared}, pin:{args.pin}, non_blocking:{args.non_blocking}", flush=True)
        x = make_tensor(args.size_gib, dtype, shared=shared, pin=args.pin)
        banner(tag, x)

        total_gib = x.numel() * x.element_size() * args.iters / (1024**3)

        if args.op in ("read", "both"):
            bw, secs = run_read_bw(x, args.iters)
            print(f"  [CPU READ] time={secs:.4f}s bytes={total_gib:.3f}GiB bw={bw:.2f} GiB/s", flush=True)

        if args.op in ("h2d", "both"):
            bw, secs = run_h2d_bw(x, args.iters, device=args.gpu, non_blocking=args.non_blocking)
            print(
                f"  [H2D COPY] time={secs:.4f}s bytes={total_gib:.3f}GiB bw={bw:.2f} GiB/s "
                f"(non_blocking={args.non_blocking})",
                flush=True,
            )

        print("", flush=True)

    if args.mode in ("normal", "both"):
        run_case("NORMAL", shared=False)

    if args.mode in ("shared", "both"):
        run_case("SHARED", shared=True)


if __name__ == "__main__":
    main()

# python shared_read_bw_1proc.py --size-gib 1 --iters 20 --op both --mode both --pin --non-blocking