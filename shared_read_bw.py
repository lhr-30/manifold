#!/usr/bin/env python3
import argparse
import os
import time
import torch
import torch.multiprocessing as mp


def worker(rank: int, tensor: torch.Tensor, iters: int, barrier: mp.Barrier, overlap: bool):
    # 避免每个进程再开很多线程（否则你测到的是 OpenMP 线程调度 + 带宽混在一起）
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # 选择读的范围：overlap=True -> 两个进程读同一段；False -> 各读一半
    n = tensor.numel()
    if overlap:
        view = tensor.view(-1)  # full overlap
    else:
        half = n // 2
        if rank == 0:
            view = tensor.view(-1)[:half]
        else:
            view = tensor.view(-1)[half: 2 * half]

    # 预热（把页 fault、cache 之类的影响减小）
    _ = float(view.sum().item())

    barrier.wait()
    t0 = time.perf_counter()

    acc = 0.0
    for _ in range(iters):
        # 读为主：sum 会流式读取内存，并聚合为标量
        acc += float(view.sum().item())

    t1 = time.perf_counter()
    barrier.wait()

    bytes_read = view.numel() * view.element_size() * iters
    gb = bytes_read / (1024**3)
    secs = t1 - t0
    bw = gb / secs

    print(
        f"[rank {rank}] read_elems={view.numel():,} "
        f"iters={iters} time={secs:.4f}s "
        f"bytes={gb:.3f}GiB bw={bw:.2f} GiB/s "
        f"(acc={acc:.3e})",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size-gib", type=float, default=4.0, help="Tensor size in GiB (approx).")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "int8"])
    ap.add_argument("--iters", type=int, default=50, help="Iterations per process.")
    ap.add_argument("--overlap", action="store_true", help="Both processes read the same region.")
    args = ap.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "int8": torch.int8}
    dtype = dtype_map[args.dtype]
    elem_size = torch.tensor([], dtype=dtype).element_size()

    # 分配一个大 tensor（CPU），并放到 shared memory
    num_bytes = int(args.size_gib * (1024**3))
    numel = max(1, num_bytes // elem_size)

    # 用 random 初始化，确保物理页真正分配（touch pages）
    x = torch.empty(numel, dtype=dtype, device="cpu")
    if dtype.is_floating_point:
        x.uniform_(0, 1)
    else:
        x.random_(0, 127)
    x.share_memory_()

    print(
        f"Shared tensor: dtype={dtype}, elem_size={elem_size}B, "
        f"numel={numel:,}, size={numel * elem_size / (1024**3):.2f} GiB, "
        f"overlap={args.overlap}, iters={args.iters}"
    )

    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)

    procs = []
    for r in range(2):
        p = ctx.Process(target=worker, args=(r, x, args.iters, barrier, args.overlap))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()

# python shared_read_bw.py --size-gib 8 --iters 50 --overlap