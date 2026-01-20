import argparse
import os
import random
import socket
import statistics as stats

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pcie_load_client import PcieLoadClient, PcieLoadProcess
from kvcache_pcie_sim import (
    BaselineLoadManager,
    OptimizedLoadManager,
    OptimizedLoadManagerClient,
    OptimizedLoadWorker,
)

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


def allocate_cache(num_cpu_blocks, num_gpu_blocks, block_elems, dtype, pin, device):
    host_cache = torch.empty((num_cpu_blocks, block_elems), dtype=dtype, device="cpu", pin_memory=pin)
    if dtype in (torch.int8, torch.uint8, torch.int32):
        host_cache.random_(0, 127)
    else:
        host_cache.uniform_(0, 1)
    gpu_cache = torch.empty((num_gpu_blocks, block_elems), dtype=dtype, device=device)
    return host_cache, gpu_cache

def allocate_shared_host_caches(num_clients, num_cpu_blocks, block_elems, dtype, pin, seed):
    host_caches = []
    for rank in range(num_clients):
        host_cache = torch.empty(
            (num_cpu_blocks, block_elems),
            dtype=dtype,
            device="cpu",
            pin_memory=pin,
        ).share_memory_()
        gen = torch.Generator()
        gen.manual_seed(seed + rank)
        if dtype in (torch.int8, torch.uint8, torch.int32):
            host_cache.random_(0, 127, generator=gen)
        else:
            host_cache.uniform_(0, 1, generator=gen)
        host_caches.append(host_cache)
    return host_caches

def bind_cpu(rank: int, cores_per_proc: int):
    start = rank * cores_per_proc
    end = start + cores_per_proc
    core_ids = list(range(start, end))

    os.sched_setaffinity(0, core_ids)  # pid=0 means current process
    
    # os.environ["OMP_NUM_THREADS"] = str(cores_per_proc)
    # os.environ["MKL_NUM_THREADS"] = str(cores_per_proc)
    # torch.set_num_threads(cores_per_proc)
    # torch.set_num_interop_threads(1)

def run_client_baseline(rank, args):
    bind_cpu(rank, args.cores_per_proc)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dtype = DTYPE_MAP[args.dtype]
    elem_size = bytes_per_elem(dtype)
    block_bytes = int(args.block_mb * 1e6)
    block_elems = max(1, block_bytes // elem_size)
    block_bytes = block_elems * elem_size
    host_cache, gpu_cache = allocate_cache(
            args.num_cpu_blocks,
            args.num_gpu_blocks,
            block_elems,
            dtype,
            args.pin,
            device,
        )
    load_manager = BaselineLoadManager(host_cache, gpu_cache, torch.cuda.Stream(device=device), args.pin, device)
    util_ratio = max(0.0, min(1.0, args.pcie_util / 100.0))
    rng = random.Random(args.seed + rank) if args.randomize else None
    block_bytes = int(args.block_mb * 1e6)
    process = PcieLoadProcess(
        None,
        load_manager,
        block_bytes,
        util_ratio,
        args.warmup,
        args.iters,
        args.sleep_min_ms,
        args.sleep_max_ms,
        rng,
    )
    client = PcieLoadClient(
        rank=rank,
        device_id=rank,
        device=device,
        num_cpu_blocks=args.num_cpu_blocks,
        num_gpu_blocks=args.num_gpu_blocks,
        read_min_blocks=args.read_min_blocks,
        read_max_blocks=args.read_max_blocks,
        util_ratio=util_ratio,
        seed=args.seed,
        randomize=args.randomize,
        load_manager=load_manager,
        process=process,
    )
    process.client = client

    per_iter_gbps = client.run()
    stats_tensor = torch.tensor(
        [stats.mean(per_iter_gbps), stats.median(per_iter_gbps)], dtype=torch.float64, device=device
    )
    print()
    print(
        f"Rank{rank} PCIe H2D bandwidth (GB/s): "
        f"avg {stats_tensor[0].item():.2f}, p50 {stats_tensor[1].item():.2f}",
        f"affinity = {sorted(os.sched_getaffinity(0))}"
    )
    print("Notes: bandwidth is per-rank based on H2D bytes and copy time.")

def run_client_optimized(rank, args, queues, host_caches):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_init_method,
        world_size=args.num_clients,
        rank=rank,
    )
    dtype = DTYPE_MAP[args.dtype]
    elem_size = bytes_per_elem(dtype)
    block_bytes = int(args.block_mb * 1e6)
    block_elems = max(1, block_bytes // elem_size)
    block_bytes = block_elems * elem_size
    host_cache = host_caches[rank]
    if not host_cache.is_shared():
        raise SystemExit("Optimized mode requires shared host caches across processes.")
    gpu_cache = torch.empty((args.num_gpu_blocks, block_elems), dtype=dtype, device=device)
    worker = OptimizedLoadWorker(
        rank=rank,
        host_cache=host_cache,
        host_caches=host_caches,
        gpu_cache=gpu_cache,
        stream=torch.cuda.Stream(device=device),
        pin=args.pin,
        device=device,
    )
    load_manager = OptimizedLoadManagerClient(
        rank=rank,
        request_queue=queues["request"],
        task_queue=queues["tasks"][rank],
        reduce_queue=queues["reduce"][rank],
        worker=worker,
    )
    util_ratio = max(0.0, min(1.0, args.pcie_util / 100.0))
    rng = random.Random(args.seed + rank) if args.randomize else None
    process = PcieLoadProcess(
        None,
        load_manager,
        block_bytes,
        util_ratio,
        args.warmup,
        args.iters,
        args.sleep_min_ms,
        args.sleep_max_ms,
        rng,
    )
    client = PcieLoadClient(
        rank=rank,
        device_id=rank,
        device=device,
        num_cpu_blocks=args.num_cpu_blocks,
        num_gpu_blocks=args.num_gpu_blocks,
        read_min_blocks=args.read_min_blocks,
        read_max_blocks=args.read_max_blocks,
        util_ratio=util_ratio,
        seed=args.seed,
        randomize=args.randomize,
        load_manager=load_manager,
        process=process,
    )
    process.client = client

    per_iter_gbps = client.run()
    stats_tensor = torch.tensor(
        [stats.mean(per_iter_gbps), stats.median(per_iter_gbps)], dtype=torch.float64, device=device
    )
    print(
        f"Rank{rank} PCIe H2D bandwidth (GB/s): "
        f"avg {stats_tensor[0].item():.2f}, p50 {stats_tensor[1].item():.2f}"
    )
    print("Notes: bandwidth is per-rank based on H2D bytes and copy time.")
    load_manager.shutdown()
    dist.destroy_process_group()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "optimized"], default="baseline")
    ap.add_argument("--num-gpu-blocks", type=int, default=256, help="Number of blocks in the KV cache per GPU.")
    ap.add_argument("--num-cpu-blocks", type=int, default=512, help="Number of blocks in the KV cache on CPU.")
    ap.add_argument("--block-mb", type=float, default=16.0, help="Block size (MB) for each KV cache block.")
    ap.add_argument("--read-min-blocks", type=int, default=32)
    ap.add_argument("--read-max-blocks", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--pcie-util", type=float, default=70.0, help="Target PCIe utilization (0-100%).")
    ap.add_argument("--sleep-min-ms", type=float, default=5.0, help="Random compute sleep min (ms).")
    ap.add_argument("--sleep-max-ms", type=float, default=5.0, help="Random compute sleep max (ms).")
    ap.add_argument("--pin", action="store_true", help="Use pinned host memory.")
    ap.add_argument("--dtype", type=str, default="float16", choices=DTYPE_MAP.keys())
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--randomize", action="store_true", help="Enable randomized block sizes and sleep jitter.")
    ap.add_argument("--cores-per-proc", type=int, help="set CPU cores per process for binding.", default=6)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    if torch.cuda.device_count() < 2:
        raise SystemExit("This script expects at least 2 CUDA devices.")
    args.num_clients = 2

    if not args.randomize:
        if args.read_min_blocks != args.read_max_blocks:
            raise SystemExit("For deterministic runs, read-min-blocks must equal read-max-blocks.")
        if args.sleep_min_ms != args.sleep_max_ms:
            raise SystemExit("For deterministic runs, sleep-min-ms must equal sleep-max-ms.")

    print("=== KV Cache PCIe Simulation ===")
    print(f"Mode: {args.mode}")
    print(f"Clients: {args.num_clients}, dtype: {DTYPE_MAP[args.dtype]}, pin: {args.pin}")
    print(
        f"CPU blocks: {args.num_cpu_blocks}, GPU blocks: {args.num_gpu_blocks}, "
        f"block: {args.block_mb:.2f} MB, read range: [{args.read_min_blocks}, {args.read_max_blocks}] blocks"
    )
    print(f"PCIe utilization target: {args.pcie_util:.1f}%")
    print(f"Sleep jitter: {args.sleep_min_ms:.2f}~{args.sleep_max_ms:.2f} ms")
    print()

    if args.mode == "baseline":
        # mp.spawn(run_client_baseline, args=(args, ), nprocs=args.num_clients, join=True)
        run_client_baseline(0, args)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
        sock.close()
        args.dist_init_method = f"tcp://127.0.0.1:{port}"
        dtype = DTYPE_MAP[args.dtype]
        elem_size = bytes_per_elem(dtype)
        block_bytes = int(args.block_mb * 1e6)
        block_elems = max(1, block_bytes // elem_size)
        host_caches = allocate_shared_host_caches(
            args.num_clients,
            args.num_cpu_blocks,
            block_elems,
            dtype,
            args.pin,
            args.seed,
        )
        request_queue = mp.Queue()
        task_queues = [mp.Queue() for _ in range(args.num_clients)]
        reduce_queues = [mp.Queue() for _ in range(args.num_clients)]
        load_manager = OptimizedLoadManager(
            request_queue=request_queue,
            task_queues=task_queues,
            reduce_queues=reduce_queues,
            num_clients=args.num_clients,
        )
        manager_process = mp.Process(target=load_manager.serve, daemon=True)
        manager_process.start()
        queues = {
            "request": request_queue,
            "tasks": task_queues,
            "reduce": reduce_queues,
        }
        mp.spawn(
            run_client_optimized,
            args=(args, queues, host_caches),
            nprocs=args.num_clients,
            join=True,
        )
        request_queue.put(None)
        manager_process.join(timeout=5)


if __name__ == "__main__":
    main()

# Example:
# python pcie_load_runner.py --mode baseline --pin
# python pcie_load_runner.py --mode optimized --pin
