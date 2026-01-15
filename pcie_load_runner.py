import argparse
import random
import statistics as stats
import torch
import torch.multiprocessing as mp

from pcie_load_client import PcieLoadClient, PcieLoadProcess
from kvcache_pcie_sim import BaselineLoadManager, OptimizedLoadManager

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


def allocate_cache(num_blocks, block_elems, dtype, pin, device):
    host_cache = torch.empty((num_blocks, block_elems), dtype=dtype, device="cpu", pin_memory=pin)
    if dtype in (torch.int8, torch.uint8, torch.int32):
        host_cache.random_(0, 127)
    else:
        host_cache.uniform_(0, 1)
    gpu_cache = torch.empty((num_blocks, block_elems), dtype=dtype, device=device)
    return host_cache, gpu_cache


def run_client(rank, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    dtype = DTYPE_MAP[args.dtype]

    elem_size = bytes_per_elem(dtype)
    block_bytes = int(args.block_mb * 1e6)
    block_elems = max(1, block_bytes // elem_size)
    block_bytes = block_elems * elem_size

    host_cache, gpu_cache = allocate_cache(args.num_blocks, block_elems, dtype, args.pin, device)
    stream = torch.cuda.Stream(device=device)

    util_ratio = max(0.0, min(1.0, args.pcie_util / 100.0))
    rng = random.Random(args.seed + rank) if args.randomize else None

    if args.mode == "baseline":
        load_manager = BaselineLoadManager(host_cache, gpu_cache, stream, args.pin, device)
    else:
        device_handlers = [torch.device(f"cuda:{idx}") for idx in range(args.num_clients)]
        load_manager = OptimizedLoadManager(
            host_cache, gpu_cache, block_elems, stream, args.pin, device, device_handlers
        )

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
        num_blocks=args.num_blocks,
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
    ap.add_argument("--randomize", action="store_true", help="Enable randomized block sizes and sleep jitter.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    if torch.cuda.device_count() < 2:
        raise SystemExit("This script expects at least 2 CUDA devices.")
    args.num_clients = 2

    if args.read_min_blocks <= 0 or args.read_max_blocks <= 0:
        raise SystemExit("read-min-blocks and read-max-blocks must be positive.")
    if args.read_min_blocks > args.read_max_blocks:
        raise SystemExit("read-min-blocks must be <= read-max-blocks.")
    if args.read_max_blocks > args.num_blocks:
        raise SystemExit("read-max-blocks must be <= num-blocks.")
    if args.block_mb <= 0:
        raise SystemExit("block-mb must be positive.")

    if not args.randomize:
        if args.read_min_blocks != args.read_max_blocks:
            raise SystemExit("For deterministic runs, read-min-blocks must equal read-max-blocks.")
        if args.sleep_min_ms != args.sleep_max_ms:
            raise SystemExit("For deterministic runs, sleep-min-ms must equal sleep-max-ms.")

    print("=== KV Cache PCIe Simulation ===")
    print(f"Mode: {args.mode}")
    print(f"Clients: {args.num_clients}, dtype: {DTYPE_MAP[args.dtype]}, pin: {args.pin}")
    print(
        f"Blocks: {args.num_blocks}, block: {args.block_mb:.2f} MB, "
        f"read range: [{args.read_min_blocks}, {args.read_max_blocks}] blocks"
    )
    print(f"PCIe utilization target: {args.pcie_util:.1f}%")
    print(f"Sleep jitter: {args.sleep_min_ms:.2f}~{args.sleep_max_ms:.2f} ms")
    print()

    mp.spawn(run_client, args=(args,), nprocs=args.num_clients, join=True)


if __name__ == "__main__":
    main()

# Example:
# python pcie_load_runner.py --mode baseline --pin
# python pcie_load_runner.py --mode optimized --pin
