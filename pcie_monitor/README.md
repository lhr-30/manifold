# PCIe Monitor

This directory implements a lightweight GPU PCIe traffic monitor. The core idea
is to use LD_PRELOAD to hook `cudaMemcpyAsync`, record H2D/D2H transfer sizes
asynchronously into shared memory, and have a user-space script poll and log.

## Design

- **Interception**: hook both `cudaMemcpy` (sync) and `cudaMemcpyAsync` in `gpu_spy.cpp`, only counting H2D/D2H.
- **Low overhead**: create a `cudaEvent` per copy and poll completion asynchronously.
- **Shared memory**: write per-GPU pending/total byte counters to `/dev/shm/gpu_pcie_scoreboard`.
- **Threshold filter**: ignore small packets via `MIN_BYTES_TO_TRACK` to reduce CPU overhead.
- **Monitor**: `pci_monitor.py` polls shared memory, prints, and appends to `pcie_traffic.csv`.
- **Cleanup**: delete shm on exit to avoid accumulated counters.

## Files

- `gpu_spy.cpp`: LD_PRELOAD hook implementation (sync + async copies)
- `build_gpu_spy.sh`: build `libgpu_spy.so`
- `pci_monitor.py`: shared memory reader + print/logging
- `bench_token.py`: token-based micro workload
- `sweep_analysis.py`: sweep overhead across token sizes
- `auto_tune_min_bytes.py`: find smallest token with overhead < 3% and update threshold

## Quick Start

1) Build the hook library

```bash
cd pcie_monitor
bash build_gpu_spy.sh
```

2) Launch the target program (with LD_PRELOAD)

```bash
LD_PRELOAD=$PWD/libgpu_spy.so python bench_token.py --tokens 128 --hidden 5120 --iters 1000
```

3) Monitor PCIe traffic

```bash
python pci_monitor.py
```

Output is appended to `pcie_traffic.csv`.

## Auto-Tune (MIN_BYTES_TO_TRACK)

Pick the smallest token whose overhead < 3%, update `gpu_spy.cpp`, and rebuild:

```bash
python auto_tune_min_bytes.py --hidden 5120 --tokens 1,8,32,128,512,2048
```

- Generates/updates `token_sweep_results.csv`
- Updates `#define MIN_BYTES_TO_TRACK ...`
- Reruns sweep by default for verification

To skip rerun:

```bash
python auto_tune_min_bytes.py --hidden 5120 --tokens 1,8,32,128,512,2048 --no-rerun
```

## Manual Sweep

```bash
python sweep_analysis.py --hidden 5120 --tokens 1,8,32,128,512,2048
```

## Notes

- Shared memory path: `/dev/shm/gpu_pcie_scoreboard`
- If the monitor starts first, it will wait for shm to be created
- Too small `MIN_BYTES_TO_TRACK` increases CPU overhead
- Default monitors the first 8 GPUs; adjust in `pci_monitor.py`
