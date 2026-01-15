import random
import time

import torch


class PcieLoadRequest:
    def __init__(self, owner_rank, num_blocks, indices_cpu):
        self.owner_rank = owner_rank
        self.num_blocks = num_blocks
        self.indices_cpu = indices_cpu


def compute_sleep_seconds(h2d_seconds, util_ratio, base_sleep_ms):
    if util_ratio <= 0:
        return base_sleep_ms / 1e3
    target_total = h2d_seconds / util_ratio
    extra_sleep = max(0.0, target_total - h2d_seconds)
    return (base_sleep_ms / 1e3) + extra_sleep


def maybe_sleep(h2d_seconds, util_ratio, sleep_min_ms, sleep_max_ms, rng):
    if sleep_max_ms < sleep_min_ms:
        base_sleep_ms = sleep_min_ms
    else:
        base_sleep_ms = sleep_min_ms if rng is None else rng.uniform(sleep_min_ms, sleep_max_ms)
    sleep_seconds = compute_sleep_seconds(h2d_seconds, util_ratio, base_sleep_ms)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


class PcieLoadProcess:
    def __init__(
        self,
        client,
        load_manager,
        block_bytes,
        util_ratio,
        warmup,
        iters,
        sleep_min_ms,
        sleep_max_ms,
        rng,
    ):
        self.client = client
        self.load_manager = load_manager
        self.block_bytes = block_bytes
        self.util_ratio = util_ratio
        self.warmup = warmup
        self.iters = iters
        self.sleep_min_ms = sleep_min_ms
        self.sleep_max_ms = sleep_max_ms
        self.rng = rng

    def run(self):
        per_iter_gbps = []
        for it in range(self.warmup + self.iters):
            request = self.client.build_request()
            if request.num_blocks == 0:
                if it >= self.warmup:
                    per_iter_gbps.append(0.0)
                maybe_sleep(0.0, self.util_ratio, self.sleep_min_ms, self.sleep_max_ms, self.rng)
                continue

            bytes_moved = request.num_blocks * self.block_bytes
            h2d_seconds = self.load_manager.load(request.indices_cpu, self.client.device_id)
            if it >= self.warmup:
                per_iter_gbps.append(self._gbps(bytes_moved, h2d_seconds))
            maybe_sleep(h2d_seconds, self.util_ratio, self.sleep_min_ms, self.sleep_max_ms, self.rng)
        return per_iter_gbps

    @staticmethod
    def _gbps(bytes_moved, seconds):
        return (bytes_moved / 1e9) / seconds if seconds > 0 else float("inf")


class PcieLoadClient:
    def __init__(
        self,
        rank,
        device_id,
        num_blocks,
        read_min_blocks,
        read_max_blocks,
        util_ratio,
        seed,
        randomize,
        load_manager,
        process=None,
    ):
        self.rank = rank
        self.device_id = device_id
        self.num_blocks = num_blocks
        self.read_min_blocks = read_min_blocks
        self.read_max_blocks = read_max_blocks
        self.util_ratio = util_ratio
        self.randomize = randomize
        self.rng = random.Random(seed + rank)
        self.iteration = 0
        self.load_manager = load_manager
        self.process = process
        torch.manual_seed(seed + rank)

    def build_request(self):
        if self.util_ratio <= 0:
            return PcieLoadRequest(self.rank, 0, torch.empty((0,), dtype=torch.int64, device="cpu"))
        if self.randomize:
            num_blocks = self.rng.randint(self.read_min_blocks, self.read_max_blocks)
            selected = self.rng.sample(range(self.num_blocks), num_blocks)
            indices = torch.tensor(selected, dtype=torch.int64, device="cpu")
        else:
            num_blocks = self.read_min_blocks
            start = (self.iteration * num_blocks) % self.num_blocks
            indices = torch.arange(start, start + num_blocks, dtype=torch.int64, device="cpu") % self.num_blocks
            self.iteration += 1
        return PcieLoadRequest(self.rank, num_blocks, indices)

    def run(self):
        return self.process.run()
