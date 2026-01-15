import random

import torch


class PcieLoadRequest:
    def __init__(self, owner_rank, num_blocks, indices_cpu):
        self.owner_rank = owner_rank
        self.num_blocks = num_blocks
        self.indices_cpu = indices_cpu


class PcieLoadClient:
    def __init__(self, rank, num_blocks, read_min_blocks, read_max_blocks, util_ratio, seed):
        self.rank = rank
        self.num_blocks = num_blocks
        self.read_min_blocks = read_min_blocks
        self.read_max_blocks = read_max_blocks
        self.util_ratio = util_ratio
        random.seed(seed + rank)
        torch.manual_seed(seed + rank)

    def build_request(self):
        if self.util_ratio <= 0:
            return PcieLoadRequest(self.rank, 0, torch.empty((0,), dtype=torch.int64, device="cpu"))
        num_blocks = random.randint(self.read_min_blocks, self.read_max_blocks)
        selected = random.sample(range(self.num_blocks), num_blocks)
        indices = torch.tensor(selected, dtype=torch.int64, device="cpu")
        return PcieLoadRequest(self.rank, num_blocks, indices)
