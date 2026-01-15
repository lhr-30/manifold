#!/usr/bin/env python3

import torch

class LoadManager:
    def load(self, indices_cpu):  # pragma: no cover - interface
        raise NotImplementedError


class BaselineLoadManager(LoadManager):
    def __init__(self, host_cache, gpu_cache, stream, pin, device):
        self.host_cache = host_cache
        self.gpu_cache = gpu_cache
        self.stream = stream
        self.pin = pin
        self.device = device
        self.copy_start = torch.cuda.Event(enable_timing=True)
        self.copy_end = torch.cuda.Event(enable_timing=True)

    def load(self, indices_cpu):
        with torch.cuda.stream(self.stream):
            self.copy_start.record(self.stream)
            host_slice = self.host_cache.index_select(0, indices_cpu)
            dev_slice = host_slice.to(device=self.device, non_blocking=self.pin)
            self.gpu_cache.index_copy_(0, indices_cpu.to(device=self.device), dev_slice)
            self.copy_end.record(self.stream)
        self.stream.synchronize()
        return self.copy_start.elapsed_time(self.copy_end) / 1e3


class OptimizedLoadManager(LoadManager):
    def __init__(
        self,
        host_cache,
        gpu_cache,
        block_elems,
        stream,
        pin,
        device,
        device_handlers,
    ):
        self.host_cache = host_cache
        self.gpu_cache = gpu_cache
        self.block_elems = block_elems
        self.stream = stream
        self.pin = pin
        self.device = device
        self.device_handlers = device_handlers
        self.copy_start = torch.cuda.Event(enable_timing=True)
        self.copy_end = torch.cuda.Event(enable_timing=True)
        self.reduce_start = torch.cuda.Event(enable_timing=True)
        self.reduce_end = torch.cuda.Event(enable_timing=True)

    def load(self, indices_cpu):
        num_blocks = indices_cpu.numel()
        if num_blocks == 0:
            return 0.0

        positions = torch.arange(num_blocks, dtype=torch.int64, device="cpu")
        positions_rank = positions
        indices_rank = indices_cpu

        with torch.cuda.stream(self.stream):
            self.copy_start.record(self.stream)
            host_slice = self.host_cache.index_select(0, indices_rank)
            dev_slice = host_slice.to(device=self.device, non_blocking=self.pin)
            self.gpu_cache.index_copy_(0, indices_rank.to(device=self.device), dev_slice)
            self.copy_end.record(self.stream)
        self.stream.synchronize()
        h2d_seconds = self.copy_start.elapsed_time(self.copy_end) / 1e3
        return h2d_seconds
