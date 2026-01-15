#!/usr/bin/env python3

import torch

class LoadManager:
    def load(self, indices_cpu, device_id):  # pragma: no cover - interface
        raise NotImplementedError


class BaselineLoadManager(LoadManager):
    def __init__(self, host_cache, gpu_caches, streams, pin, devices):
        self.host_cache = host_cache
        self.gpu_caches = gpu_caches
        self.streams = streams
        self.pin = pin
        self.devices = devices
        self.copy_start = [torch.cuda.Event(enable_timing=True) for _ in devices]
        self.copy_end = [torch.cuda.Event(enable_timing=True) for _ in devices]

    def load(self, indices_cpu, device_id):
        stream = self.streams[device_id]
        device = self.devices[device_id]
        with torch.cuda.stream(stream):
            self.copy_start[device_id].record(stream)
            host_slice = self.host_cache.index_select(0, indices_cpu)
            dev_slice = host_slice.to(device=device, non_blocking=self.pin)
            self.gpu_caches[device_id].index_copy_(0, indices_cpu.to(device=device), dev_slice)
            self.copy_end[device_id].record(stream)
        stream.synchronize()
        return self.copy_start[device_id].elapsed_time(self.copy_end[device_id]) / 1e3


class OptimizedLoadManager(LoadManager):
    def __init__(
        self,
        host_cache,
        block_elems,
        pin,
        devices,
        streams,
        gpu_caches,
    ):
        self.host_cache = host_cache
        self.block_elems = block_elems
        self.pin = pin
        self.devices = devices
        self.streams = streams
        self.gpu_caches = gpu_caches
        self.copy_start = [torch.cuda.Event(enable_timing=True) for _ in devices]
        self.copy_end = [torch.cuda.Event(enable_timing=True) for _ in devices]

    def load(self, indices_cpu, device_id):
        num_blocks = indices_cpu.numel()
        if num_blocks == 0:
            return 0.0

        positions = torch.arange(num_blocks, dtype=torch.int64, device="cpu")
        load_buffers = []
        for idx, device in enumerate(self.devices):
            load_buffer = torch.zeros(
                (num_blocks, self.block_elems), dtype=self.host_cache.dtype, device=device
            )
            stream = self.streams[idx]
            with torch.cuda.stream(stream):
                self.copy_start[idx].record(stream)
                host_slice = self.host_cache.index_select(0, indices_cpu)
                dev_slice = host_slice.to(device=device, non_blocking=self.pin)
                load_buffer.index_copy_(0, positions.to(device=device), dev_slice)
                self.copy_end[idx].record(stream)
            load_buffers.append(load_buffer)

        for stream in self.streams:
            stream.synchronize()

        copy_seconds = max(
            self.copy_start[idx].elapsed_time(self.copy_end[idx]) for idx in range(len(self.devices))
        ) / 1e3

        target_device = self.devices[device_id]
        reduce_buffer = torch.zeros(
            (num_blocks, self.block_elems), dtype=self.host_cache.dtype, device=target_device
        )
        for idx, device in enumerate(self.devices):
            if device == target_device:
                reduce_buffer.add_(load_buffers[idx])
            else:
                reduce_buffer.add_(load_buffers[idx].to(device=target_device))

        self.gpu_caches[device_id].index_copy_(
            0, indices_cpu.to(device=target_device), reduce_buffer
        )
        return copy_seconds
