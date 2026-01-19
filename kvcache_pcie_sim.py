#!/usr/bin/env python3

import torch

class LoadManager:
    def load(self, indices_cpu, indices_gpu, device_id):  # pragma: no cover - interface
        raise NotImplementedError


class BaselineLoadManager(LoadManager):
    def __init__(self, host_cache, gpu_cache, stream, pin, device):
        self.host_cache = host_cache
        self.gpu_cache = gpu_cache
        self.stream = stream
        self.device = device
        self.pin = pin
        self.copy_start = torch.cuda.Event(enable_timing=True)
        self.copy_end = torch.cuda.Event(enable_timing=True)

    def load(self, indices_cpu, indices_gpu, device_id):
        # print(f"Loading to device {device} indices {indices_cpu}")
        with torch.cuda.stream(self.stream):
            self.copy_start.record(self.stream)
            for cpu_idx, gpu_idx in zip(indices_cpu, indices_gpu):
                self.gpu_cache[gpu_idx].copy_(self.host_cache[cpu_idx], non_blocking=self.pin)
            self.copy_end.record(self.stream)
        self.stream.synchronize()
        load_time = self.copy_start.elapsed_time(self.copy_end) / 1e3
        bytes_per_block = self.host_cache[0].numel() * self.host_cache[0].element_size()
        load_data_size = len(indices_cpu) * bytes_per_block / 1e6
        print(f"device = {self.device}, load time = {load_time} seconds, load data size = {load_data_size} MB, Bandwidth = {load_data_size / 1e3 / load_time } GB/s")
        return self.copy_start.elapsed_time(self.copy_end) / 1e3


class OptimizedLoadManager(LoadManager):
    def __init__(
        self,
        host_caches,
        block_elems,
        pin,
        devices,
        streams,
        gpu_caches,
    ):
        self.host_caches = host_caches
        self.block_elems = block_elems
        self.pin = pin
        self.devices = devices
        self.streams = streams
        self.gpu_caches = gpu_caches
        self.copy_start = [torch.cuda.Event(enable_timing=True) for _ in devices]
        self.copy_end = [torch.cuda.Event(enable_timing=True) for _ in devices]

    def load(self, indices_cpu, indices_gpu, device_id):
        num_blocks = indices_cpu.numel()
        if num_blocks == 0:
            return 0.0

        positions = torch.arange(num_blocks, dtype=torch.int64, device="cpu")
        load_buffers = []
        for idx, device in enumerate(self.devices):
            load_buffer = torch.zeros(
                (num_blocks, self.block_elems), dtype=self.host_caches[device_id].dtype, device=device
            )
            stream = self.streams[idx]
            with torch.cuda.stream(stream):
                self.copy_start[idx].record(stream)
                host_slice = self.host_caches[device_id].index_select(0, indices_cpu)
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
            (num_blocks, self.block_elems), dtype=self.host_caches[device_id].dtype, device=target_device
        )
        for idx, device in enumerate(self.devices):
            if device == target_device:
                reduce_buffer.add_(load_buffers[idx])
            else:
                reduce_buffer.add_(load_buffers[idx].to(device=target_device))

        self.gpu_caches[device_id].index_copy_(
            0, indices_gpu, reduce_buffer
        )
        return copy_seconds
