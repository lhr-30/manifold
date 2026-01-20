#!/usr/bin/env python3

from dataclasses import dataclass
import threading
import time
from typing import List, Sequence

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
        request_queue,
        task_queues,
        completion_queue,
        reduce_queues,
        num_clients,
    ):
        self.request_queue = request_queue
        self.task_queues = task_queues
        self.completion_queue = completion_queue
        self.reduce_queues = reduce_queues
        self.num_clients = num_clients

    def serve(self):
        while True:
            request = self.request_queue.get()
            if request is None:
                break
            worker_ranks = self._dispatch_tasks(request)
            self._wait_for_completion(request.request_id, worker_ranks)
            self.reduce_queues[request.owner_rank].put(
                ReduceCommand(
                    request_id=request.request_id,
                    owner_rank=request.owner_rank,
                    action="reduce",
                    contributors=worker_ranks,
                )
            )
            for rank in worker_ranks:
                if rank == request.owner_rank:
                    continue
                self.task_queues[rank].put(
                    ControlCommand(
                        request_id=request.request_id,
                        owner_rank=request.owner_rank,
                        action="copy",
                    )
                )

    def _dispatch_tasks(self, request):
        indices_cpu = list(request.indices_cpu)
        indices_gpu = list(request.indices_gpu)
        splits = _split_indices(len(indices_cpu), self.num_clients)
        worker_ranks = []
        for rank, (start, end) in enumerate(splits):
            if start == end:
                continue
            task = LoadTask(
                request_id=request.request_id,
                owner_rank=request.owner_rank,
                worker_rank=rank,
                indices_cpu=indices_cpu[start:end],
                indices_gpu=indices_gpu[start:end],
            )
            self.task_queues[rank].put(task)
            worker_ranks.append(rank)
        return worker_ranks

    def _wait_for_completion(self, request_id, worker_ranks):
        pending = set(worker_ranks)
        while pending:
            completion = self.completion_queue.get()
            if completion.request_id != request_id:
                continue
            pending.discard(completion.worker_rank)


@dataclass(frozen=True)
class LoadRequest:
    request_id: int
    owner_rank: int
    indices_cpu: Sequence[int]
    indices_gpu: Sequence[int]


@dataclass(frozen=True)
class LoadTask:
    request_id: int
    owner_rank: int
    worker_rank: int
    indices_cpu: Sequence[int]
    indices_gpu: Sequence[int]


@dataclass(frozen=True)
class TaskCompletion:
    request_id: int
    worker_rank: int


@dataclass(frozen=True)
class ReduceCommand:
    request_id: int
    owner_rank: int
    action: str
    contributors: List[int]


@dataclass(frozen=True)
class ControlCommand:
    request_id: int
    owner_rank: int
    action: str


class OptimizedLoadWorker:
    def __init__(self, host_cache, gpu_cache, stream, pin, device):
        self.host_cache = host_cache
        self.gpu_cache = gpu_cache
        self.stream = stream
        self.pin = pin
        self.device = device
        self.copy_start = torch.cuda.Event(enable_timing=True)
        self.copy_end = torch.cuda.Event(enable_timing=True)

    def execute_task(self, task: LoadTask):
        if not task.indices_cpu:
            return 0.0
        torch.cuda.set_device(self.device)
        indices_cpu = torch.tensor(task.indices_cpu, dtype=torch.int64, device="cpu")
        indices_gpu = torch.tensor(task.indices_gpu, dtype=torch.int64, device=self.device)
        with torch.cuda.stream(self.stream):
            self.copy_start.record(self.stream)
            host_slice = self.host_cache.index_select(0, indices_cpu)
            dev_slice = host_slice.to(device=self.device, non_blocking=self.pin)
            self.gpu_cache.index_copy_(0, indices_gpu, dev_slice)
            self.copy_end.record(self.stream)
        self.stream.synchronize()
        return self.copy_start.elapsed_time(self.copy_end) / 1e3

    def handle_control(self, command: ControlCommand):
        _ = command


class OptimizedLoadManagerClient(LoadManager):
    def __init__(
        self,
        rank,
        request_queue,
        task_queue,
        completion_queue,
        reduce_queue,
        worker: OptimizedLoadWorker,
    ):
        self.rank = rank
        self.request_queue = request_queue
        self.task_queue = task_queue
        self.completion_queue = completion_queue
        self.reduce_queue = reduce_queue
        self.worker = worker
        self._request_counter = 0
        self._stop_event = threading.Event()
        self._task_thread = threading.Thread(target=self._task_loop, daemon=True)
        self._task_thread.start()

    def load(self, indices_cpu, indices_gpu, device_id):
        _ = device_id
        request_id = (self.rank << 32) | self._request_counter
        self._request_counter += 1
        request = LoadRequest(
            request_id=request_id,
            owner_rank=self.rank,
            indices_cpu=list(indices_cpu),
            indices_gpu=list(indices_gpu),
        )
        self.request_queue.put(request)
        start_time = time.perf_counter()
        while True:
            command = self.reduce_queue.get()
            if command is None:
                break
            if command.request_id == request_id and command.action == "reduce":
                break
        return time.perf_counter() - start_time

    def _task_loop(self):
        while not self._stop_event.is_set():
            task = self.task_queue.get()
            if task is None:
                break
            if isinstance(task, LoadTask):
                self.worker.execute_task(task)
                self.completion_queue.put(
                    TaskCompletion(request_id=task.request_id, worker_rank=self.rank)
                )
            elif isinstance(task, ControlCommand):
                self.worker.handle_control(task)

    def shutdown(self):
        self._stop_event.set()
        self.task_queue.put(None)
        self._task_thread.join(timeout=5)


def _split_indices(total_blocks: int, num_clients: int):
    base, extra = divmod(total_blocks, num_clients)
    splits = []
    start = 0
    for idx in range(num_clients):
        end = start + base + (1 if idx < extra else 0)
        splits.append((start, end))
        start = end
    return splits
