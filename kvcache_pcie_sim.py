#!/usr/bin/env python3

from dataclasses import dataclass
import logging
import threading
import time
from typing import Dict, List, Sequence

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

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
        logger.info(
            "device = %s, load time = %s seconds, load data size = %s MB, Bandwidth = %s GB/s",
            self.device,
            load_time,
            load_data_size,
            load_data_size / 1e3 / load_time,
        )
        return self.copy_start.elapsed_time(self.copy_end) / 1e3


class OptimizedLoadManager(LoadManager):
    def __init__(
        self,
        request_queue,
        task_queues,
        reduce_queues,
        complete_queue,
        num_clients,
        num_requests,
    ):
        self.request_queue = request_queue
        self.task_queues = task_queues
        self.reduce_queues = reduce_queues
        self.complete_queue = complete_queue
        self.num_clients = num_clients
        self.num_requests = num_requests
        self.counter = 0

    def serve(self):
        while True:
            request = self.request_queue.get()
            if request is None:
                logger.debug("Manager received shutdown signal.")
                break
            logger.debug(
                "Manager received load request id %s from owner rank %s",
                request.request_id,
                request.owner_rank,
            )
            worker_ranks = self._dispatch_tasks(request)
            self.reduce_queues[request.owner_rank].put(
                ReduceCommand(
                    request_id=request.request_id,
                    owner_rank=request.owner_rank,
                    action="reduce",
                    contributors=worker_ranks,
                )
            )
            while True:
                completion = self.complete_queue.get()
                if completion is None:
                    return
                if (
                    completion.request_id == request.request_id
                    and completion.owner_rank == request.owner_rank
                ):
                    break
            logger.debug(
                "[Manager] completed load request id %s from owner rank %s",
                request.request_id,
                request.owner_rank,
            )
            self.counter += 1
            if self.counter >= self.num_requests:
                logger.debug("Manager processed all requests, shutting down.")
                for i in range(self.num_clients):
                    self.reduce_queues[i].put(None)
                    self.task_queues[i].put(None)
                break

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
                host_cache_rank=request.owner_rank,
                indices_cpu=indices_cpu[start:end],
                indices_gpu=indices_gpu[start:end],
            )
            self.task_queues[rank].put(task)
            worker_ranks.append(rank)
        return worker_ranks

@dataclass
class LoadRequest:
    request_id: int
    owner_rank: int
    indices_cpu: Sequence[int]
    indices_gpu: Sequence[int]

@dataclass
class LoadTask:
    request_id: int
    owner_rank: int
    worker_rank: int
    host_cache_rank: int
    indices_cpu: Sequence[int]
    indices_gpu: Sequence[int]

@dataclass
class ReduceCommand:
    request_id: int
    owner_rank: int
    action: str
    contributors: List[int]

@dataclass
class LoadComplete:
    request_id: int
    owner_rank: int

from contextlib import contextmanager


class OptimizedLoadWorker:
    def __init__(self, rank, host_cache, host_caches, gpu_cache, stream, pin, device):
        self.rank = rank
        self.host_cache = host_cache
        self.host_caches = host_caches
        self.gpu_cache = gpu_cache
        self.stream = stream
        self.pin = pin
        self.device = device
        self.copy_start = torch.cuda.Event(enable_timing=True)
        self.copy_end = torch.cuda.Event(enable_timing=True)
        self._task_indices: Dict[int, torch.Tensor] = {}
        self._dist_lock = threading.Lock()
        self._cpu_group = None
        if dist.is_initialized():
            self._cpu_group = dist.new_group(backend="gloo")

    @contextmanager
    def dist_lock(self, op: str, req: int, peer: int = -1):
        tid = threading.current_thread().name
        t0 = time.perf_counter()
        logger.debug("[R%s][%s] ACQUIRE_WAIT op=%s req=%s peer=%s", self.rank, tid, op, req, peer)
        self._dist_lock.acquire()
        t1 = time.perf_counter()
        logger.debug(
            "[R%s][%s] ACQUIRE_OK   op=%s req=%s peer=%s wait_ms=%.3f",
            self.rank,
            tid,
            op,
            req,
            peer,
            (t1 - t0) * 1e3,
        )
        try:
            yield
        finally:
            t2 = time.perf_counter()
            self._dist_lock.release()
            logger.debug(
                "[R%s][%s] RELEASE     op=%s req=%s peer=%s hold_ms=%.3f",
                self.rank,
                tid,
                op,
                req,
                peer,
                (t2 - t1) * 1e3,
            )

    def execute_task(self, task: LoadTask):
        if not task.indices_cpu:
            return 0.0
        torch.cuda.set_device(self.device)
        indices_cpu = task.indices_cpu
        indices_gpu = task.indices_gpu
        self._task_indices[task.request_id] = indices_gpu
        host_cache = self.host_caches[task.host_cache_rank]
        with torch.cuda.stream(self.stream):
            self.copy_start.record(self.stream)
            for cpu_idx, gpu_idx in zip(indices_cpu, indices_gpu):
                self.gpu_cache[gpu_idx].copy_(host_cache[cpu_idx], non_blocking=self.pin)
            self.copy_end.record(self.stream)
        self.stream.synchronize()
        logger.info(f"load time at rank {self.rank} for request id {task.request_id}: {self.copy_start.elapsed_time(self.copy_end) / 1e3:.3f} seconds")
        if task.owner_rank != self.rank:
            logger.debug(
                "Worker rank %s, owner rank %s, send shards for request id %s",
                self.rank,
                task.owner_rank,
                task.request_id,
            )
            self._send_shard(task.request_id, task.owner_rank)
        return self.copy_start.elapsed_time(self.copy_end) / 1e3

    def reduce_from_contributors(self, command: ReduceCommand):
        reduce_start = time.perf_counter()
        torch.cuda.set_device(self.device)
        logger.info(f"torch init time at rank {self.rank}: {time.perf_counter() - reduce_start:.3f} seconds")
        self._task_indices.pop(command.request_id, None)
        if not command.contributors:
            return
        pending = []
        pending_meta = []
        for rank in command.contributors:
            if rank == command.owner_rank:
                logger.debug(
                    "Worker rank %s, skipping receiving shard from owner rank %s for request id %s",
                    self.rank,
                    rank,
                    command.request_id,
                )
                continue
            logger.debug(
                "Worker rank %s, preparing to receive shard from contributor rank %s for request id %s",
                self.rank,
                rank,
                command.request_id,
            )
            # with self._dist_lock:
            recv_start = time.perf_counter()
            with self.dist_lock("recv_size", req=command.request_id, peer=rank):
                size = torch.empty((1,), dtype=torch.int64, device="cpu")
                dist.recv(size, src=rank, group=self._cpu_group)
            logger.info(f"recv size time at rank {self.rank} from rank {rank}: {time.perf_counter() - recv_start:.3f} seconds")
            
            logger.debug(
                "Worker rank %s, receiving shard from contributor rank %s for request id %s, size = %s",
                self.rank,
                rank,
                command.request_id,
                size.item(),
            )

            num_indices = int(size.item())
            if num_indices == 0:
                continue
            indices = torch.empty((num_indices,), dtype=torch.int64, device=self.device)
            data = torch.empty(
                (num_indices, self.gpu_cache.shape[1]),
                dtype=self.gpu_cache.dtype,
                device=self.device,
            )
            pending.append(dist.P2POp(dist.irecv, indices, peer=rank))
            pending.append(dist.P2POp(dist.irecv, data, peer=rank))
            pending_meta.append((indices, data))
        if pending:
            # with self._dist_lock:
            recv_payload_start = time.perf_counter()
            with self.dist_lock("recv_payload", req=command.request_id, peer=-1):
                for work in dist.batch_isend_irecv(pending):
                    work.wait()
            logger.info(f"recv payload time at rank {self.rank}: {time.perf_counter() - recv_payload_start:.3f} seconds")
            
            logger.debug(
                "Worker rank %s, finished receiving shards for request id %s",
                self.rank,
                command.request_id,
            )
            index_copy_start = time.perf_counter()
            for indices, data in pending_meta:
                self.gpu_cache.index_copy_(0, indices, data)
            logger.info(f"index copy time at rank {self.rank}: {time.perf_counter() - index_copy_start:.3f} seconds")

    def _send_shard(self, request_id: int, owner_rank: int):
        torch.cuda.set_device(self.device)
        indices = self._task_indices.pop(request_id, None)
        if indices is None:
            logger.warning("!!!!!")
            # with self._dist_lock:
            with self.dist_lock("send_size", req=request_id, peer=owner_rank):
                size = torch.zeros((1,), dtype=torch.int64, device="cpu")
                dist.send(size, dst=owner_rank, group=self._cpu_group)
            return
        indices = torch.tensor(indices, dtype=torch.int64, device=self.device)
        num_indices = int(indices.numel())
        logger.debug(
            "Worker rank %s, sending shard to owner rank %s for request id %s, num indices = %s",
            self.rank,
            owner_rank,
            request_id,
            num_indices,
        )
        # with self._dist_lock:
        send_start = time.perf_counter()
        with self.dist_lock("send_size", req=request_id, peer=owner_rank):
            size = torch.tensor([num_indices], dtype=torch.int64, device="cpu")
            dist.send(size, dst=owner_rank, group=self._cpu_group)
        logger.info(f"send size time at rank {self.rank} to rank {owner_rank}: {time.perf_counter() - send_start:.3f} seconds")
        
        if num_indices == 0:
            return
        data = self.gpu_cache.index_select(0, indices)
        ops = [
            dist.P2POp(dist.isend, indices, peer=owner_rank),
            dist.P2POp(dist.isend, data, peer=owner_rank),
        ]
        # with self._dist_lock:
        with self.dist_lock("send_payload", req=request_id, peer=owner_rank):
            for work in dist.batch_isend_irecv(ops):
                work.wait()
        logger.debug(
            "Finish sending: Worker rank %s, sent shard to owner rank %s for request id %s",
            self.rank,
            owner_rank,
            request_id,
        )


class OptimizedLoadManagerClient(LoadManager):
    def __init__(
        self,
        rank,
        request_queue,
        task_queue,
        reduce_queue,
        complete_queue,
        worker: OptimizedLoadWorker,
    ):
        self.rank = rank
        self.request_queue = request_queue
        self.task_queue = task_queue
        self.reduce_queue = reduce_queue
        self.complete_queue = complete_queue
        self.worker = worker
        self._request_counter = 0
        self._stop_event = threading.Event()
        self._task_thread = threading.Thread(target=self._task_loop, daemon=True)
        self._task_thread.start()

    def load(self, indices_cpu, indices_gpu, device_id):
        _ = device_id
        request_id = self._request_counter + 100000 * self.rank
        self._request_counter += 1
        logger.debug("Client rank %s, submitting load request id %s", self.rank, request_id)
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
            logger.info(f"msg received at client rank {self.rank} for request id {request_id}, receiving time: {time.perf_counter() - start_time:.3f} seconds")
            if command is None:
                break
            if command.request_id == request_id and command.action == "reduce":
                logger.debug(
                    "Client rank %s, processing reduce command for request id %s",
                    self.rank,
                    request_id,
                )
                self.worker.reduce_from_contributors(command)
                break
        self.complete_queue.put(
            LoadComplete(
                request_id=request_id,
                owner_rank=self.rank,
            )
        )
        data_size = len(indices_cpu) * self.worker.host_cache.shape[1] * self.worker.host_cache.element_size() / 1e6
        logger.info(f"client rank {self.rank}, completed load request id {request_id}, time taken: {time.perf_counter() - start_time:.3f} seconds, data size: {data_size} MB, bandwidth: {data_size / 1e3 / (time.perf_counter() - start_time)} GB/s")
        return time.perf_counter() - start_time

    def _task_loop(self):
        while not self._stop_event.is_set():
            task = self.task_queue.get()
            if task is None:
                break
            if isinstance(task, LoadTask):
                logger.debug(
                    "Client rank %s, executing load task for request id %s",
                    self.rank,
                    task.request_id,
                )
                self.worker.execute_task(task)

    def shutdown(self):
        logger.info("[Client] rank %s shutting down.", self.rank)
        self._stop_event.set()
        self.task_queue.put(None)
        self._task_thread.join(timeout=5)

    def wait_for_shutdown(self):
        while True:
            command = self.reduce_queue.get()
            if command is None:
                break
            logger.debug(
                "Client rank %s, skipping unexpected command while waiting for shutdown: %s",
                self.rank,
                command,
            )
        self.shutdown()


def _split_indices(total_blocks: int, num_clients: int):
    base, extra = divmod(total_blocks, num_clients)
    splits = []
    start = 0
    for idx in range(num_clients):
        end = start + base + (1 if idx < extra else 0)
        splits.append((start, end))
        start = end
    return splits
