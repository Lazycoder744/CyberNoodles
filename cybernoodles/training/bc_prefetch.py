import queue
import threading
import time
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LoaderProfileSnapshot:
    shard_count: int
    batch_count: int
    sample_count: int
    shard_load_s: float
    shard_shuffle_s: float
    batch_pack_s: float
    queue_wait_s: float
    h2d_submit_s: float
    h2d_copy_s: float
    compute_s: float


class LoaderProfile:
    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._shard_count = 0
        self._batch_count = 0
        self._sample_count = 0
        self._shard_load_s = 0.0
        self._shard_shuffle_s = 0.0
        self._batch_pack_s = 0.0
        self._queue_wait_s = 0.0
        self._h2d_submit_s = 0.0
        self._h2d_copy_s = 0.0
        self._compute_s = 0.0
        self._copy_events = []
        self._compute_events = []

    def add_shard(self, sample_count, load_s, shuffle_s=0.0):
        if not self.enabled:
            return
        with self._lock:
            self._shard_count += 1
            self._shard_load_s += float(load_s)
            self._shard_shuffle_s += float(shuffle_s)

    def add_batch_pack(self, duration_s):
        if not self.enabled:
            return
        with self._lock:
            self._batch_pack_s += float(duration_s)

    def add_consumed_batch(self, batch_samples):
        if not self.enabled:
            return
        with self._lock:
            self._batch_count += 1
            self._sample_count += max(0, int(batch_samples))

    def add_queue_wait(self, duration_s):
        if not self.enabled:
            return
        with self._lock:
            self._queue_wait_s += float(duration_s)

    def add_h2d_submit(self, duration_s):
        if not self.enabled:
            return
        with self._lock:
            self._h2d_submit_s += float(duration_s)

    def add_h2d_event(self, start_event, end_event):
        if not self.enabled:
            return
        with self._lock:
            self._copy_events.append((start_event, end_event))

    def add_compute_time(self, duration_s):
        if not self.enabled:
            return
        with self._lock:
            self._compute_s += float(duration_s)

    def add_compute_event(self, start_event, end_event):
        if not self.enabled:
            return
        with self._lock:
            self._compute_events.append((start_event, end_event))

    def finalize_cuda(self, device=None):
        if not self.enabled:
            return

        with self._lock:
            copy_events = self._copy_events
            compute_events = self._compute_events
            self._copy_events = []
            self._compute_events = []

        if not copy_events and not compute_events:
            return

        torch.cuda.synchronize(device)
        copy_total = 0.0
        for start_event, end_event in copy_events:
            copy_total += float(start_event.elapsed_time(end_event)) / 1000.0

        compute_total = 0.0
        for start_event, end_event in compute_events:
            compute_total += float(start_event.elapsed_time(end_event)) / 1000.0

        with self._lock:
            self._h2d_copy_s += copy_total
            self._compute_s += compute_total

    def snapshot(self):
        with self._lock:
            return LoaderProfileSnapshot(
                shard_count=self._shard_count,
                batch_count=self._batch_count,
                sample_count=self._sample_count,
                shard_load_s=self._shard_load_s,
                shard_shuffle_s=self._shard_shuffle_s,
                batch_pack_s=self._batch_pack_s,
                queue_wait_s=self._queue_wait_s,
                h2d_submit_s=self._h2d_submit_s,
                h2d_copy_s=self._h2d_copy_s,
                compute_s=self._compute_s,
            )


class ThreadedPrefetchIterator:
    _END = object()

    def __init__(self, iterator_factory, max_prefetch=1, profile=None, thread_name="bc-loader"):
        self._iterator_factory = iterator_factory
        self._queue = queue.Queue(maxsize=max(1, int(max_prefetch)))
        self._profile = profile
        self._thread_name = thread_name
        self._thread = None
        self._stop_event = threading.Event()

    def __enter__(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._worker_main, name=self._thread_name, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _put_with_stop(self, item):
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _worker_main(self):
        iterator = None
        try:
            iterator = iter(self._iterator_factory())
            for item in iterator:
                if self._stop_event.is_set():
                    break
                if not self._put_with_stop(item):
                    break
        except BaseException as exc:  # pragma: no cover - error path
            self._put_with_stop(exc)
        finally:
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            self._put_with_stop(self._END)

    def __iter__(self):
        return self

    def __next__(self):
        wait_started = time.perf_counter()
        item = self._queue.get()
        if self._profile is not None:
            self._profile.add_queue_wait(time.perf_counter() - wait_started)
        if item is self._END:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


class CudaBatchPrefetcher:
    def __init__(self, batch_iterable, device, dtype=torch.float32, non_blocking=True, profile=None):
        self._batch_iterable = batch_iterable
        self._batch_iterator = None
        self._device = device
        self._dtype = dtype
        self._non_blocking = bool(non_blocking)
        self._profile = profile
        self._stream = None
        self._next_batch = None
        self._next_copy_events = None

    def __enter__(self):
        self._batch_iterator = iter(self._batch_iterable)
        self._stream = torch.cuda.Stream(device=self._device)
        self._preload_next()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _preload_next(self):
        try:
            cpu_batch = next(self._batch_iterator)
        except StopIteration:
            self._next_batch = None
            self._next_copy_events = None
            return

        if isinstance(cpu_batch, dict):
            cpu_x = cpu_batch["x"]
            cpu_y = cpu_batch["y"]
            batch_extra = {key: value for key, value in cpu_batch.items() if key not in ("x", "y")}
        else:
            cpu_x, cpu_y = cpu_batch[:2]
            batch_extra = tuple(cpu_batch[2:])

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        submit_started = time.perf_counter()
        with torch.cuda.stream(self._stream):
            start_event.record(self._stream)
            batch_x = cpu_x.to(device=self._device, dtype=self._dtype, non_blocking=self._non_blocking)
            batch_y = cpu_y.to(device=self._device, dtype=self._dtype, non_blocking=self._non_blocking)
            end_event.record(self._stream)

        if self._profile is not None:
            self._profile.add_h2d_submit(time.perf_counter() - submit_started)
            self._profile.add_h2d_event(start_event, end_event)
        if isinstance(cpu_batch, dict):
            self._next_batch = {"x": batch_x, "y": batch_y, **batch_extra}
        else:
            self._next_batch = (batch_x, batch_y, *batch_extra)
        self._next_copy_events = (start_event, end_event)

    def __iter__(self):
        return self

    def __next__(self):
        if self._next_batch is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream(self._device)
        current_stream.wait_stream(self._stream)
        if isinstance(self._next_batch, dict):
            batch_x = self._next_batch["x"]
            batch_y = self._next_batch["y"]
        else:
            batch_x, batch_y = self._next_batch[:2]
        batch_x.record_stream(current_stream)
        batch_y.record_stream(current_stream)
        if self._profile is not None:
            self._profile.add_consumed_batch(batch_x.shape[0])
        current_batch = self._next_batch
        self._preload_next()
        return current_batch

    def close(self):
        self._next_batch = None
        self._next_copy_events = None
        if self._batch_iterator is not None:
            close = getattr(self._batch_iterator, "close", None)
            if callable(close):
                close()
            self._batch_iterator = None
