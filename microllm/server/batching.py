from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from collections.abc import Callable, AsyncIterator
import queue
import threading
import time

from microllm.engine.engine import Engine
from microllm.engine.sampler import SamplingParams


@dataclass(frozen=True, slots=True)
class _QueueItem:
    prompt: str
    sampling_params: SamplingParams
    future: asyncio.Future


@dataclass(frozen=True, slots=True)
class _ModelCall:
    fn: Callable
    args: tuple
    kwargs: dict
    result_queue: queue.Queue | None = None


@dataclass(slots=True)
class BatchingStats:
    total_requests: int = 0
    total_batches: int = 0
    max_observed_batch_size: int = 0

    @property
    def average_batch_size(self) -> float:
        if self.total_batches == 0:
            return 0.0
        return self.total_requests / self.total_batches

    def as_dict(self, queue_depth: int) -> dict:
        return {
            "total_requests": self.total_requests,
            "total_batches": self.total_batches,
            "max_observed_batch_size": self.max_observed_batch_size,
            "average_batch_size": self.average_batch_size,
            "queue_depth": queue_depth,
        }


class BatchingEngine:
    """Async request queue that feeds non-streaming HTTP requests into Engine.generate_batch."""

    def __init__(
        self,
        engine: Engine,
        *,
        max_batch_size: int = 8,
        batch_wait_ms: float = 2.0,
        max_num_batched_tokens: int = 2048,
        max_queue_size: int | None = None,
        max_prompt_chars: int | None = None,
        enable_chunked_prefill: bool = False,
        cache_aware_scheduling: bool = False,
    ):
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if batch_wait_ms < 0:
            raise ValueError("batch_wait_ms must be >= 0")
        if max_queue_size is not None and max_queue_size <= 0:
            raise ValueError("max_queue_size must be > 0 when provided")
        if max_prompt_chars is not None and max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be > 0 when provided")
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.batch_wait_s = batch_wait_ms / 1000.0
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_queue_size = max_queue_size
        self.max_prompt_chars = max_prompt_chars
        self.enable_chunked_prefill = enable_chunked_prefill
        self.cache_aware_scheduling = cache_aware_scheduling
        self._queue: asyncio.Queue[_QueueItem | None] = asyncio.Queue()
        self._pending: deque[_QueueItem | None] = deque()
        self._worker: asyncio.Task | None = None
        self._model_queue: queue.Queue[_ModelCall | None] = queue.Queue()
        self._model_thread: threading.Thread | None = None
        self.stats = BatchingStats()

    async def start(self) -> None:
        if self._model_thread is None:
            self._model_thread = threading.Thread(target=self._run_model_worker, daemon=True)
            self._model_thread.start()
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._worker is not None:
            await self._queue.put(None)
            await self._worker
            self._worker = None
        if self._model_thread is not None:
            self._model_queue.put(None)
            self._model_thread.join()
            self._model_thread = None

    async def generate(self, prompt: str, sampling_params: SamplingParams) -> dict:
        await self.start()
        self._admit(prompt)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        await self._queue.put(_QueueItem(prompt=prompt, sampling_params=sampling_params, future=future))
        return await future

    async def stream(self, prompt: str, sampling_params: SamplingParams) -> AsyncIterator[dict]:
        await self.start()
        self._admit(prompt)
        events: queue.Queue[dict | BaseException | None] = queue.Queue()

        def run_stream() -> None:
            try:
                for event in self.engine.generate_stream(prompt, sampling_params):
                    events.put(event)
            except BaseException as exc:
                events.put(exc)
            finally:
                events.put(None)

        self._model_queue.put(_ModelCall(fn=run_stream, args=(), kwargs={}))
        started = time.perf_counter()
        stage = "waiting"
        while True:
            try:
                event = await asyncio.wait_for(_async_queue_get(events), timeout=5.0)
            except asyncio.TimeoutError:
                yield {"event": "heartbeat", "stage": stage, "elapsed_ms": round((time.perf_counter() - started) * 1000)}
                continue
            if event is None:
                return
            if isinstance(event, BaseException):
                raise event
            if event.get("event") == "phase":
                if event.get("phase") == "prefill" and event.get("status") == "start":
                    stage = "prefill"
                elif event.get("phase") == "prefill" and event.get("status") == "complete":
                    stage = "decode"
            yield event

    async def _run(self) -> None:
        while True:
            item = await self._next_item()
            if item is None:
                return
            batch = [item]
            await self._collect_batch(batch)
            await self._process_batch(batch)

    async def _collect_batch(self, batch: list[_QueueItem]) -> None:
        if self.batch_wait_s > 0:
            await asyncio.sleep(self.batch_wait_s)
        while len(batch) < self.max_batch_size:
            try:
                item = self._next_item_nowait()
            except asyncio.QueueEmpty:
                break
            if item is None:
                self._pending.appendleft(None)
                break
            if item.sampling_params != batch[0].sampling_params:
                self._pending.appendleft(item)
                break
            batch.append(item)

    async def _process_batch(self, batch: list[_QueueItem]) -> None:
        self.stats.total_requests += len(batch)
        self.stats.total_batches += 1
        self.stats.max_observed_batch_size = max(self.stats.max_observed_batch_size, len(batch))
        try:
            results = await self._run_model(
                self.engine.generate_batch,
                [item.prompt for item in batch],
                batch[0].sampling_params,
                max_num_seqs=self.max_batch_size,
                max_num_batched_tokens=self.max_num_batched_tokens,
                enable_chunked_prefill=self.enable_chunked_prefill,
                cache_aware_scheduling=self.cache_aware_scheduling,
            )
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return
        for item, result in zip(batch, results):
            if not item.future.done():
                item.future.set_result(result)

    async def _next_item(self) -> _QueueItem | None:
        if self._pending:
            return self._pending.popleft()
        return await self._queue.get()

    def _next_item_nowait(self) -> _QueueItem | None:
        if self._pending:
            return self._pending.popleft()
        return self._queue.get_nowait()

    def stats_dict(self) -> dict:
        stats = self.stats.as_dict(queue_depth=self._queue.qsize() + len(self._pending))
        stats["max_queue_size"] = self.max_queue_size
        stats["max_prompt_chars"] = self.max_prompt_chars
        stats["enable_chunked_prefill"] = self.enable_chunked_prefill
        stats["cache_aware_scheduling"] = self.cache_aware_scheduling
        return stats

    def _admit(self, prompt: str) -> None:
        if self.max_prompt_chars is not None and len(prompt) > self.max_prompt_chars:
            raise ValueError(
                f"prompt too long: {len(prompt)} chars exceeds max_prompt_chars={self.max_prompt_chars}"
            )
        if self.max_queue_size is not None:
            depth = self._queue.qsize() + len(self._pending)
            if depth >= self.max_queue_size:
                raise RuntimeError(f"request queue is full: depth={depth}, max_queue_size={self.max_queue_size}")

    async def _run_model(self, fn: Callable, *args, **kwargs):
        result_queue: queue.Queue = queue.Queue(maxsize=1)
        self._model_queue.put(_ModelCall(fn=fn, args=args, kwargs=kwargs, result_queue=result_queue))
        status, value = await _async_queue_get(result_queue)
        if status == "error":
            raise value
        return value

    def _run_model_worker(self) -> None:
        while True:
            call = self._model_queue.get()
            if call is None:
                return
            try:
                result = call.fn(*call.args, **call.kwargs)
            except BaseException as exc:
                if call.result_queue is not None:
                    call.result_queue.put(("error", exc))
            else:
                if call.result_queue is not None:
                    call.result_queue.put(("ok", result))


async def _async_queue_get(items: queue.Queue):
    while True:
        try:
            return items.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.001)
