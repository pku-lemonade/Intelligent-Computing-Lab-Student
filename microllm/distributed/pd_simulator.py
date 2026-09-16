from __future__ import annotations

from microllm.distributed.pd_runtime import PDEvent, PDPhase, PDTrace, PDWorkload


class PDRouter:
    def __init__(self, *, prefill_workers: int = 1, decode_workers: int = 1):
        if prefill_workers <= 0 or decode_workers <= 0:
            raise ValueError("worker counts must be positive")
        self.prefill_workers = prefill_workers
        self.decode_workers = decode_workers

    def schedule(self, workloads: list[PDWorkload]) -> PDTrace:
        # TODO_BEGIN(W15_T03)
        return PDTrace(events=(), completion_s={})
        # TODO_END(W15_T03)
