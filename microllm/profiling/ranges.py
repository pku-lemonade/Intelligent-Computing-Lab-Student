from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


PHASES = frozenset({"prefill", "decode", "h2d", "sampling"})


@contextmanager
def phase_range(phase: str, *, enable_nvtx: bool = True) -> Iterator[str]:
    if phase not in PHASES:
        raise ValueError(f"unknown runtime phase {phase!r}; expected one of {sorted(PHASES)}")
    name = f"microllm::{phase}"
    pushed = False
    with torch.autograd.profiler.record_function(name):
        try:
            if enable_nvtx and torch.cuda.is_available():
                torch.cuda.nvtx.range_push(name)
                pushed = True
            yield name
        finally:
            if pushed:
                torch.cuda.nvtx.range_pop()
