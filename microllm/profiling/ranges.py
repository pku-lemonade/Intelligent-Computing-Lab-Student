from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch


PHASES = frozenset({"prefill", "decode", "h2d", "sampling"})


@contextmanager
def phase_range(phase: str, *, enable_nvtx: bool = True) -> Iterator[str]:
    # TODO_BEGIN(W02_T01)
    name = f"microllm::{phase}"
    yield name
    # TODO_END(W02_T01)
