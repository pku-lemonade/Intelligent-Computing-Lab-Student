from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

import microllm.profiling.ranges as ranges
from microllm.profiling import phase_range


def test_w02_t01_phase_range_records_and_balances_nvtx(monkeypatch):
    events: list[str] = []

    @contextmanager
    def fake_record_function(name: str):
        events.append(f"record_enter:{name}")
        try:
            yield
        finally:
            events.append(f"record_exit:{name}")

    monkeypatch.setattr(ranges.torch.autograd.profiler, "record_function", fake_record_function)
    monkeypatch.setattr(ranges.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(ranges.torch.cuda.nvtx, "range_push", lambda name: events.append(f"push:{name}"))
    monkeypatch.setattr(ranges.torch.cuda.nvtx, "range_pop", lambda: events.append("pop"))

    with pytest.raises(RuntimeError, match="work failed"):
        with phase_range("prefill") as name:
            assert name == "microllm::prefill"
            events.append("work")
            raise RuntimeError("work failed")

    assert events == [
        "record_enter:microllm::prefill",
        "push:microllm::prefill",
        "work",
        "pop",
        "record_exit:microllm::prefill",
    ]

    with pytest.raises(ValueError, match="unknown runtime phase"):
        with phase_range("unknown"):
            pass

    monkeypatch.undo()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with phase_range("decode", enable_nvtx=False):
            torch.ones(4) + 1
    assert "microllm::decode" in {event.key for event in profile.key_averages()}
