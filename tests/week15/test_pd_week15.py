from __future__ import annotations

import pytest
import torch

from microllm.distributed import (
    KVTransferPlan,
    PDPhase,
    PDRequestState,
    PDRouter,
    PDWorkload,
)
from microllm.engine.request import Request, Sequence
from microllm.engine.sampler import SamplingParams
from microllm.model.qwen3_torch import Qwen3Config


def test_w15_t01_pd_state_allows_only_monotonic_pipeline_transitions():
    state = _state([1, 2, 3])
    state.advance(PDPhase.PREFILLING, at_s=0.5)
    state.advance(PDPhase.TRANSFERRING, at_s=1.0)
    state.advance(PDPhase.DECODING, at_s=1.25)
    state.advance(PDPhase.FINISHED, at_s=2.0)
    assert [phase for phase, _ in state.history] == list(PDPhase)

    invalid = _state([1])
    with pytest.raises(ValueError, match="invalid P/D transition"):
        invalid.advance(PDPhase.DECODING, at_s=0.1)
    invalid.advance(PDPhase.PREFILLING, at_s=0.2)
    with pytest.raises(ValueError, match="monotonic"):
        invalid.advance(PDPhase.TRANSFERRING, at_s=0.1)


def test_w15_t02_kv_transfer_uses_real_model_shape():
    state = _state([1, 2, 3])
    plan = KVTransferPlan.for_request(
        state, _config(), dtype=torch.bfloat16, bandwidth_gbps=2.0
    )
    assert plan.tokens == 3
    assert plan.bytes == 2 * 2 * 2 * 4 * 3 * 2
    assert plan.duration_s == pytest.approx(plan.bytes / 2_000_000_000)
    with pytest.raises(ValueError, match="positive"):
        KVTransferPlan.for_request(state, _config(), dtype=torch.float32, bandwidth_gbps=0)


def test_w15_t03_router_respects_compute_and_link_dependencies():
    first = _state([1], arrival=0.0)
    second = _state([2], arrival=0.2)
    workloads = [
        PDWorkload(first, KVTransferPlan(first.request_id, 1, 1, 1.0, 0.5), 1.0, 2.0),
        PDWorkload(second, KVTransferPlan(second.request_id, 1, 1, 1.0, 0.5), 1.0, 1.0),
    ]
    trace = PDRouter().schedule(workloads)
    events = { (event.request_id, event.phase): event for event in trace.events }
    assert events[(first.request_id, PDPhase.TRANSFERRING)].start_s >= events[(first.request_id, PDPhase.PREFILLING)].end_s
    assert events[(first.request_id, PDPhase.DECODING)].start_s >= events[(first.request_id, PDPhase.TRANSFERRING)].end_s
    assert events[(second.request_id, PDPhase.PREFILLING)].start_s >= events[(first.request_id, PDPhase.PREFILLING)].end_s
    assert events[(second.request_id, PDPhase.DECODING)].start_s >= events[(first.request_id, PDPhase.DECODING)].end_s
    assert first.phase == second.phase == PDPhase.FINISHED


def _state(tokens: list[int], *, arrival: float = 0.0) -> PDRequestState:
    sequence = Sequence(Request("x", SamplingParams()), prompt_token_ids=tokens)
    return PDRequestState(sequence, decode_tokens=4, arrival_s=arrival)


def _config() -> Qwen3Config:
    return Qwen3Config(32, 8, 16, 2, 4, 2, 4, 128, 1e-6, 10000.0)
