from __future__ import annotations

import torch

from microllm.engine.engine import Engine
from microllm.engine.kv_cache import ContinuousKVCache
from microllm.engine.request import Request, RequestStatus, Sequence
from microllm.engine.sampler import SamplingParams
from microllm.model.types import ModelOutput


def _sequence() -> Sequence:
    request = Request("prompt", SamplingParams(max_tokens=4))
    return Sequence(request=request, prompt_token_ids=[10, 11])


def test_w11_t01_append_token_updates_sequence():
    sequence = _sequence()
    sequence.append_token(12.0)
    sequence.append_token(13)
    assert sequence.generated_token_ids == [12, 13]
    assert sequence.token_ids == [10, 11, 12, 13]
    assert sequence.reached_max_tokens() is False


def test_w11_t02_finish_updates_sequence_and_request():
    sequence = _sequence()
    sequence.status = RequestStatus.RUNNING
    sequence.request.status = RequestStatus.RUNNING
    sequence.finish()
    assert sequence.status == RequestStatus.FINISHED
    assert sequence.request.status == RequestStatus.FINISHED


def test_w11_t03_continuous_kv_appends_token_dimension():
    cache = ContinuousKVCache()
    first = torch.randn((1, 2, 3, 4))
    second = torch.randn((1, 2, 2, 4))
    cache.append(7, 1, first, first + 1)
    first.zero_()
    cache.append(7, 1, second, second + 1)
    stored = cache.get(7, 1)
    assert stored.key.shape == (1, 2, 5, 4)
    assert torch.count_nonzero(stored.key[:, :, :3]) > 0
    assert torch.equal(stored.key[:, :, 3:], second)


def test_w11_t04_continuous_kv_releases_one_sequence():
    cache = ContinuousKVCache()
    value = torch.ones((1, 1, 1, 2))
    for sequence_id in (1, 2):
        for layer_id in (0, 1):
            cache.append(sequence_id, layer_id, value, value)
    cache.release(1)
    assert cache.num_layers_for(1) == 0
    assert cache.num_layers_for(2) == 2


def test_w11_t05_generate_stream_orders_events_and_releases_cache():
    backend = _StreamingBackend()
    engine = object.__new__(Engine)
    engine.backend = backend
    token_state = []
    finished_state = []
    original_token_event = engine._stream_token_event
    original_finished_event = engine._stream_finished_event

    def record_token_state(sequence, token_id):
        token_state.append((sequence.status, sequence.request.status))
        return original_token_event(sequence, token_id)

    def record_finished_state(sequence, finish_reason):
        finished_state.append((sequence.status, sequence.finish_reason))
        return original_finished_event(sequence, finish_reason)

    engine._stream_token_event = record_token_state
    engine._stream_finished_event = record_finished_state

    events = list(engine.generate_stream("x", SamplingParams(temperature=0.0, max_tokens=4)))

    token_events = [event for event in events if event["event"] == "token"]
    assert events[0]["event"] == "request"
    assert {event["event"] for event in events} >= {"request", "scheduler", "phase", "kv_cache", "token", "finished"}
    assert [event["token_id"] for event in token_events] == [3, backend.eos_token_id]
    assert events[-1]["finish_reason"] == "eos"
    assert token_state == [
        (RequestStatus.RUNNING, RequestStatus.RUNNING),
        (RequestStatus.RUNNING, RequestStatus.RUNNING),
    ]
    assert finished_state == [(RequestStatus.FINISHED, "eos")]
    assert backend.released == [{"step": 1}]


def test_w11_t05_generate_stream_releases_cache_when_consumer_closes_early():
    backend = _StreamingBackend()
    engine = object.__new__(Engine)
    engine.backend = backend
    stream = engine.generate_stream("x", SamplingParams(temperature=0.0, max_tokens=4))

    while next(stream)["event"] != "token":
        pass
    stream.close()

    assert backend.released == [{"step": 0}]


def test_w11_t05_generate_stream_releases_cache_when_decode_fails():
    backend = _StreamingBackend()
    backend.decode_step = lambda *_: (_ for _ in ()).throw(RuntimeError("decode failed"))
    engine = object.__new__(Engine)
    engine.backend = backend

    try:
        list(engine.generate_stream("x", SamplingParams(temperature=0.0, max_tokens=4)))
    except RuntimeError as exc:
        assert str(exc) == "decode failed"
    else:
        raise AssertionError("decode failure must propagate")

    assert backend.released == [{"step": 0}]


def test_stream_rejects_context_beyond_paged_capacity_before_prefill():
    backend = _StreamingBackend()
    backend.max_context_tokens = lambda: 16
    backend.encode = lambda _: list(range(15))
    engine = object.__new__(Engine)
    engine.backend = backend

    try:
        list(engine.generate_stream("x", SamplingParams(max_tokens=4)))
    except ValueError as exc:
        assert "capacity is 16 tokens" in str(exc)
    else:
        raise AssertionError("oversized context must be rejected before prefill")
    assert backend.released == []


class _StreamingBackend:
    eos_token_id = 9
    tokenizer = object()

    def __init__(self):
        self.released = []

    def encode(self, prompt: str) -> list[int]:
        return [1]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(str(token) for token in token_ids)

    def prefill(self, token_ids: list[int]) -> ModelOutput:
        logits = torch.zeros(10)
        logits[3] = 1
        return ModelOutput(logits, {"step": 0})

    def decode_step(self, token_id: int, cache: dict) -> ModelOutput:
        cache["step"] += 1
        logits = torch.zeros(10)
        logits[self.eos_token_id] = 1
        return ModelOutput(logits, cache)

    def release_cache(self, cache: object) -> None:
        self.released.append(cache)
