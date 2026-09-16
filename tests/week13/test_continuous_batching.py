from __future__ import annotations

import torch

from microllm.engine.async_engine import AsyncEngine
from microllm.engine.request import Request, RequestStatus, Sequence
from microllm.engine.sampler import SamplingParams
from microllm.engine.scheduler import BatchKind, Scheduler
from microllm.model.types import BatchModelOutput, ModelOutput


def _sequence(prompt_tokens: int = 2) -> Sequence:
    return Sequence(
        Request("x", SamplingParams(temperature=0.0, max_tokens=4)),
        list(range(prompt_tokens)),
    )


def test_w13_t01_add_initializes_waiting_state():
    scheduler = Scheduler()
    sequence = _sequence()
    sequence.status = RequestStatus.FINISHED
    sequence.request.status = RequestStatus.FINISHED
    scheduler.add(sequence)
    assert sequence.status == RequestStatus.WAITING
    assert sequence.request.status == RequestStatus.WAITING
    assert list(scheduler.waiting) == [sequence]


def test_w13_t02_prefill_respects_token_budget_and_chunking():
    scheduler = Scheduler(max_num_seqs=2, max_num_batched_tokens=3, enable_chunked_prefill=True)
    sequence = _sequence(prompt_tokens=5)
    scheduler.add(sequence)
    first = scheduler.schedule()
    assert first is not None and first.kind == BatchKind.PREFILL
    assert first.num_tokens == 3
    assert (sequence.scheduled_start, sequence.scheduled_end) == (0, 3)
    sequence.prefill_offset = sequence.scheduled_end
    second = scheduler.schedule()
    assert second is not None and second.num_tokens == 2
    assert (sequence.scheduled_start, sequence.scheduled_end) == (3, 5)


def test_w13_t03_decode_selects_running_sequences():
    scheduler = Scheduler(max_num_seqs=2, max_num_batched_tokens=8)
    sequences = [_sequence(), _sequence()]
    for sequence in sequences:
        scheduler.add(sequence)
    prefill = scheduler.schedule()
    assert prefill is not None and prefill.kind == BatchKind.PREFILL
    for sequence in sequences:
        sequence.prefill_offset = len(sequence.prompt_token_ids)
        sequence.next_token_id = 10
    decode = scheduler.schedule()
    assert decode is not None and decode.kind == BatchKind.DECODE
    assert decode.sequences == sequences
    assert decode.num_tokens == 2


def test_w13_t04_async_engine_accepts_request_during_decode():
    backend = _OnlineBackend()
    engine = AsyncEngine(backend, scheduler=Scheduler(max_num_seqs=4, max_num_batched_tokens=16))
    first_id = engine.add_request("a", SamplingParams(temperature=0.0, max_tokens=4))
    assert engine.step() == []
    assert [event["token_id"] for event in engine.step() if event["event"] == "token"] == [10]

    second_id = engine.add_request("bb", SamplingParams(temperature=0.0, max_tokens=4))
    assert engine.step() == []
    events = engine.step()
    tokens = [(event["request_id"], event["token_id"]) for event in events if event["event"] == "token"]
    assert tokens == [(first_id, 11), (second_id, 10)]
    assert backend.prefill_batch_sizes == [1, 1]


class _OnlineBackend:
    eos_token_id = 99

    def __init__(self):
        self.prefill_batch_sizes = []
        self.decode_batch_sizes = []
        self.released = []

    def encode(self, prompt: str) -> list[int]:
        return [ord(char) for char in prompt]

    def decode(self, token_ids: list[int]) -> str:
        return ",".join(str(token) for token in token_ids)

    def prefill_batch(self, prompts: list[list[int]]) -> BatchModelOutput:
        self.prefill_batch_sizes.append(len(prompts))
        outputs = [self._output(10, {"steps": 0}) for _ in prompts]
        return BatchModelOutput([output.logits for output in outputs], [output.past_key_values for output in outputs])

    def decode_batch(self, token_ids: list[int], caches: list[dict]) -> BatchModelOutput:
        self.decode_batch_sizes.append(len(token_ids))
        outputs = []
        for token_id, cache in zip(token_ids, caches):
            cache["steps"] += 1
            outputs.append(self._output(self.eos_token_id if token_id >= 11 else token_id + 1, cache))
        return BatchModelOutput([output.logits for output in outputs], [output.past_key_values for output in outputs])

    def release_cache(self, cache: object) -> None:
        self.released.append(cache)

    def _output(self, token_id: int, cache: object) -> ModelOutput:
        logits = torch.zeros(128)
        logits[token_id] = 1
        return ModelOutput(logits, cache)
