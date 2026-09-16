from __future__ import annotations

from typing import Protocol

from microllm.engine.request import Request, Sequence
from microllm.engine.sampler import Sampler, SamplingParams
from microllm.engine.scheduler import BatchKind, ScheduledBatch, Scheduler
from microllm.model.types import BatchModelOutput, ModelOutput
from microllm.profiling import phase_range


class Backend(Protocol):
    eos_token_id: int

    def encode(self, prompt: str) -> list[int]: ...
    def decode(self, token_ids: list[int]) -> str: ...
    def prefill(self, token_ids: list[int]) -> ModelOutput: ...
    def decode_step(self, token_id: int, past_key_values: object) -> ModelOutput: ...


class AsyncEngine:
    """Iteration-level engine that accepts requests while other requests are running."""

    def __init__(self, backend: Backend, *, scheduler: Scheduler | None = None):
        self.backend = backend
        self.scheduler = scheduler or Scheduler()
        self.sequences: dict[int, Sequence] = {}
        self.samplers: dict[int, Sampler] = {}

    def add_request(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> int:
        params = sampling_params or SamplingParams()
        token_ids = self.backend.encode(prompt)
        if not token_ids:
            raise ValueError("prompt encoded to an empty token sequence")
        request = Request(prompt=prompt, sampling_params=params)
        sequence = Sequence(request=request, prompt_token_ids=token_ids)
        self.sequences[sequence.request_id] = sequence
        self.samplers[sequence.request_id] = Sampler(params)
        self.scheduler.add(sequence)
        return sequence.request_id

    def step(self) -> list[dict]:
        # TODO_BEGIN(W13_T04)
        return []
        # TODO_END(W13_T04)

    def _token_event(self, sequence: Sequence, token_id: int) -> dict:
        return {
            "event": "token",
            "request_id": sequence.request_id,
            "token_id": token_id,
            "text": self.backend.decode([token_id]),
        }

    def _finish_sequence(self, sequence: Sequence, finish_reason: str) -> dict:
        sequence.finish_reason = finish_reason
        self.scheduler.finish(sequence)
        self._release(sequence.past_key_values)
        sequence.past_key_values = None
        return {
            "event": "finished",
            "request_id": sequence.request_id,
            "finish_reason": finish_reason,
        }

    def has_unfinished(self) -> bool:
        return self.scheduler.has_unfinished()

    def result(self, request_id: int) -> dict:
        sequence = self.sequences[request_id]
        return {
            "text": self.backend.decode(sequence.generated_token_ids),
            "token_ids": list(sequence.generated_token_ids),
            "finish_reason": sequence.finish_reason,
        }

    def _run_prefill(self, batch: ScheduledBatch) -> list[ModelOutput]:
        full_prompts = all(
            sequence.scheduled_start == 0
            and sequence.scheduled_end == len(sequence.prompt_token_ids)
            for sequence in batch.sequences
        )
        prefill_batch = getattr(self.backend, "prefill_batch", None)
        with phase_range("prefill"):
            if full_prompts and prefill_batch is not None:
                output: BatchModelOutput = prefill_batch(
                    [sequence.prompt_token_ids for sequence in batch.sequences]
                )
                return [
                    ModelOutput(logits=logits, past_key_values=cache)
                    for logits, cache in zip(output.logits, output.past_key_values)
                ]
            outputs = []
            for sequence in batch.sequences:
                tokens = sequence.scheduled_prompt_tokens()
                if sequence.scheduled_start == 0 and sequence.scheduled_end == len(sequence.prompt_token_ids):
                    outputs.append(self.backend.prefill(tokens))
                    continue
                prefill_chunk = getattr(self.backend, "prefill_chunk", None)
                if prefill_chunk is None:
                    raise RuntimeError("backend does not support chunked prefill")
                outputs.append(prefill_chunk(tokens, sequence.past_key_values))
            return outputs

    def _run_decode(self, sequences: list[Sequence]) -> list[ModelOutput]:
        if not sequences:
            return []
        decode_batch = getattr(self.backend, "decode_batch", None)
        with phase_range("decode"):
            if decode_batch is not None:
                output: BatchModelOutput = decode_batch(
                    [sequence.next_token_id for sequence in sequences],
                    [sequence.past_key_values for sequence in sequences],
                )
                return [
                    ModelOutput(logits=logits, past_key_values=cache)
                    for logits, cache in zip(output.logits, output.past_key_values)
                ]
            return [
                self.backend.decode_step(sequence.next_token_id, sequence.past_key_values)
                for sequence in sequences
            ]

    def _finish_reason(self, sequence: Sequence, token_id: int) -> str | None:
        params = sequence.request.sampling_params
        if token_id == self.backend.eos_token_id:
            return "eos"
        if token_id in params.stop_token_ids:
            return "stop"
        if sequence.reached_max_tokens():
            return "length"
        return None

    def _release(self, cache: object | None) -> None:
        release_cache = getattr(self.backend, "release_cache", None)
        if release_cache is not None:
            release_cache(cache)
