from __future__ import annotations

from collections.abc import Iterator
import time

from microllm.engine.request import Request, RequestStatus, Sequence
from microllm.engine.sampler import Sampler, SamplingParams
from microllm.engine.scheduler import BatchKind, Scheduler
from microllm.engine.policies import cache_aware_order
from microllm.model.hf_backend import HFModelBackend
from microllm.model.qwen3_backend import Qwen3PagedBackend, Qwen3TorchBackend
from microllm.model.types import ModelOutput
from microllm.profiling import phase_range
from microllm.stages import normalize_stage, stage_overview


def normalize_backend_config(backend: str, kv_mode: str = "paged") -> tuple[str, str]:
    """Normalize public backend/kv-mode settings while preserving old aliases."""
    if backend == "hf":
        return "reference", "dense"
    if backend == "paged":
        return "microllm", "paged"
    if backend not in {"reference", "microllm"}:
        raise ValueError("unsupported backend: expected reference or microllm")
    if kv_mode not in {"dense", "paged"}:
        raise ValueError("unsupported kv_mode: expected dense or paged")
    if backend == "reference":
        return backend, "dense"
    return backend, kv_mode


class Engine:
    def __init__(
        self,
        model: str,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        backend: str = "microllm",
        kv_mode: str = "paged",
        stage: str | int | None = None,
        kernel_impl: str = "torch",
        use_pinned_memory: bool = False,
        use_transfer_stream: bool = False,
    ):
        self.stage = normalize_stage(stage)
        self.kernel_impl = kernel_impl
        self.use_pinned_memory = use_pinned_memory
        self.use_transfer_stream = use_transfer_stream
        backend, kv_mode = normalize_backend_config(backend, kv_mode)
        if backend == "reference":
            self.backend = HFModelBackend(model, dtype=dtype, device=device)
        elif kv_mode == "dense":
            self.backend = Qwen3TorchBackend(
                model,
                dtype=dtype,
                device=device,
                kernel_impl=kernel_impl,
                use_pinned_memory=use_pinned_memory,
                use_transfer_stream=use_transfer_stream,
            )
        elif kv_mode == "paged":
            self.backend = Qwen3PagedBackend(
                model,
                dtype=dtype,
                device=device,
                kernel_impl=kernel_impl,
                use_pinned_memory=use_pinned_memory,
                use_transfer_stream=use_transfer_stream,
            )
        else:
            raise ValueError(f"unsupported kv_mode: {kv_mode}")
        self.backend_name = backend
        self.kv_mode = kv_mode

    @property
    def tokenizer(self):
        return self.backend.tokenizer

    def info(self) -> dict:
        return {
            "backend": getattr(self, "backend_name", type(self.backend).__name__),
            "kv_mode": getattr(self, "kv_mode", "unknown"),
            "stage": stage_overview(self.stage),
            "kernel_impl": self.kernel_impl,
            "model_backend": type(self.backend).__name__,
            "system_optimizations": {
                "pinned_memory": self.use_pinned_memory,
                "transfer_stream": self.use_transfer_stream,
            },
        }

    def generate(self, prompt: str, sampling_params: SamplingParams | None = None) -> dict:
        sampling_params = sampling_params or SamplingParams()
        text = ""
        token_ids: list[int] = []
        for event in self.generate_stream(prompt, sampling_params):
            if event["event"] == "token":
                text += event["text"]
                token_ids.append(event["token_id"])
            elif event["event"] == "finished":
                return {
                    "text": text,
                    "token_ids": token_ids,
                    "finish_reason": event["finish_reason"],
                }
        return {"text": text, "token_ids": token_ids, "finish_reason": "unknown"}

    def generate_batch(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | None = None,
        *,
        max_num_seqs: int = 8,
        max_num_batched_tokens: int = 2048,
        enable_chunked_prefill: bool = False,
        cache_aware_scheduling: bool = False,
    ) -> list[dict]:
        sampling_params = sampling_params or SamplingParams()
        scheduler = Scheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_chunked_prefill=enable_chunked_prefill,
        )
        seqs: list[Sequence] = []
        samplers: dict[int, Sampler] = {}
        encoded_prompts: list[tuple[int, str, list[int]]] = []
        for index, prompt in enumerate(prompts):
            prompt_token_ids = self.backend.encode(prompt)
            if not prompt_token_ids:
                raise ValueError("prompt encoded to an empty token sequence")
            encoded_prompts.append((index, prompt, prompt_token_ids))
        if cache_aware_scheduling:
            order = cache_aware_order([tokens for _, _, tokens in encoded_prompts])
            encoded_prompts = [encoded_prompts[index] for index in order]
        original_index_by_request: dict[int, int] = {}
        for original_index, prompt, prompt_token_ids in encoded_prompts:
            request = Request(prompt=prompt, sampling_params=sampling_params)
            seq = Sequence(request=request, prompt_token_ids=prompt_token_ids)
            seqs.append(seq)
            original_index_by_request[seq.request_id] = original_index
            samplers[seq.request_id] = Sampler(sampling_params)
            scheduler.add(seq)

        try:
            while scheduler.has_unfinished():
                batch = scheduler.schedule()
                if batch is None:
                    break
                if batch.kind == BatchKind.PREFILL:
                    outputs = self._prefill_batch(batch.sequences, chunked=enable_chunked_prefill)
                    for seq, output in zip(batch.sequences, outputs):
                        seq.past_key_values = output.past_key_values
                        seq.prefill_offset = seq.scheduled_end
                        if seq.prefill_complete():
                            with phase_range("sampling"):
                                seq.next_token_id = samplers[seq.request_id].sample(output.logits)
                else:
                    decode_seqs = []
                    for seq in batch.sequences:
                        if seq.next_token_id is None:
                            raise RuntimeError("decode scheduled before prefill")
                        self._accept_next_token(seq, seq.next_token_id, sampling_params)
                        if seq.finish_reason is not None:
                            scheduler.finish(seq)
                            self._release_cache(seq.past_key_values)
                            seq.past_key_values = None
                            continue
                        decode_seqs.append(seq)
                    outputs = self._decode_batch(decode_seqs)
                    for seq, output in zip(decode_seqs, outputs):
                        seq.past_key_values = output.past_key_values
                        with phase_range("sampling"):
                            seq.next_token_id = samplers[seq.request_id].sample(output.logits)
        finally:
            for seq in seqs:
                self._release_cache(seq.past_key_values)
                seq.past_key_values = None

        ordered_results = [
            {
                "text": self.backend.decode(seq.generated_token_ids),
                "token_ids": seq.generated_token_ids,
                "finish_reason": seq.finish_reason or "unknown",
            }
            for seq in seqs
        ]
        results_by_original = [None] * len(ordered_results)
        for seq, result in zip(seqs, ordered_results):
            results_by_original[original_index_by_request[seq.request_id]] = result
        return [result for result in results_by_original if result is not None]

    def generate_stream(
        self,
        prompt: str,
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[dict]:
        # TODO_BEGIN(W11_T05)
        if False:
            yield {}
        return
        # TODO_END(W11_T05)

    def _stream_token_event(self, seq: Sequence, token_id: int) -> dict:
        return {
            "event": "token",
            "request_id": seq.request_id,
            "token_id": token_id,
            "text": self.backend.decode([token_id]),
        }

    def _stream_finished_event(self, seq: Sequence, finish_reason: str) -> dict:
        return {
            "event": "finished",
            "request_id": seq.request_id,
            "finish_reason": finish_reason,
            "token_ids": seq.generated_token_ids,
            "text": self.backend.decode(seq.generated_token_ids),
        }

    def _cache_stats(self, past_key_values: object | None) -> dict:
        stats = getattr(self.backend, "cache_stats", None)
        return stats(past_key_values) if stats is not None else {"mode": getattr(self, "kv_mode", "unknown"), "active": False}

    def chat(self, messages: list[dict[str, str]], sampling_params: SamplingParams | None = None) -> dict:
        prompt = self.backend.apply_chat_template(messages)
        return self.generate(prompt, sampling_params)

    def chat_stream(
        self,
        messages: list[dict[str, str]],
        sampling_params: SamplingParams | None = None,
    ) -> Iterator[dict]:
        prompt = self.backend.apply_chat_template(messages)
        yield from self.generate_stream(prompt, sampling_params)

    def _finish_reason(
        self,
        seq: Sequence,
        token_id: int,
        sampling_params: SamplingParams,
    ) -> str | None:
        if token_id == self.backend.eos_token_id:
            return "eos"
        if token_id in sampling_params.stop_token_ids:
            return "stop"
        if seq.reached_max_tokens():
            return "length"
        return None

    def _accept_next_token(
        self,
        seq: Sequence,
        token_id: int,
        sampling_params: SamplingParams,
    ) -> None:
        seq.append_token(token_id)
        seq.finish_reason = self._finish_reason(seq, token_id, sampling_params)

    def _prefill_batch(self, seqs: list[Sequence], *, chunked: bool = False) -> list[object]:
        with phase_range("prefill"):
            if chunked:
                return [self._prefill_chunk(seq) for seq in seqs]
            prefill_batch = getattr(self.backend, "prefill_batch", None)
            if prefill_batch is None:
                return [self.backend.prefill(seq.prompt_token_ids) for seq in seqs]
            output = prefill_batch([seq.prompt_token_ids for seq in seqs])
            return [
                ModelOutput(logits=logits, past_key_values=past_key_values)
                for logits, past_key_values in zip(output.logits, output.past_key_values)
            ]

    def _prefill_chunk(self, seq: Sequence) -> ModelOutput:
        tokens = seq.scheduled_prompt_tokens()
        if not tokens:
            raise RuntimeError("empty prefill chunk scheduled")
        prefill_chunk = getattr(self.backend, "prefill_chunk", None)
        if prefill_chunk is None:
            if seq.scheduled_end != len(seq.prompt_token_ids):
                raise RuntimeError("backend does not support chunked prefill")
            return self.backend.prefill(seq.prompt_token_ids)
        return prefill_chunk(tokens, seq.past_key_values)

    def _decode_batch(self, seqs: list[Sequence]) -> list[ModelOutput]:
        if not seqs:
            return []
        with phase_range("decode"):
            decode_batch = getattr(self.backend, "decode_batch", None)
            if decode_batch is None:
                return [
                    self.backend.decode_step(seq.next_token_id, seq.past_key_values)
                    for seq in seqs
                ]
            if any(seq.next_token_id is None for seq in seqs):
                raise RuntimeError("decode scheduled before prefill")
            output = decode_batch(
                [seq.next_token_id for seq in seqs],
                [seq.past_key_values for seq in seqs],
            )
            return [
                ModelOutput(logits=logits, past_key_values=past_key_values)
                for logits, past_key_values in zip(output.logits, output.past_key_values)
            ]

    def _release_cache(self, past_key_values: object | None) -> None:
        release_cache = getattr(self.backend, "release_cache", None)
        if release_cache is not None:
            release_cache(past_key_values)
