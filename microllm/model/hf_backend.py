from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from microllm.model.types import BatchModelOutput, ModelOutput, bucket_by_length, parse_dtype


class HFModelBackend:
    """HuggingFace-backed model runner used as the first stable reference path."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        trust_remote_code: bool = True,
    ):
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = parse_dtype(dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def prefill(self, token_ids: list[int]) -> ModelOutput:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        return ModelOutput(logits=out.logits[0, -1], past_key_values=out.past_key_values)

    @torch.inference_mode()
    def prefill_batch(self, batch_token_ids: list[list[int]]) -> BatchModelOutput:
        if not batch_token_ids:
            return BatchModelOutput(logits=[], past_key_values=[])
        logits: list[torch.Tensor | None] = [None] * len(batch_token_ids)
        past_key_values: list[object | None] = [None] * len(batch_token_ids)
        for indices in bucket_by_length(batch_token_ids).values():
            bucket_token_ids = [batch_token_ids[index] for index in indices]
            bucket_out = self._prefill_same_length_batch(bucket_token_ids)
            for index, bucket_index in enumerate(indices):
                logits[bucket_index] = bucket_out.logits[index]
                past_key_values[bucket_index] = bucket_out.past_key_values[index]
        if any(item is None for item in logits):
            raise RuntimeError("internal error: missing batch logits")
        return BatchModelOutput(
            logits=[item for item in logits if item is not None],
            past_key_values=past_key_values,
        )

    @torch.inference_mode()
    def _prefill_same_length_batch(self, batch_token_ids: list[list[int]]) -> BatchModelOutput:
        input_ids = torch.tensor(batch_token_ids, dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, use_cache=True)
        return BatchModelOutput(
            logits=[out.logits[batch_index, -1] for batch_index in range(len(batch_token_ids))],
            past_key_values=[
                tuple(
                    (
                        key[batch_index : batch_index + 1],
                        value[batch_index : batch_index + 1],
                    )
                    for key, value in out.past_key_values
                )
                for batch_index in range(len(batch_token_ids))
            ],
        )

    @torch.inference_mode()
    def decode_step(self, token_id: int, past_key_values: object) -> ModelOutput:
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        out = self.model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        return ModelOutput(logits=out.logits[0, -1], past_key_values=out.past_key_values)

    @torch.inference_mode()
    def decode_batch(
        self,
        token_ids: list[int],
        past_key_values: list[object | None],
    ) -> BatchModelOutput:
        outputs = [
            self.decode_step(token_id, past_key_value)
            for token_id, past_key_value in zip(token_ids, past_key_values)
        ]
        return BatchModelOutput(
            logits=[output.logits for output in outputs],
            past_key_values=[output.past_key_values for output in outputs],
        )
