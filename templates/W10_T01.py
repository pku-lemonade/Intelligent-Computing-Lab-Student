from __future__ import annotations

from itertools import count

import torch
from transformers import AutoTokenizer

from microllm.engine.kv_cache import ContinuousKVCache, KVCacheHandle
from microllm.model.model_path import resolve_local_model_path
from microllm.model.qwen3_torch import Qwen3ForCausalLM, Qwen3KVCache
from microllm.model.types import BatchModelOutput, ModelOutput, parse_dtype
from microllm.profiling import phase_range


class Qwen3TorchBackend:
    """Course-owned Qwen3 runner with an explicit dense KV-cache path."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        trust_remote_code: bool = True,
        kernel_impl: str = "torch",
        use_pinned_memory: bool = False,
        use_transfer_stream: bool = False,
    ):
        self.model_path = model_path
        local_model_path = resolve_local_model_path(model_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = parse_dtype(dtype)
        self.kernel_impl = kernel_impl
        self.use_pinned_memory = use_pinned_memory
        self.transfer_stream = (
            torch.cuda.Stream(device=self.device)
            if use_transfer_stream and self.device.type == "cuda"
            else None
        )
        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = Qwen3ForCausalLM.from_pretrained(
            local_model_path,
            device=self.device,
            dtype=self.dtype,
            kernel_impl=kernel_impl,
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        self.kv_cache = ContinuousKVCache()
        self._cache_ids = count()

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def cache_stats(self, handle: KVCacheHandle | None) -> dict:
        if handle is None:
            return {"mode": "continuous", "active": False}
        layers = self.kv_cache.layers(handle.seq_id)
        bytes_used = sum(layer.key.numel() * layer.key.element_size() + layer.value.numel() * layer.value.element_size() for layer in layers)
        return {
            "mode": "continuous",
            "seq_id": handle.seq_id,
            "seq_len": handle.seq_len,
            "layers": len(layers),
            "bytes": bytes_used,
            "dtype": str(layers[0].key.dtype) if layers else str(self.dtype),
            "device": str(self.device),
        }

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @torch.inference_mode()
    def prefill(self, token_ids: list[int]) -> ModelOutput:
        input_ids = self._token_tensor([token_ids])
        logits, cache = self.model.forward_with_cache(input_ids)
        handle = self._store_cache(cache, token_ids=token_ids)
        return ModelOutput(logits=logits[0, -1], past_key_values=handle)

    @torch.inference_mode()
    def prefill_chunk(self, token_ids: list[int], past_key_values: object | None = None) -> ModelOutput:
        if not token_ids:
            raise ValueError("token_ids must not be empty")
        input_ids = self._token_tensor([token_ids])
        if past_key_values is None:
            logits, cache = self.model.forward_with_cache(input_ids)
            handle = self._store_cache(cache, token_ids=token_ids)
            return ModelOutput(logits=logits[0, -1], past_key_values=handle)
        handle = self._expect_handle(past_key_values)
        cache = self._load_cache(handle)
        logits, cache = self.model.forward_with_cache(input_ids, past_key_values=cache)
        new_handle = self._store_cache(cache, seq_id=handle.seq_id, append_from=handle.seq_len)
        return ModelOutput(logits=logits[0, -1], past_key_values=new_handle)

    @torch.inference_mode()
    def prefill_batch(self, batch_token_ids: list[list[int]]) -> BatchModelOutput:
        if not batch_token_ids:
            return BatchModelOutput(logits=[], past_key_values=[])
        lengths = [len(token_ids) for token_ids in batch_token_ids]
        max_len = max(lengths)
        pad_token_id = getattr(getattr(self, "tokenizer", None), "pad_token_id", 0) or 0
        input_ids = self._full_token_tensor((len(batch_token_ids), max_len), pad_token_id)
        for batch_index, token_ids in enumerate(batch_token_ids):
            input_ids[batch_index, : len(token_ids)] = self._token_tensor(token_ids)
        logits, cache = self.model.forward_with_cache(input_ids)
        handles = [
            self._store_cache(self._slice_cache(cache, batch_index, seq_len=length), token_ids=batch_token_ids[batch_index])
            for batch_index, length in enumerate(lengths)
        ]
        return BatchModelOutput(
            logits=[logits[batch_index, length - 1] for batch_index, length in enumerate(lengths)],
            past_key_values=handles,
        )

    @torch.inference_mode()
    def decode_step(self, token_id: int, past_key_values: KVCacheHandle) -> ModelOutput:
        input_ids = self._token_tensor([[token_id]])
        cache = self._load_cache(past_key_values)
        logits, cache = self.model.forward_with_cache(input_ids, past_key_values=cache)
        handle = self._store_cache(
            cache,
            seq_id=past_key_values.seq_id,
            append_from=past_key_values.seq_len,
        )
        return ModelOutput(logits=logits[0, -1], past_key_values=handle)

    @torch.inference_mode()
    def decode_batch(
        self,
        token_ids: list[int],
        past_key_values: list[object | None],
    ) -> BatchModelOutput:
        if not token_ids:
            return BatchModelOutput(logits=[], past_key_values=[])
        outputs: list[ModelOutput | None] = [None] * len(token_ids)
        for indices in self._bucket_decode_handles(past_key_values).values():
            bucket_token_ids = [token_ids[index] for index in indices]
            bucket_handles = [past_key_values[index] for index in indices]
            bucket_out = self._decode_same_length_batch(bucket_token_ids, bucket_handles)
            for index, bucket_index in enumerate(indices):
                outputs[bucket_index] = ModelOutput(
                    logits=bucket_out.logits[index],
                    past_key_values=bucket_out.past_key_values[index],
                )
        if any(output is None for output in outputs):
            raise RuntimeError("internal error: missing decode output")
        return BatchModelOutput(
            logits=[output.logits for output in outputs if output is not None],
            past_key_values=[
                output.past_key_values
                for output in outputs
                if output is not None
            ],
        )

    @torch.inference_mode()
    def _decode_same_length_batch(
        self,
        token_ids: list[int],
        past_key_values: list[object | None],
    ) -> BatchModelOutput:
        handles = [self._expect_handle(handle) for handle in past_key_values]
        input_ids = self._token_tensor([[token_id] for token_id in token_ids])
        cache = self._load_batch_cache(handles)
        logits, cache = self.model.forward_with_cache(input_ids, past_key_values=cache)
        new_handles = [
            self._store_cache(
                self._slice_cache(cache, batch_index),
                seq_id=handle.seq_id,
                append_from=handle.seq_len,
            )
            for batch_index, handle in enumerate(handles)
        ]
        return BatchModelOutput(
            logits=[logits[batch_index, -1] for batch_index in range(len(token_ids))],
            past_key_values=new_handles,
        )

    def release_cache(self, past_key_values: object | None) -> None:
        if isinstance(past_key_values, KVCacheHandle):
            self.kv_cache.release(past_key_values.seq_id)

    def _store_cache(
        self,
        cache: Qwen3KVCache,
        seq_id: int | None = None,
        append_from: int = 0,
        token_ids: list[int] | None = None,
    ) -> KVCacheHandle:
        seq_id = next(self._cache_ids) if seq_id is None else seq_id
        for layer_id, (key, value) in enumerate(cache.key_values):
            self.kv_cache.append(
                seq_id=seq_id,
                layer_id=layer_id,
                key=key[:, :, append_from:, :],
                value=value[:, :, append_from:, :],
            )
        return KVCacheHandle(seq_id=seq_id, seq_len=cache.seq_len)

    def _load_cache(self, handle: KVCacheHandle) -> Qwen3KVCache:
        layer_kvs = self.kv_cache.layers(handle.seq_id)
        return Qwen3KVCache(
            key_values=[(layer.key, layer.value) for layer in layer_kvs],
            seq_len=handle.seq_len,
        )

    def _load_batch_cache(self, handles: list[KVCacheHandle]) -> Qwen3KVCache:
        if not handles:
            raise ValueError("handles must not be empty")
        seq_len = handles[0].seq_len
        if any(handle.seq_len != seq_len for handle in handles):
            raise ValueError("all handles must have the same seq_len")
        per_seq = [self._load_cache(handle) for handle in handles]
        key_values = []
        for layer_id in range(len(per_seq[0].key_values)):
            keys = [cache.key_values[layer_id][0] for cache in per_seq]
            values = [cache.key_values[layer_id][1] for cache in per_seq]
            key_values.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        return Qwen3KVCache(key_values=key_values, seq_len=seq_len)

    def _slice_cache(self, cache: Qwen3KVCache, batch_index: int, seq_len: int | None = None) -> Qwen3KVCache:
        seq_len = cache.seq_len if seq_len is None else seq_len
        return Qwen3KVCache(
            key_values=[
                (key[batch_index : batch_index + 1, :, :seq_len], value[batch_index : batch_index + 1, :, :seq_len])
                for key, value in cache.key_values
            ],
            seq_len=seq_len,
        )

    def _bucket_decode_handles(self, past_key_values: list[object | None]) -> dict[int, list[int]]:
        buckets: dict[int, list[int]] = {}
        for index, handle in enumerate(past_key_values):
            kv_handle = self._expect_handle(handle)
            buckets.setdefault(kv_handle.seq_len, []).append(index)
        return buckets

    def _expect_handle(self, handle: object | None) -> KVCacheHandle:
        if not isinstance(handle, KVCacheHandle):
            raise TypeError(f"expected KVCacheHandle, got {type(handle).__name__}")
        return handle

    def _token_tensor(self, data) -> torch.Tensor:
        tensor = torch.tensor(data, dtype=torch.long)
        return self._to_device(tensor)

    def _full_token_tensor(self, shape: tuple[int, int], fill_value: int) -> torch.Tensor:
        tensor = torch.full(shape, fill_value, dtype=torch.long)
        return self._to_device(tensor)

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        with phase_range("h2d"):
            # TODO_BEGIN(W10_T01)
            return tensor.to(self.device)
            # TODO_END(W10_T01)
