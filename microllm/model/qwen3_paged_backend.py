from __future__ import annotations

import torch

from microllm.engine.kv_cache import KVCacheHandle
from microllm.engine.paged_kv_cache import PagedKVCache, PagedKVConfig
from microllm.model.attention import paged_attention_decode
from microllm.model.qwen3_continuous_backend import Qwen3TorchBackend
from microllm.model.qwen3_torch import Qwen3KVCache, apply_rotary_pos_emb
from microllm.model.types import BatchModelOutput, ModelOutput


class Qwen3PagedBackend(Qwen3TorchBackend):
    """Qwen3 runner that stores KV tensors in physical paged slots."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        trust_remote_code: bool = True,
        num_blocks: int = 512,
        block_size: int = 16,
        kernel_impl: str = "torch",
        use_pinned_memory: bool = False,
        use_transfer_stream: bool = False,
    ):
        super().__init__(
            model_path,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
            kernel_impl=kernel_impl,
            use_pinned_memory=use_pinned_memory,
            use_transfer_stream=use_transfer_stream,
        )
        config = self.model.config
        self.kv_cache = PagedKVCache(
            PagedKVConfig(
                num_layers=config.num_hidden_layers,
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
        )

    def max_context_tokens(self) -> int:
        return min(
            self.model.config.max_position_embeddings,
            self.kv_cache.config.num_blocks * self.kv_cache.config.block_size,
        )

    def release_cache(self, past_key_values: object | None) -> None:
        if isinstance(past_key_values, KVCacheHandle):
            self.kv_cache.release(past_key_values.seq_id)

    @torch.inference_mode()
    def decode_step(self, token_id: int, past_key_values: KVCacheHandle) -> ModelOutput:
        out = self.decode_batch([token_id], [past_key_values])
        return ModelOutput(logits=out.logits[0], past_key_values=out.past_key_values[0])

    def cache_stats(self, handle: KVCacheHandle | None) -> dict:
        stats = self.kv_cache.usage_stats()
        if handle is None:
            return {"mode": "paged", **stats, "active": False}
        config = self.kv_cache.config
        bytes_per_token = config.num_layers * 2 * config.num_kv_heads * config.head_dim * config.dtype.itemsize
        device_bytes = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        return {
            "mode": "paged",
            "active": True,
            "seq_id": handle.seq_id,
            "seq_len": handle.seq_len,
            "block_size": config.block_size,
            "block_table": self.kv_cache.block_table(handle.seq_id),
            "num_layers": config.num_layers,
            "num_kv_heads": config.num_kv_heads,
            "head_dim": config.head_dim,
            "dtype": str(config.dtype),
            "device": str(config.device),
            "key_shape": list(self.kv_cache.key_cache.shape),
            "value_shape": list(self.kv_cache.value_cache.shape),
            "bytes_per_token": bytes_per_token,
            "estimated_bytes": handle.seq_len * bytes_per_token,
            "device_allocated_bytes": device_bytes,
            **stats,
        }

    @torch.inference_mode()
    def decode_batch(
        self,
        token_ids: list[int],
        past_key_values: list[object | None],
    ) -> BatchModelOutput:
        if not token_ids:
            return BatchModelOutput(logits=[], past_key_values=[])
        if len(token_ids) != len(past_key_values):
            raise ValueError("token_ids and past_key_values must have the same length")
        handles = [self._expect_handle(handle) for handle in past_key_values]
        outputs = self._decode_paged_batch(token_ids, handles)
        new_handles = [
            KVCacheHandle(seq_id=handle.seq_id, seq_len=handle.seq_len + 1)
            for handle in handles
        ]
        return BatchModelOutput(
            logits=[outputs[batch_index] for batch_index in range(len(token_ids))],
            past_key_values=new_handles,
        )

    def _decode_paged_batch(
        self,
        token_ids: list[int],
        handles: list[KVCacheHandle],
    ) -> torch.Tensor:
        if len({handle.seq_id for handle in handles}) != len(handles):
            raise ValueError("paged decode batch cannot contain duplicate sequence handles")

        batch_size = len(token_ids)
        input_ids = self._token_tensor([[token_id] for token_id in token_ids])
        hidden_states = self.model.model.embed_tokens(input_ids)
        position_ids = torch.tensor(
            [[handle.seq_len] for handle in handles],
            dtype=torch.long,
            device=self.device,
        )
        cos, sin = self.model.model.rotary_emb(hidden_states, position_ids)
        write_positions = {
            handle.seq_id: self.kv_cache.reserve(seq_id=handle.seq_id, num_new_tokens=1)
            for handle in handles
        }
        block_tables = [self.kv_cache.block_table(handle.seq_id) for handle in handles]
        context_lens = [handle.seq_len + 1 for handle in handles]

        for layer_id, layer in enumerate(self.model.model.layers):
            residual = hidden_states
            layer_input = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
            query = attn.q_proj(layer_input).view(batch_size, 1, attn.num_heads, attn.head_dim)
            key = attn.k_proj(layer_input).view(batch_size, 1, attn.num_kv_heads, attn.head_dim)
            value = attn.v_proj(layer_input).view(batch_size, 1, attn.num_kv_heads, attn.head_dim)

            query = attn.q_norm(query).transpose(1, 2)
            key = attn.k_norm(key).transpose(1, 2)
            value = value.transpose(1, 2)
            query, key = apply_rotary_pos_emb(query, key, cos, sin, kernel_impl=attn.kernel_impl)

            for batch_index, handle in enumerate(handles):
                self.kv_cache.write(
                    seq_id=handle.seq_id,
                    layer_id=layer_id,
                    positions=write_positions[handle.seq_id],
                    key=key[batch_index : batch_index + 1],
                    value=value[batch_index : batch_index + 1],
                )

            attn_output = paged_attention_decode(
                query=query.squeeze(-2),
                key_cache=self.kv_cache.key_cache[layer_id],
                value_cache=self.kv_cache.value_cache[layer_id],
                block_tables=block_tables,
                context_lens=context_lens,
                block_size=self.kv_cache.config.block_size,
                scale=attn.scaling,
            )
            attn_output = attn_output[:, :, None, :].transpose(1, 2).contiguous()
            attn_output = attn_output.view(batch_size, 1, attn.num_heads * attn.head_dim)
            hidden_states = residual + attn.o_proj(attn_output)

            residual = hidden_states
            hidden_states = residual + layer.mlp(layer.post_attention_layernorm(hidden_states))

        hidden_states = self.model.model.norm(hidden_states)
        return self.model.lm_head(hidden_states)[:, -1]

    def _store_cache(
        self,
        cache: Qwen3KVCache,
        seq_id: int | None = None,
        append_from: int = 0,
        token_ids: list[int] | None = None,
    ) -> KVCacheHandle:
        if seq_id is None:
            seq_id = next(self._cache_ids)
            if token_ids is not None:
                self.kv_cache.allocate(seq_id=seq_id, num_tokens=cache.seq_len, token_ids=token_ids)
                positions = list(range(append_from, cache.seq_len))
                skip_shared = True
            else:
                self.kv_cache.allocate(seq_id=seq_id, num_tokens=0)
                positions = self.kv_cache.reserve(seq_id=seq_id, num_new_tokens=cache.seq_len - append_from)
                skip_shared = False
        else:
            positions = self.kv_cache.reserve(seq_id=seq_id, num_new_tokens=cache.seq_len - append_from)
            skip_shared = False
        for layer_id, (key, value) in enumerate(cache.key_values):
            self.kv_cache.write(
                seq_id=seq_id,
                layer_id=layer_id,
                positions=positions,
                key=key[:, :, append_from:, :],
                value=value[:, :, append_from:, :],
                skip_shared=skip_shared,
            )
        return KVCacheHandle(seq_id=seq_id, seq_len=cache.seq_len)

    def _load_cache(self, handle: KVCacheHandle) -> Qwen3KVCache:
        key_values = [
            self.kv_cache.get_dense(seq_id=handle.seq_id, layer_id=layer_id)
            for layer_id in range(self.model.config.num_hidden_layers)
        ]
        return Qwen3KVCache(key_values=key_values, seq_len=handle.seq_len)
