from __future__ import annotations

from dataclasses import dataclass

import torch

from microllm.engine.block_manager import BlockManager


@dataclass(frozen=True, slots=True)
class PagedKVConfig:
    num_layers: int
    num_blocks: int
    block_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float32
    device: str | torch.device = "cpu"


class PagedKVCache:
    """Physical-slot KV cache backed by a vLLM-style block table."""

    def __init__(self, config: PagedKVConfig):
        if config.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if config.num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be > 0")
        if config.head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        self.config = config
        self.block_manager = BlockManager(
            num_blocks=config.num_blocks,
            block_size=config.block_size,
        )
        shape = (
            config.num_layers,
            config.num_blocks * config.block_size,
            config.num_kv_heads,
            config.head_dim,
        )
        self.key_cache = torch.empty(shape, dtype=config.dtype, device=config.device)
        self.value_cache = torch.empty(shape, dtype=config.dtype, device=config.device)

    def allocate(self, seq_id: int, num_tokens: int, *, token_ids: list[int] | None = None) -> None:
        self.block_manager.allocate(seq_id=seq_id, num_tokens=num_tokens, token_ids=token_ids)

    def append(
        self,
        seq_id: int,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._validate_layer_id(layer_id)
        self._validate_kv(key, value)
        positions = self.reserve(seq_id=seq_id, num_new_tokens=key.shape[-2])
        self.write(seq_id=seq_id, layer_id=layer_id, positions=positions, key=key, value=value)

    def reserve(self, seq_id: int, num_new_tokens: int) -> list[int]:
        # TODO_BEGIN(W12_T03)
        return []
        # TODO_END(W12_T03)

    def write(
        self,
        seq_id: int,
        layer_id: int,
        positions: list[int],
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        skip_shared: bool = False,
    ) -> None:
        self._validate_layer_id(layer_id)
        self._validate_kv(key, value)
        if key.shape[-2] != len(positions):
            raise ValueError("number of positions must match KV sequence length")
        key_tokens = key.squeeze(0).transpose(0, 1).contiguous()
        value_tokens = value.squeeze(0).transpose(0, 1).contiguous()
        if skip_shared:
            positions, key_tokens, value_tokens = self._owned_write_view(
                seq_id,
                positions,
                key_tokens,
                value_tokens,
            )
            if not positions:
                return
        # TODO_BEGIN(W12_T04)
        return None
        # TODO_END(W12_T04)

    def get_dense(self, seq_id: int, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_layer_id(layer_id)
        table = self.block_manager.tables[seq_id]
        positions = list(range(table.length))
        slots = self.block_manager.slot_mapping(seq_id, positions)
        key = self.key_cache[layer_id, slots].transpose(0, 1).unsqueeze(0).contiguous()
        value = self.value_cache[layer_id, slots].transpose(0, 1).unsqueeze(0).contiguous()
        return key, value

    def block_table(self, seq_id: int) -> list[int]:
        return self.block_manager.block_table(seq_id)

    def slot_mapping(self, seq_id: int, positions: list[int]) -> list[int]:
        return self.block_manager.slot_mapping(seq_id, positions)

    def release(self, seq_id: int) -> None:
        self.block_manager.release(seq_id)

    def usage_stats(self) -> dict:
        return self.block_manager.usage_stats()

    def _owned_write_view(
        self,
        seq_id: int,
        positions: list[int],
        key_tokens: torch.Tensor,
        value_tokens: torch.Tensor,
    ) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        table = self.block_manager.tables[seq_id]
        owned_indices = [
            index
            for index, position in enumerate(positions)
            if table.block_ids[position // self.config.block_size] in table.owned_block_ids
        ]
        if len(owned_indices) == len(positions):
            return positions, key_tokens, value_tokens
        if not owned_indices:
            return [], key_tokens[:0], value_tokens[:0]
        index_tensor = torch.tensor(owned_indices, dtype=torch.long, device=key_tokens.device)
        return (
            [positions[index] for index in owned_indices],
            key_tokens.index_select(0, index_tensor),
            value_tokens.index_select(0, index_tensor),
        )

    def _validate_layer_id(self, layer_id: int) -> None:
        if layer_id < 0 or layer_id >= self.config.num_layers:
            raise IndexError(f"layer_id {layer_id} outside [0, {self.config.num_layers})")

    def _validate_kv(self, key: torch.Tensor, value: torch.Tensor) -> None:
        expected_prefix = (1, self.config.num_kv_heads)
        expected_suffix = self.config.head_dim
        if key.shape != value.shape:
            raise ValueError(f"key/value shape mismatch: {tuple(key.shape)} vs {tuple(value.shape)}")
        if key.ndim != 4:
            raise ValueError(f"expected KV shape [1, heads, tokens, dim], got {tuple(key.shape)}")
        if key.shape[:2] != expected_prefix or key.shape[-1] != expected_suffix:
            raise ValueError(
                "expected KV shape "
                f"[1, {self.config.num_kv_heads}, tokens, {expected_suffix}], got {tuple(key.shape)}"
            )
