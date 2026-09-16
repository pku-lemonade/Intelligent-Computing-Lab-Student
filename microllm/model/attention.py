from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F

from microllm.kernels.errors import KernelUnavailable
from microllm.model.qwen3_torch import repeat_kv


def paged_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: Sequence[Sequence[int]] | torch.Tensor,
    context_lens: Sequence[int] | torch.Tensor,
    block_size: int,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Decode attention over vLLM-style paged KV slots.

    CUDA tensors use the microLLM CUDA extension when available. CPU tensors, or
    environments without a buildable CUDA extension, use the PyTorch reference path.
    """

    _validate_decode_inputs(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=block_size,
    )
    if query.is_cuda:
        try:
            from microllm.kernels.cuda_ops import cuda_paged_attention_decode

            return cuda_paged_attention_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=block_tables,
                context_lens=context_lens,
                block_size=block_size,
                scale=scale,
            )
        except KernelUnavailable:
            pass
    return paged_attention_decode_reference(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        context_lens=context_lens,
        block_size=block_size,
        scale=scale,
    )


def paged_attention_decode_reference(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: Sequence[Sequence[int]] | torch.Tensor,
    context_lens: Sequence[int] | torch.Tensor,
    block_size: int,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Readable PyTorch correctness oracle for paged decode attention."""

    scale = query.shape[-1] ** -0.5 if scale is None else scale

    outputs = []
    for batch_index in range(query.shape[0]):
        slots = _slots_for_sequence(
            block_table=block_tables[batch_index],
            context_len=int(context_lens[batch_index]),
            block_size=block_size,
            device=query.device,
        )
        key = key_cache.index_select(0, slots).transpose(0, 1).unsqueeze(0)
        value = value_cache.index_select(0, slots).transpose(0, 1).unsqueeze(0)
        num_kv_groups = query.shape[1] // key.shape[1]
        key = repeat_kv(key, num_kv_groups).squeeze(0)
        value = repeat_kv(value, num_kv_groups).squeeze(0)

        scores = torch.matmul(query[batch_index].unsqueeze(1), key.transpose(-2, -1)).squeeze(1)
        weights = F.softmax((scores * scale).float(), dim=-1).to(query.dtype)
        outputs.append(torch.matmul(weights.unsqueeze(1), value).squeeze(1))
    return torch.stack(outputs, dim=0)


def _slots_for_sequence(
    block_table: Sequence[int] | torch.Tensor,
    context_len: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(context_len, dtype=torch.long, device=device)
    block_indices = positions // block_size
    block_offsets = positions % block_size
    blocks = torch.as_tensor(block_table, dtype=torch.long, device=device)
    return blocks.index_select(0, block_indices) * block_size + block_offsets


def _validate_decode_inputs(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: Sequence[Sequence[int]] | torch.Tensor,
    context_lens: Sequence[int] | torch.Tensor,
    block_size: int,
) -> None:
    if query.ndim != 3:
        raise ValueError(f"expected query shape [batch, heads, dim], got {tuple(query.shape)}")
    if key_cache.ndim != 3:
        raise ValueError(f"expected key_cache shape [slots, kv_heads, dim], got {tuple(key_cache.shape)}")
    if value_cache.shape != key_cache.shape:
        raise ValueError(f"key/value cache shape mismatch: {tuple(key_cache.shape)} vs {tuple(value_cache.shape)}")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    if len(block_tables) != query.shape[0]:
        raise ValueError("block_tables batch size must match query batch size")
    if len(context_lens) != query.shape[0]:
        raise ValueError("context_lens batch size must match query batch size")
    if query.shape[-1] != key_cache.shape[-1]:
        raise ValueError("query and key/value head_dim must match")
    if query.device != key_cache.device or query.device != value_cache.device:
        raise ValueError("query and key/value cache tensors must be on the same device")
    if query.shape[1] % key_cache.shape[1] != 0:
        raise ValueError("query heads must be divisible by KV heads")
    for batch_index, context_len in enumerate(context_lens):
        if int(context_len) <= 0:
            raise ValueError("context lengths must be > 0")
        required_blocks = (int(context_len) + block_size - 1) // block_size
        if len(block_tables[batch_index]) < required_blocks:
            raise ValueError("block table does not cover context length")
