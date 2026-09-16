from __future__ import annotations

from collections.abc import Sequence

import torch

from microllm.kernels.harness import KernelUnavailable, load_cuda_extension


def cuda_softmax(x: torch.Tensor) -> torch.Tensor:
    return _module().softmax(_require_cuda("x", x).contiguous())


def cuda_rms_norm(x: torch.Tensor, weight: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    x = _require_cuda("x", x).contiguous()
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if weight.ndim != 1 or weight.shape[0] != x.shape[1]:
        raise ValueError("weight must have shape [hidden_size]")
    return _module().rms_norm(x, weight.to(device=x.device, dtype=x.dtype).contiguous(), float(eps))


def cuda_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x = _require_cuda("x", x).contiguous()
    if x.ndim != 2:
        raise ValueError("x must be 2D")
    if x.shape[1] % 2 != 0:
        raise ValueError("RoPE head dimension must be even")
    cos = _rope_table("cos", cos, x)
    sin = _rope_table("sin", sin, x)
    return _module().rope(x, cos, sin)


def cuda_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _require_cuda("a", a).contiguous()
    b = _require_cuda("b", b).contiguous()
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul inputs must be 2D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return _module().matmul(a, b)


def cuda_matmul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _require_cuda("a", a).contiguous()
    b = _require_cuda("b", b).contiguous()
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("matmul inputs must be 2D")
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"matmul shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return _module().matmul_tiled(a, b)


def cuda_paged_attention_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: Sequence[Sequence[int]] | torch.Tensor,
    context_lens: Sequence[int] | torch.Tensor,
    block_size: int,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    query = _require_cuda("query", query).contiguous()
    key_cache = _require_cuda("key_cache", key_cache).contiguous()
    value_cache = _require_cuda("value_cache", value_cache).contiguous()
    if query.ndim != 3 or key_cache.ndim != 3 or value_cache.shape != key_cache.shape:
        raise ValueError("expected query [batch, heads, dim] and KV cache [slots, kv_heads, dim]")
    batch, num_heads, head_dim = query.shape
    _, num_kv_heads, kv_dim = key_cache.shape
    if head_dim != kv_dim or num_heads % num_kv_heads != 0:
        raise ValueError("query heads/dim must match grouped KV cache")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    tables, lens, min_context_len, max_context_len = _metadata_tensors(
        block_tables, context_lens, batch, query.device
    )
    max_context = int(tables.shape[1]) * block_size
    if max_context <= 0:
        raise ValueError("block_tables must contain at least one block")
    if min_context_len <= 0:
        raise ValueError("context lengths must be > 0")
    if max_context_len > max_context:
        raise ValueError("block table does not cover context length")
    if head_dim > 256:
        raise KernelUnavailable("teaching CUDA paged attention kernel supports head_dim <= 256")
    scale = head_dim**-0.5 if scale is None else scale
    return _module().paged_attention_decode(query, key_cache, value_cache, tables, lens, block_size, float(scale))


def cuda_dense_attention_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    query = _require_cuda("query", query).contiguous()
    key = _require_cuda("key", key).contiguous()
    value = _require_cuda("value", value).contiguous()
    if query.ndim != 4 or key.shape != query.shape or value.shape != query.shape:
        raise ValueError("expected Q/K/V shape [batch, heads, seq, dim]")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if query.shape[-1] > 256:
        raise KernelUnavailable("teaching CUDA dense attention kernel supports head_dim <= 256")
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _module().dense_attention_prefill(query, key, value, float(scale))


def cuda_dense_attention_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    query = _require_cuda("query", query).contiguous()
    key = _require_cuda("key", key).contiguous()
    value = _require_cuda("value", value).contiguous()
    if query.ndim != 3 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("expected query [batch, heads, dim] and K/V [batch, heads, seq, dim]")
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1] or query.shape[2] != key.shape[3]:
        raise ValueError("query and K/V shapes must align")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Q/K/V dtypes must match")
    if query.shape[-1] > 256:
        raise KernelUnavailable("teaching CUDA dense decode attention kernel supports head_dim <= 256")
    scale = query.shape[-1] ** -0.5 if scale is None else scale
    return _module().dense_attention_decode(query, key, value, float(scale))


def _module():
    return load_cuda_extension("microllm_cuda_ops", ("kernels/microllm_ops.cpp", "kernels/microllm_ops.cu"))


def _require_cuda(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda:
        raise KernelUnavailable(f"{name} must be a CUDA tensor")
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must be a floating point tensor")
    return tensor


def _rope_table(name: str, table: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if table.ndim == 1:
        table = table.reshape(1, -1).expand(x.shape[0], -1)
    if table.shape != x.shape:
        raise ValueError(f"{name} must have shape [dim] or match x")
    return table.to(device=x.device, dtype=x.dtype).contiguous()


def _metadata_tensors(
    block_tables: Sequence[Sequence[int]] | torch.Tensor,
    context_lens: Sequence[int] | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if torch.is_tensor(block_tables):
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            raise ValueError("block_tables tensor must have shape [batch, max_blocks]")
        tables = block_tables.to(device=device, dtype=torch.int64).contiguous()
    else:
        if len(block_tables) != batch_size:
            raise ValueError("block_tables batch size must match query batch size")
        max_blocks = max(len(table) for table in block_tables)
        tables = torch.zeros((batch_size, max_blocks), dtype=torch.int64, device=device)
        for index, table in enumerate(block_tables):
            tables[index, : len(table)] = torch.tensor(table, dtype=torch.int64, device=device)

    if torch.is_tensor(context_lens):
        if context_lens.ndim != 1 or context_lens.shape[0] != batch_size:
            raise ValueError("context_lens tensor must have shape [batch]")
        if context_lens.is_cuda:
            min_context_len = 1
            max_context_len = 0
        else:
            min_context_len = int(context_lens.min().item())
            max_context_len = int(context_lens.max().item())
        lens = context_lens.to(device=device, dtype=torch.int64).contiguous()
    else:
        if len(context_lens) != batch_size:
            raise ValueError("context_lens batch size must match query batch size")
        min_context_len = min(int(length) for length in context_lens)
        max_context_len = max(int(length) for length in context_lens)
        lens = torch.tensor(context_lens, dtype=torch.int64, device=device)
    return tables, lens, min_context_len, max_context_len


__all__ = [
    "cuda_dense_attention_decode",
    "cuda_dense_attention_prefill",
    "cuda_matmul",
    "cuda_matmul_tiled",
    "cuda_paged_attention_decode",
    "cuda_rms_norm",
    "cuda_rope",
    "cuda_softmax",
]
