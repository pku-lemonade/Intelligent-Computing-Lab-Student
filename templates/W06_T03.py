from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from microllm.kernels.errors import KernelUnavailable


class CourseLinear(nn.Module):
    """Linear layer with a teaching CUDA matmul dispatch point."""

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False, kernel_impl: str = "torch"):
        super().__init__()
        if kernel_impl not in {"torch", "auto", "cuda"}:
            raise ValueError("kernel_impl must be one of: torch, auto, cuda")
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_impl = kernel_impl
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO_BEGIN(W06_T03)
        return x
        # TODO_END(W06_T03)


def dense_attention_prefill_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    block_size: int | None = None,
) -> torch.Tensor:
    """Causal prefill attention using online softmax over key tiles.

    The implementation is intentionally readable rather than fast. It avoids
    materializing the full [seq, seq] score matrix and mirrors the recurrence
    used by FlashAttention-style kernels.
    """

    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("expected Q/K/V to have the same [batch, heads, seq, dim] shape")
    batch, heads, seq_len, head_dim = query.shape
    tile = block_size or max(1, seq_len)
    out = torch.empty_like(query)
    for pos in range(seq_len):
        q = query[:, :, pos : pos + 1, :].float()
        row_max = torch.full((batch, heads, 1), -torch.inf, device=query.device, dtype=torch.float32)
        row_sum = torch.zeros((batch, heads, 1), device=query.device, dtype=torch.float32)
        acc = torch.zeros((batch, heads, head_dim), device=query.device, dtype=torch.float32)
        for start in range(0, pos + 1, tile):
            end = min(pos + 1, start + tile)
            scores = torch.matmul(q, key[:, :, start:end, :].float().transpose(-2, -1)).squeeze(-2) * scale
            tile_max = scores.max(dim=-1, keepdim=True).values
            next_max = torch.maximum(row_max, tile_max)
            old_scale = torch.exp(row_max - next_max)
            probs = torch.exp(scores - next_max)
            acc = acc * old_scale + torch.matmul(probs.unsqueeze(-2), value[:, :, start:end, :].float()).squeeze(-2)
            row_sum = row_sum * old_scale + probs.sum(dim=-1, keepdim=True)
            row_max = next_max
        out[:, :, pos, :] = (acc / row_sum).to(query.dtype)
    return out


def dense_attention_prefill(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    kernel_impl: str = "torch",
    block_size: int | None = None,
) -> torch.Tensor:
    # TODO_BEGIN(W09_T01)
    return dense_attention_prefill_reference(
        query, key, value, scale=scale, block_size=block_size
    )
    # TODO_END(W09_T01)


def dense_attention_decode_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    if query.ndim != 3 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("expected query [batch, heads, dim] and K/V [batch, heads, seq, dim]")
    if query.shape[0] != key.shape[0] or query.shape[1] != key.shape[1] or query.shape[2] != key.shape[3]:
        raise ValueError("query and K/V shapes must align")
    scores = torch.matmul(query.unsqueeze(2), key.float().transpose(-2, -1)).squeeze(2) * scale
    weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
    return torch.matmul(weights.unsqueeze(2), value).squeeze(2)


def dense_attention_decode(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    scale: float,
    kernel_impl: str = "torch",
) -> torch.Tensor:
    if kernel_impl in {"auto", "cuda"} and query.is_cuda:
        try:
            from microllm.kernels.cuda_ops import cuda_dense_attention_decode

            return cuda_dense_attention_decode(query, key, value, scale=scale)
        except KernelUnavailable:
            if kernel_impl == "cuda":
                raise
    return dense_attention_decode_reference(query, key, value, scale=scale)
