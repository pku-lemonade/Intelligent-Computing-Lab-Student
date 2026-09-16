from __future__ import annotations

import math

import torch

from microllm.kernels import assert_close, cuda_dense_attention_decode, cuda_dense_attention_prefill
from microllm.kernels.errors import KernelUnavailable
from microllm.model.ops import (
    dense_attention_decode_reference,
    dense_attention_prefill,
    dense_attention_prefill_reference,
)


def test_w09_t01_prefill_dispatch_uses_requested_cuda_path(monkeypatch):
    _require_cuda()
    calls = []
    sentinel = torch.randn((1, 2, 3, 8), device="cuda")

    def fake_attention(query, key, value, *, scale):
        calls.append((query.shape, key.shape, value.shape, scale))
        return sentinel

    monkeypatch.setattr("microllm.kernels.cuda_ops.cuda_dense_attention_prefill", fake_attention)
    q = torch.randn((1, 2, 3, 8), device="cuda")

    actual = dense_attention_prefill(q, q, q, scale=0.5, kernel_impl="cuda")

    assert actual is sentinel
    assert calls == [(q.shape, q.shape, q.shape, 0.5)]


def test_w09_t01_prefill_honors_torch_and_auto_fallback(monkeypatch):
    q = torch.randn((1, 2, 4, 8))
    expected = dense_attention_prefill_reference(q, q, q, scale=0.5)
    assert_close(
        "W09_T01[torch]",
        dense_attention_prefill(q, q, q, scale=0.5, kernel_impl="torch"),
        expected,
    )

    _require_cuda()
    q = q.cuda()
    monkeypatch.setattr(
        "microllm.kernels.cuda_ops.cuda_dense_attention_prefill",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KernelUnavailable("test fallback")),
    )
    assert_close(
        "W09_T01[auto]",
        dense_attention_prefill(q, q, q, scale=0.5, kernel_impl="auto"),
        dense_attention_prefill_reference(q, q, q, scale=0.5),
    )


def test_w09_t02_causal_prefill_attention_matches_reference():
    _require_cuda()
    q = torch.randn((2, 3, 7, 32), device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(q.shape[-1])
    actual = cuda_dense_attention_prefill(q, k, v, scale=scale)
    expected = dense_attention_prefill_reference(q, k, v, scale=scale)
    assert_close("W09_T02", actual, expected, atol=1e-4, rtol=1e-4)


def test_w09_t03_dense_decode_attention_matches_reference():
    _require_cuda()
    q = torch.randn((2, 3, 32), device="cuda")
    k = torch.randn((2, 3, 11, 32), device="cuda")
    v = torch.randn_like(k)
    scale = 1.0 / math.sqrt(q.shape[-1])
    actual = cuda_dense_attention_decode(q, k, v, scale=scale)
    expected = dense_attention_decode_reference(q, k, v, scale=scale)
    assert_close("W09_T03", actual, expected, atol=1e-4, rtol=1e-4)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Week 9 public tests require CUDA")
