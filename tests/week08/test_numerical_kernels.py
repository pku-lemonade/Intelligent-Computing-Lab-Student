from __future__ import annotations

import torch

from microllm.kernels import assert_close, cuda_rms_norm, cuda_softmax
from microllm.kernels.errors import KernelUnavailable
from microllm.model.qwen3_torch import Qwen3RMSNorm


def test_w08_t01_softmax_is_stable_for_extreme_logits():
    _require_cuda()
    x = torch.tensor(
        [[10000.0, 9999.0, 9998.0], [-10000.0, -10001.0, -9999.0]],
        device="cuda",
    )
    actual = cuda_softmax(x)
    expected = torch.softmax(x, dim=-1)
    assert torch.isfinite(actual).all()
    assert_close("W08_T01", actual, expected, atol=1e-5, rtol=1e-5)

    low_precision = torch.linspace(-80, 80, 257, device="cuda", dtype=torch.float32)
    low_precision = torch.stack((low_precision, torch.zeros_like(low_precision))).bfloat16()
    actual = cuda_softmax(low_precision)
    expected = torch.softmax(low_precision.float(), dim=-1).bfloat16()
    assert torch.isfinite(actual).all()
    assert_close("W08_T01[bfloat16]", actual, expected, atol=2e-2, rtol=2e-2)


def test_w08_t02_rms_norm_matches_float_accumulation():
    _require_cuda()
    for dtype, tolerance in ((torch.float32, 1e-5), (torch.bfloat16, 2e-2)):
        x = torch.randn((3, 513), device="cuda", dtype=dtype)
        weight = torch.randn(513, device="cuda", dtype=dtype)
        actual = cuda_rms_norm(x, weight, eps=1e-6)
        expected = (x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + 1e-6) * weight.float()).to(dtype)
        assert_close(f"W08_T02[{dtype}]", actual, expected, atol=tolerance, rtol=tolerance)


def test_w08_t03_qwen_rms_norm_dispatches_cuda(monkeypatch):
    _require_cuda()
    calls = []

    def fake_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float):
        calls.append((x.shape, weight.shape, eps))
        return torch.full_like(x, 3.0)

    monkeypatch.setattr("microllm.kernels.cuda_ops.cuda_rms_norm", fake_rms_norm)
    layer = Qwen3RMSNorm(8, 1e-6, kernel_impl="cuda").cuda()
    x = torch.randn((2, 4, 8), device="cuda")

    actual = layer(x)

    assert calls == [(torch.Size([8, 8]), torch.Size([8]), 1e-6)]
    assert torch.equal(actual, torch.full_like(x, 3.0))


def test_w08_t03_qwen_rms_norm_honors_torch_and_auto_fallback(monkeypatch):
    cpu_layer = Qwen3RMSNorm(9, 1e-6, kernel_impl="torch")
    cpu_x = torch.randn((2, 4, 9))
    expected = cpu_layer.weight * (
        cpu_x.float()
        * torch.rsqrt(cpu_x.float().square().mean(dim=-1, keepdim=True) + cpu_layer.eps)
    ).to(cpu_x.dtype)
    assert_close("W08_T03[torch]", cpu_layer(cpu_x), expected)

    _require_cuda()
    layer = Qwen3RMSNorm(9, 1e-6, kernel_impl="auto").cuda()
    x = torch.randn((2, 3, 9), device="cuda")
    monkeypatch.setattr(
        "microllm.kernels.cuda_ops.cuda_rms_norm",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KernelUnavailable("test fallback")),
    )
    expected = layer.weight * (
        x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + layer.eps)
    ).to(x.dtype)
    assert_close("W08_T03[auto]", layer(x), expected)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Week 8 public tests require CUDA")
