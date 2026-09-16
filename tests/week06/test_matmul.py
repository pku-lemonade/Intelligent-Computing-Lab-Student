from __future__ import annotations

import torch

from microllm.kernels import assert_close, cuda_matmul, cuda_matmul_tiled
from microllm.kernels.errors import KernelUnavailable
from microllm.model.ops import CourseLinear


def test_w06_t01_naive_matmul_matches_torch():
    _require_cuda()
    for m, k, n in ((1, 1, 1), (3, 5, 7), (19, 33, 11)):
        a = torch.randn((m, k), device="cuda")
        b = torch.randn((k, n), device="cuda")
        assert_close(f"W06_T01[{m},{k},{n}]", cuda_matmul(a, b), a @ b, atol=1e-3, rtol=1e-3)
    a = torch.randn((7, 13), device="cuda", dtype=torch.float16)
    b = torch.randn((13, 5), device="cuda", dtype=torch.float16)
    assert_close("W06_T01[float16]", cuda_matmul(a, b), a @ b, atol=2e-2, rtol=2e-2)


def test_w06_t02_tiled_matmul_handles_tail_tiles():
    _require_cuda()
    for m, k, n in ((2, 17, 3), (31, 65, 17), (33, 47, 35)):
        a = torch.randn((m, k), device="cuda")
        b = torch.randn((k, n), device="cuda")
        assert_close(f"W06_T02[{m},{k},{n}]", cuda_matmul_tiled(a, b), a @ b, atol=1e-3, rtol=1e-3)
    a = torch.randn((9, 19), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((19, 11), device="cuda", dtype=torch.bfloat16)
    assert_close("W06_T02[bfloat16]", cuda_matmul_tiled(a, b), a @ b, atol=5e-2, rtol=5e-2)


def test_w06_t03_course_linear_dispatches_tiled_matmul(monkeypatch):
    _require_cuda()
    calls = []

    def fake_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        calls.append((a.shape, b.shape))
        return a @ b

    monkeypatch.setattr("microllm.kernels.cuda_ops.cuda_matmul_tiled", fake_matmul)
    layer = CourseLinear(7, 5, bias=True, kernel_impl="cuda").cuda()
    x = torch.randn((2, 3, 7), device="cuda")
    expected = torch.nn.functional.linear(x, layer.weight, layer.bias)

    actual = layer(x)

    assert calls == [(torch.Size([6, 7]), torch.Size([7, 5]))]
    assert_close("W06_T03", actual, expected)


def test_w06_t03_course_linear_honors_torch_and_auto_fallback(monkeypatch):
    cpu_layer = CourseLinear(7, 5, bias=True, kernel_impl="torch")
    cpu_x = torch.randn((2, 3, 7))
    assert_close(
        "W06_T03[torch]",
        cpu_layer(cpu_x),
        torch.nn.functional.linear(cpu_x, cpu_layer.weight, cpu_layer.bias),
    )

    _require_cuda()
    layer = CourseLinear(7, 5, bias=False, kernel_impl="auto").cuda()
    x = torch.randn((4, 7), device="cuda")
    monkeypatch.setattr(
        "microllm.kernels.cuda_ops.cuda_matmul_tiled",
        lambda *_: (_ for _ in ()).throw(KernelUnavailable("test fallback")),
    )
    assert_close("W06_T03[auto]", layer(x), torch.nn.functional.linear(x, layer.weight))


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Week 6 public tests require CUDA")
