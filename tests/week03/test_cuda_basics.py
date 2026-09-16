from __future__ import annotations

from functools import lru_cache

import torch

from microllm.kernels import assert_close, load_cuda_extension


def test_w03_t01_vector_add_kernel():
    module = _vector_add_module()
    for length in (1, 255, 257, 4097):
        a = torch.randn(length, device="cuda", dtype=torch.float32)
        b = torch.randn(length, device="cuda", dtype=torch.float32)
        assert_close(f"W03_T01[n={length}]", module.vector_add(a, b), a + b)


def test_w03_t02_vector_add_launcher():
    module = _vector_add_module()
    empty = torch.empty(0, device="cuda", dtype=torch.float32)
    assert module.vector_add(empty, empty).numel() == 0
    a = torch.randn(1025, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    assert_close("W03_T02[tail]", module.vector_add(a, b), a + b)


def test_w03_t03_saxpy_kernel():
    module = _saxpy_module()
    for length, alpha in ((1, 0.0), (17, -1.5), (256, 2.0), (1025, 0.125)):
        x = torch.randn(length, device="cuda", dtype=torch.float32)
        y = torch.randn(length, device="cuda", dtype=torch.float32)
        expected = alpha * x + y
        assert_close(f"W03_T03[n={length}]", module.saxpy(x, y, alpha), expected)


@lru_cache(maxsize=1)
def _vector_add_module():
    _require_cuda()
    return load_cuda_extension("microllm_week03_vector_add", ("kernels/vector_add.cu",))


@lru_cache(maxsize=1)
def _saxpy_module():
    _require_cuda()
    return load_cuda_extension("microllm_week03_saxpy", ("kernels/saxpy.cu",))


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Week 3 public tests require a CUDA-capable evaluation worker")
