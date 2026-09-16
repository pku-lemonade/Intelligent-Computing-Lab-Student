from __future__ import annotations

from functools import lru_cache

import torch

from microllm.kernels import assert_close, load_cuda_extension


def test_w04_t01_saxpy_grid_stride_covers_the_full_input():
    module = _saxpy_module()
    length = 4097
    x = torch.randn(length, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    alpha = -0.75

    actual = module.saxpy_grid_stride(x, y, alpha, 2)

    assert_close("W04_T01", actual, alpha * x + y)


def test_w04_t02_fused_elementwise_matches_unfused_semantics():
    module = _elementwise_module()
    for length, scale, bias in ((0, 2.0, 1.0), (1, -1.0, 0.25), (257, 0.5, -0.75), (4099, 1.25, 2.0)):
        x = torch.randn(length, device="cuda", dtype=torch.float32)
        expected = torch.relu(scale * x + bias)
        fused = module.fused_scale_bias_relu(x, scale, bias)
        unfused = module.unfused_scale_bias_relu(x, scale, bias)
        assert_close(f"W04_T02[fused,n={length}]", fused, expected)
        assert_close(f"W04_T02[unfused,n={length}]", unfused, expected)


@lru_cache(maxsize=1)
def _saxpy_module():
    _require_cuda()
    return load_cuda_extension("microllm_week04_saxpy", ("kernels/saxpy.cu",))


@lru_cache(maxsize=1)
def _elementwise_module():
    _require_cuda()
    return load_cuda_extension("microllm_week04_elementwise", ("kernels/elementwise.cu",))


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Week 4 public tests require a CUDA-capable evaluation worker")
