from functools import lru_cache

import torch

from microllm.kernels import assert_close, load_cuda_extension


def test_w07_t01_warp_sum_reduces_partial_warp():
    module = _module()
    # 17 columns < 32 lanes, so lanes 17..31 must contribute the neutral element
    # without dropping out of the shuffle.
    value = torch.randn((4, 17), device="cuda")
    assert_close("W07_T01", module.warp_sum_probe(value), value.sum(dim=-1), atol=1e-5, rtol=1e-5)


def test_w07_t02_row_sum_handles_multiple_warps():
    module = _module()
    for cols in (33, 255, 513, 1025):
        value = torch.randn((5, cols), device="cuda")
        assert_close(f"W07_T02[cols={cols}]", module.row_sum(value), value.sum(dim=-1), atol=2e-4, rtol=2e-4)


def test_w07_t03_row_max_handles_negative_values():
    module = _module()
    for cols in (1, 17, 257, 513):
        value = -torch.rand((3, cols), device="cuda") - 0.5
        assert_close(f"W07_T03[cols={cols}]", module.row_max(value), value.max(dim=-1).values)


@lru_cache(maxsize=1)
def _module():
    if not torch.cuda.is_available():
        raise RuntimeError("Week 7 public tests require CUDA")
    return load_cuda_extension("microllm_week07_reduction", ("kernels/reduction.cu",))
