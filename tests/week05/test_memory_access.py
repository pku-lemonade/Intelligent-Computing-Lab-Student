from __future__ import annotations

from functools import lru_cache

import torch

from microllm.kernels import assert_close, load_cuda_extension


def test_w05_t01_coalesced_copy_preserves_the_matrix():
    module = _module()
    for shape in ((0, 7), (1, 1), (3, 5), (33, 65)):
        value = torch.randn(shape, device="cuda", dtype=torch.float32)
        assert_close(f"W05_T01[shape={shape}]", module.copy_coalesced(value), value)


def test_w05_t02_strided_copy_preserves_the_matrix():
    module = _module()
    for shape in ((1, 17), (17, 1), (3, 5), (31, 67)):
        value = torch.randn(shape, device="cuda", dtype=torch.float32)
        assert_close(f"W05_T02[shape={shape}]", module.copy_strided(value), value)


def test_w05_t03_naive_transpose_handles_rectangular_matrices():
    module = _module()
    for shape in ((0, 7), (1, 19), (3, 5), (17, 33), (32, 65)):
        value = torch.randn(shape, device="cuda", dtype=torch.float32)
        expected = value.transpose(0, 1).contiguous()
        actual = module.transpose_naive(value)
        assert actual.shape == expected.shape
        assert_close(f"W05_T03[shape={shape}]", actual, expected)


@lru_cache(maxsize=1)
def _module():
    if not torch.cuda.is_available():
        raise RuntimeError("Week 5 public tests require a CUDA-capable evaluation worker")
    return load_cuda_extension("microllm_week05_memory", ("kernels/memory_access.cu",))
