from microllm.kernels.errors import KernelUnavailable

__all__ = [
    "KernelUnavailable",
    "assert_close",
    "benchmark_cuda",
    "cuda_dense_attention_decode",
    "cuda_dense_attention_prefill",
    "cuda_matmul",
    "cuda_matmul_tiled",
    "cuda_paged_attention_decode",
    "cuda_rms_norm",
    "cuda_rope",
    "cuda_softmax",
    "load_cuda_extension",
]


def __getattr__(name: str):
    if name in {"assert_close", "benchmark_cuda", "load_cuda_extension"}:
        from microllm.kernels import harness

        return getattr(harness, name)
    if name in {
        "cuda_dense_attention_decode",
        "cuda_dense_attention_prefill",
        "cuda_matmul",
        "cuda_matmul_tiled",
        "cuda_paged_attention_decode",
        "cuda_rms_norm",
        "cuda_rope",
        "cuda_softmax",
    }:
        from microllm.kernels import cuda_ops

        return getattr(cuda_ops, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
