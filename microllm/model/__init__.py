__all__ = [
    "HFModelBackend",
    "Qwen3ForCausalLM",
    "Qwen3PagedBackend",
    "Qwen3TorchBackend",
    "paged_attention_decode",
]


def __getattr__(name: str):
    if name == "paged_attention_decode":
        from microllm.model.attention import paged_attention_decode

        return paged_attention_decode
    if name == "HFModelBackend":
        from microllm.model.hf_backend import HFModelBackend

        return HFModelBackend
    if name in {"Qwen3PagedBackend", "Qwen3TorchBackend"}:
        from microllm.model import qwen3_backend

        return getattr(qwen3_backend, name)
    if name == "Qwen3ForCausalLM":
        from microllm.model.qwen3_torch import Qwen3ForCausalLM

        return Qwen3ForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
