__all__ = ["Engine", "PagedKVCache", "PagedKVConfig", "SamplingParams"]


def __getattr__(name: str):
    if name == "Engine":
        from microllm.engine.engine import Engine

        return Engine
    if name in {"PagedKVCache", "PagedKVConfig"}:
        from microllm.engine import paged_kv_cache

        return getattr(paged_kv_cache, name)
    if name == "SamplingParams":
        from microllm.engine.sampler import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
