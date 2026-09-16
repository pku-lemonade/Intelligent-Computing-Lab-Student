from microllm.stages import all_stage_overviews, stage_overview

__all__ = ["Engine", "SamplingParams", "all_stage_overviews", "stage_overview"]


def __getattr__(name: str):
    if name == "Engine":
        from microllm.engine.engine import Engine

        return Engine
    if name == "SamplingParams":
        from microllm.engine.sampler import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
