from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import mean


@dataclass(slots=True)
class RequestTrace:
    request_id: str
    arrived_at: float
    token_timestamps: list[float] = field(default_factory=list)
    finished_at: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.arrived_at):
            raise ValueError("arrival timestamp must be finite")

    def observe_token(self, timestamp: float) -> None:
        if self.finished_at is not None:
            raise RuntimeError("cannot observe a token after the request finished")
        if not math.isfinite(timestamp):
            raise ValueError("token timestamp must be finite")
        lower_bound = self.token_timestamps[-1] if self.token_timestamps else self.arrived_at
        if timestamp < lower_bound:
            raise ValueError("token timestamps must be monotonic")
        self.token_timestamps.append(timestamp)

    def finish(self, timestamp: float) -> None:
        if self.finished_at is not None:
            raise RuntimeError("request is already finished")
        if not math.isfinite(timestamp):
            raise ValueError("finish timestamp must be finite")
        lower_bound = self.token_timestamps[-1] if self.token_timestamps else self.arrived_at
        if timestamp < lower_bound:
            raise ValueError("finish timestamp must not precede request events")
        self.finished_at = timestamp

    @property
    def output_tokens(self) -> int:
        return len(self.token_timestamps)

    @property
    def ttft_s(self) -> float | None:
        if not self.token_timestamps:
            return None
        return self.token_timestamps[0] - self.arrived_at

    @property
    def inter_token_latencies_s(self) -> list[float]:
        return [
            right - left
            for left, right in zip(self.token_timestamps, self.token_timestamps[1:])
        ]

    @property
    def tpot_s(self) -> float | None:
        intervals = self.inter_token_latencies_s
        return mean(intervals) if intervals else None

    @property
    def e2e_latency_s(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.arrived_at


def summarize_serving_metrics(
    traces: list[RequestTrace],
    *,
    window_start: float,
    window_end: float,
) -> dict[str, float | int]:
    if not math.isfinite(window_start) or not math.isfinite(window_end):
        raise ValueError("measurement window must be finite")
    elapsed_s = window_end - window_start
    if elapsed_s <= 0:
        raise ValueError("window_end must be greater than window_start")

    completed = [trace for trace in traces if trace.finished_at is not None]
    ttft = [value for trace in completed if (value := trace.ttft_s) is not None]
    tpot = [value for trace in completed if (value := trace.tpot_s) is not None]
    itl = [value for trace in completed for value in trace.inter_token_latencies_s]
    e2e = [value for trace in completed if (value := trace.e2e_latency_s) is not None]
    output_tokens = sum(trace.output_tokens for trace in completed)

    return {
        "requests": len(traces),
        "completed": len(completed),
        "output_tokens": output_tokens,
        "elapsed_s": elapsed_s,
        "requests_per_s": len(completed) / elapsed_s,
        "output_tokens_per_s": output_tokens / elapsed_s,
        **_distribution("ttft", ttft),
        **_distribution("tpot", tpot),
        **_distribution("itl", itl),
        **_distribution("e2e", e2e),
    }


def _distribution(name: str, values: list[float]) -> dict[str, float]:
    return {
        f"{name}_avg_s": mean(values) if values else 0.0,
        f"{name}_p50_s": _percentile(values, 0.50),
        f"{name}_p99_s": _percentile(values, 0.99),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * quantile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction
