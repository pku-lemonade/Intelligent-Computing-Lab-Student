from __future__ import annotations


def shared_prefix_len(a: list[int], b: list[int]) -> int:
    total = 0
    for left, right in zip(a, b):
        if left != right:
            break
        total += 1
    return total


def cache_aware_order(prompts: list[list[int]]) -> list[int]:
    """Greedy request ordering that keeps similar prefixes adjacent."""

    if not prompts:
        return []
    remaining = set(range(len(prompts)))
    order = [min(remaining)]
    remaining.remove(order[0])
    while remaining:
        previous = prompts[order[-1]]
        next_index = max(
            remaining,
            key=lambda index: (shared_prefix_len(previous, prompts[index]), -index),
        )
        order.append(next_index)
        remaining.remove(next_index)
    return order


def score_order(prompts: list[list[int]], order: list[int]) -> int:
    score = 0
    for left, right in zip(order, order[1:]):
        score += shared_prefix_len(prompts[left], prompts[right])
    return score


def estimate_pd_disaggregation(
    requests: list[dict[str, int]],
    *,
    prefill_tokens_per_s: float = 4000.0,
    decode_tokens_per_s: float = 250.0,
) -> dict:
    """Estimate a teaching approximation of prefill/decode disaggregation."""
    if prefill_tokens_per_s <= 0 or decode_tokens_per_s <= 0:
        raise ValueError("throughput values must be positive")
    total_prefill = sum(max(0, request["prompt_len"]) for request in requests)
    total_decode = sum(max(0, request["decode_len"]) for request in requests)
    coupled_s = total_prefill / prefill_tokens_per_s + total_decode / decode_tokens_per_s
    disaggregated_s = max(total_prefill / prefill_tokens_per_s, total_decode / decode_tokens_per_s)
    return {
        "total_prefill_tokens": total_prefill,
        "total_decode_tokens": total_decode,
        "coupled_time_s": coupled_s,
        "disaggregated_time_s": disaggregated_s,
        "estimated_speedup": coupled_s / disaggregated_s if disaggregated_s else 1.0,
    }


def remaining_work_order(requests: list[dict[str, int]]) -> list[int]:
    """Prioritize short remaining decode work while keeping older requests stable."""
    return sorted(
        range(len(requests)),
        key=lambda index: (
            max(0, requests[index]["decode_len"] - requests[index].get("generated", 0)),
            requests[index].get("age", index),
            index,
        ),
    )


def score_decode_completion_cost(requests: list[dict[str, int]], order: list[int]) -> int:
    elapsed = 0
    total_completion = 0
    for index in order:
        elapsed += max(0, requests[index]["decode_len"] - requests[index].get("generated", 0))
        total_completion += elapsed
    return total_completion


def paper_to_system_map(mechanism: str) -> dict:
    mappings = {
        "cache-aware serving": {
            "paper_idea": "schedule requests with shared prefixes close together to increase KV reuse",
            "engine_modules": ["engine/policies.py", "engine/block_manager.py", "engine/engine.py"],
            "data_structures": ["prompt token ids", "block hash table", "scheduler waiting queue"],
            "metrics": ["prefix_cached_blocks", "TTFT", "requests/s", "KV fragmentation"],
            "minimal_experiment": "compare baseline order and cache_aware_order shared-prefix score",
        },
        "prefill-decode disaggregation": {
            "paper_idea": "separate prefill-heavy and decode-heavy work so each can use a tailored batch policy",
            "engine_modules": ["engine/scheduler.py", "server/batching.py", "engine/engine.py"],
            "data_structures": ["waiting queue", "running sequences", "KV cache handles"],
            "metrics": ["TTFT", "TPOT", "queue depth", "batch size"],
            "minimal_experiment": "toggle chunked prefill and compare latency distribution under mixed prompt lengths",
        },
        "remaining-work scheduling": {
            "paper_idea": "prioritize estimated remaining decode work while using request age as a tie-breaker",
            "engine_modules": ["engine/scheduler.py", "engine/request.py"],
            "data_structures": ["per-sequence generated token count", "decode batch"],
            "metrics": ["tail latency", "fairness", "output_tokens/s"],
            "minimal_experiment": "compare FIFO decode with age/token-count priority on synthetic requests",
        },
    }
    key = mechanism.strip().lower()
    if key not in mappings:
        raise ValueError(f"unknown mechanism {mechanism!r}; choose one of: {', '.join(sorted(mappings))}")
    return {"mechanism": key, **mappings[key]}
