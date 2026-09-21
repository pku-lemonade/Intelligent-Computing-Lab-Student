from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemOptimizationConfig:
    pinned_memory: bool = False
    transfer_stream: bool = False
    max_queue_size: int | None = None
    max_prompt_chars: int | None = None
    batch_wait_ms: float = 2.0
    max_batch_size: int = 8


def admission_decision(
    *,
    queue_depth: int,
    prompt_chars: int,
    max_queue_size: int | None,
    max_prompt_chars: int | None,
) -> dict:
    # TODO_BEGIN(W14_T02)
    raise NotImplementedError("implement admission control")
    # TODO_END(W14_T02)


def estimate_overlap_plan(config: SystemOptimizationConfig) -> dict:
    # TODO_BEGIN(W14_T03)
    raise NotImplementedError("implement overlap planning")
    # TODO_END(W14_T03)


def run_synthetic_experiment(config: SystemOptimizationConfig, *, repeats: int = 5) -> dict:
    started = time.perf_counter()
    for _ in range(repeats):
        time.sleep(0)
    elapsed = time.perf_counter() - started
    return {
        "config": {
            "pinned_memory": config.pinned_memory,
            "transfer_stream": config.transfer_stream,
            "batch_wait_ms": config.batch_wait_ms,
            "max_batch_size": config.max_batch_size,
            "max_queue_size": config.max_queue_size,
            "max_prompt_chars": config.max_prompt_chars,
        },
        "overlap_plan": estimate_overlap_plan(config),
        "admission_examples": [
            admission_decision(
                queue_depth=0,
                prompt_chars=32,
                max_queue_size=config.max_queue_size,
                max_prompt_chars=config.max_prompt_chars,
            ),
            admission_decision(
                queue_depth=config.max_queue_size or 1,
                prompt_chars=(config.max_prompt_chars or 32) + 1,
                max_queue_size=config.max_queue_size,
                max_prompt_chars=config.max_prompt_chars,
            ),
        ],
        "synthetic_elapsed_s": elapsed,
        "measurement_note": "Use nsys_server.sh and bench_server.py for real GPU serving numbers.",
    }


# TODO_BEGIN(W14_T01)
def measure_three_runs(fn, *, warmup: int = 5) -> dict:
    raise NotImplementedError("implement three-sample measurement")
# TODO_END(W14_T01)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14 system optimization experiment planner.")
    parser.add_argument("--pinned-memory", action="store_true")
    parser.add_argument("--transfer-stream", action="store_true")
    parser.add_argument("--max-queue-size", type=int, default=128)
    parser.add_argument("--max-prompt-chars", type=int, default=8192)
    parser.add_argument("--batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--max-batch-size", type=int, default=8)
    args = parser.parse_args()
    result = run_synthetic_experiment(SystemOptimizationConfig(**vars(args)))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
