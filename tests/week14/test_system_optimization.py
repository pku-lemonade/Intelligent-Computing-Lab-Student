from microllm.benchmarks.system_optimization import (
    SystemOptimizationConfig,
    admission_decision,
    estimate_overlap_plan,
    measure_three_runs,
)


def test_w14_t01_runs_three_samples_after_warmup():
    calls = []
    result = measure_three_runs(lambda: calls.append(1), warmup=2)
    assert len(calls) == 5
    assert len(result["samples_s"]) == 3
    assert result["mean_s"] >= 0
    assert result["median_s"] >= 0


def test_w14_t02_admission_rejects_full_or_long_requests():
    assert admission_decision(queue_depth=2, prompt_chars=4, max_queue_size=2, max_prompt_chars=10)["reason"] == "queue_full"
    assert admission_decision(queue_depth=0, prompt_chars=11, max_queue_size=2, max_prompt_chars=10)["reason"] == "prompt_too_long"


def test_w14_t03_overlap_requires_pinned_and_transfer_stream():
    assert not estimate_overlap_plan(SystemOptimizationConfig()).get("overlap_enabled")
    assert estimate_overlap_plan(SystemOptimizationConfig(pinned_memory=True, transfer_stream=True))["overlap_enabled"]
