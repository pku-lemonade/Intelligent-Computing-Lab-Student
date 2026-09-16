from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageSpec:
    key: str
    week: int
    title: str
    topic: str
    code_status: str
    required_outputs: tuple[str, ...]
    test_hints: tuple[str, ...]


_STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        key="week01",
        week=1,
        title="Serving 指标与基线",
        topic="prefill/decode, token events, TTFT/TPOT/E2E, serving throughput",
        code_status="implemented",
        required_outputs=("参考服务流式记录", "请求事件时间线", "固定窗口 baseline 指标"),
        test_hints=("tests/public/week01/test_request_metrics.py", "tests/integration/week01/test_metrics_pipeline.py"),
    ),
    StageSpec(
        key="week02",
        week=2,
        title="Profiling 与 GPU 观测",
        topic="roofline, nsys, ncu, PyTorch profiler, serving metrics",
        code_status="implemented",
        required_outputs=("nsys timeline", "ncu kernel report", "性能分析报告"),
        test_hints=("tests/public/week02/test_runtime_ranges.py", "scripts/profile/"),
    ),
    StageSpec(
        key="week03",
        week=3,
        title="CUDA 基础",
        topic="thread hierarchy, host/device, vector add, SAXPY, correctness and timing",
        code_status="implemented",
        required_outputs=("Vector Add", "SAXPY", "边界正确性与设备记录"),
        test_hints=("tests/public/week03/test_cuda_basics.py", "tests/integration/week03/test_cuda_pipeline.py"),
    ),
    StageSpec(
        key="week04",
        week=4,
        title="优化框架",
        topic="grid-stride loop, elementwise fusion, controlled performance comparison",
        code_status="implemented",
        required_outputs=("grid-stride SAXPY", "fused elementwise", "优化前后执行证据"),
        test_hints=("tests/public/week04/test_optimization_framework.py", "tests/integration/week04/test_optimization_pipeline.py"),
    ),
    StageSpec(
        key="week05",
        week=5,
        title="全局访存",
        topic="coalescing, strided access, row-major layout, matrix transpose",
        code_status="implemented",
        required_outputs=("coalesced/strided copy", "naive transpose", "地址与访存分析"),
        test_hints=("tests/public/week05/test_memory_access.py", "tests/integration/week05/test_memory_pipeline.py"),
    ),
    StageSpec(
        key="week06",
        week=6,
        title="Matmul 与 Linear",
        topic="naive/tiled matmul, shared memory, tail tiles, linear dispatch",
        code_status="implemented",
        required_outputs=("naive/tiled matmul", "CourseLinear 主路径接入", "正确性与性能对照"),
        test_hints=("tests/public/week06/test_matmul.py", "tests/integration/week06/test_linear_pipeline.py"),
    ),
    StageSpec(
        key="week07",
        week=7,
        title="Reduction",
        topic="warp shuffle, shared-memory reduction, row-wise sum and maximum",
        code_status="implemented",
        required_outputs=("warp sum", "row sum/max", "同步与边界分析"),
        test_hints=("tests/public/week07/test_reduction.py", "tests/integration/week07/test_reduction_pipeline.py"),
    ),
    StageSpec(
        key="week08",
        week=8,
        title="Softmax 与 RMSNorm",
        topic="stable softmax, RMSNorm, mixed-precision accumulation, model dispatch",
        code_status="implemented",
        required_outputs=("stable Softmax", "RMSNorm", "Qwen3 RMSNorm 接入与误差分析"),
        test_hints=("tests/public/week08/test_numerical_kernels.py", "tests/integration/week08/test_numerical_pipeline.py"),
    ),
    StageSpec(
        key="week09",
        week=9,
        title="Attention",
        topic="causal prefill, dense decode, online softmax, CUDA dispatch",
        code_status="implemented",
        required_outputs=("prefill/decode attention", "reference 对齐", "主路径派发"),
        test_hints=("tests/public/week09/test_week09_attention.py", "tests/integration/week09/test_attention_pipeline.py"),
    ),
    StageSpec(
        key="week10",
        week=10,
        title="Host/Device Transfer",
        topic="pinned memory, CUDA streams/events, transfer dependencies, buffer reuse",
        code_status="implemented",
        required_outputs=("安全的 non-blocking H2D", "transfer pipeline", "真实时间线"),
        test_hints=("tests/public/week10/test_transfer.py", "tests/integration/week10/test_transfer_pipeline.py"),
    ),
    StageSpec(
        key="week11",
        week=11,
        title="推理 Engine 与连续 KV",
        topic="request lifecycle, streaming generation, sampling, continuous KV append/release",
        code_status="implemented",
        required_outputs=("请求状态机", "连续 KV 生命周期", "单请求流式生成闭环"),
        test_hints=("tests/public/week11/test_request_kv_engine.py", "tests/integration/week11/test_generation_pipeline.py"),
    ),
    StageSpec(
        key="week12",
        week=12,
        title="Paged KV",
        topic="block manager, block table, slot mapping, physical KV pool, paged decode",
        code_status="implemented",
        required_outputs=("分页分配与寻址", "KV 读写", "paged attention 接入"),
        test_hints=("tests/public/week12/test_paged_kv.py", "tests/integration/week12/test_paged_decode_pipeline.py"),
    ),
    StageSpec(
        key="week13",
        week=13,
        title="Continuous Batching",
        topic="online admission, iteration scheduling, token budget, chunked prefill",
        code_status="implemented",
        required_outputs=("waiting/running 调度", "chunked prefill", "在线请求迭代闭环"),
        test_hints=("tests/public/week13/test_continuous_batching.py", "tests/integration/week13/test_online_engine.py"),
    ),
    StageSpec(
        key="week14",
        week=14,
        title="System Optimization",
        topic="admission control, overlap planning, reproducible three-run measurement",
        code_status="implemented",
        required_outputs=("准入决策", "重叠计划", "优化前后三次原始样本"),
        test_hints=("tests/public/week14/test_system_optimization.py", "tests/integration/week14/test_system_optimization_pipeline.py"),
    ),
    StageSpec(
        key="week15",
        week=15,
        title="Prefill/Decode Disaggregation",
        topic="P/D state, KV transfer, resource routing, latency model",
        code_status="implemented",
        required_outputs=("P/D 状态机", "KV 交接计划", "资源时间线与 paper-to-system 分析"),
        test_hints=("tests/public/week15/test_pd_week15.py", "tests/integration/week15/test_pd_week15_pipeline.py"),
    ),
    StageSpec(
        key="week16",
        week=16,
        title="综合验收与展示",
        topic="operators, cache, scheduler, serving, performance evidence, diagnosis",
        code_status="implemented",
        required_outputs=("最终展示", "性能证据", "故障诊断复盘"),
        test_hints=("tests", "docs/reports/week16_final_report_template.md"),
    ),
)

STAGES: dict[str, StageSpec] = {stage.key: stage for stage in _STAGES}
LATEST_STAGE = _STAGES[-1].key


def normalize_stage(stage: str | int | None) -> str:
    if stage is None:
        return LATEST_STAGE
    if isinstance(stage, int):
        key = f"week{stage:02d}"
    else:
        text = stage.strip().lower()
        key = f"week{int(text):02d}" if text.isdigit() else text
    if key not in STAGES:
        valid = ", ".join(STAGES)
        raise ValueError(f"unsupported course stage {stage!r}; valid stages: {valid}")
    return key


def stage_spec(stage: str | int | None = None) -> StageSpec:
    return STAGES[normalize_stage(stage)]


def stage_overview(stage: str | int | None = None) -> dict:
    spec = stage_spec(stage)
    return {
        "key": spec.key,
        "week": spec.week,
        "title": spec.title,
        "topic": spec.topic,
        "code_status": spec.code_status,
        "required_outputs": list(spec.required_outputs),
        "test_hints": list(spec.test_hints),
    }


def all_stage_overviews() -> list[dict]:
    return [stage_overview(stage.key) for stage in _STAGES]
