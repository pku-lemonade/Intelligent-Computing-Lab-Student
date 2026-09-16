# Week 16 综合实验报告模板

本文档是学生最终报告模板。请只填写自己的实验环境、命令、指标、截图和分析，不要复制 TA validation log 中的一次性实测数字。

## 1. 系统概览

| 项目 | 填写 |
| --- | --- |
| backend |  |
| model |  |
| dtype |  |
| kernel_impl |  |
| GPU / CUDA / PyTorch |  |
| supported endpoints |  |

简要说明你的系统如何串联 prefill、decode、KV cache、paged KV、scheduler、HTTP serving 和 sampling。

## 2. 正确性证据

填写你实际运行的测试命令和结果。

| 检查项 | 命令 | 结果摘要 |
| --- | --- | --- |
| full pytest | `pytest -q -rs` |  |
| stage grader | `python -m microllm.benchmarks.grader <week>` |  |
| CUDA smoke | `python -m microllm.benchmarks.grader cuda_smoke` |  |
| Qwen3/HF alignment | `python validation/compare_qwen3.py ...` |  |

回答：

1. 哪些测试是 correctness oracle？
1. 哪些测试只是 smoke test？
1. 如果 CUDA smoke 没有运行，原因是什么？这对你的结论有什么影响？

## 3. 性能证据

填写 profiler 和 benchmark 结果。

| 工具 | 产物路径或截图 | 主要发现 |
| --- | --- | --- |
| torch profiler |  |  |
| nsys |  |  |
| ncu |  |  |
| HTTP benchmark |  |  |

至少回答：

1. 主要耗时在 Python、GPU kernel、memcpy 还是同步等待？
1. prefill 和 decode 的瓶颈是否相同？
1. 你会优先优化哪个模块？为什么？

## 4. Week 14 系统优化对比

| 配置 | requests/s | output tokens/s | p50 | p90 | p99 | 流式 TPOT 或 unavailable | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline |  |  |  |  |  |  |  |
| optimized |  |  |  |  |  |  |  |

回答：

1. 你的 workload 是短 prompt 还是长 prompt？并发是多少？
1. pinned memory、transfer stream、chunked prefill、cache-aware scheduling 中哪些可能影响结果？
1. 如果优化收益不明显，给出至少两个可能原因。

## 5. 容量与调度规划

| 项目 | 数值 |
| --- | ---: |
| GPU memory |  |
| weight memory |  |
| KV budget |  |
| KV blocks |  |
| token slots |  |
| target concurrency |  |
| target sequence length |  |

判断：当前目标是否需要更多 KV 容量、tensor parallelism 或 pipeline parallelism？说明依据。

## 6. 前沿机制映射

选择一个机制，例如 cache-aware serving、prefill/decode disaggregation 或 remaining-work scheduling。

| 项目 | 填写 |
| --- | --- |
| 机制名称 |  |
| 论文/资料来源 |  |
| 影响的 engine modules |  |
| 需要新增或修改的数据结构 |  |
| 目标指标 |  |
| demo 或实验结果 |  |

说明该机制在本课程工程中是完整实现、教学近似，还是 paper-to-system 设计草案。

## 7. 故障诊断复盘

选择一次你实际遇到的问题。

| 项目 | 填写 |
| --- | --- |
| 现象 |  |
| 初始假设 |  |
| 定位证据 |  |
| 修复或规避方式 |  |
| 验证命令 |  |
| 剩余风险 |  |

## 8. 最终总结

用 300-500 字总结：

1. 你实现或理解最深入的模块。
1. 你认为当前工程距离工业级 vLLM/SGLang 还差什么。
1. 如果继续迭代两周，你会优先做什么。
