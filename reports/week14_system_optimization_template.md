# Week 14 系统优化报告模板

本周报告回答一个问题：在固定 workload 和设备上，准入、batching、传输重叠与测量协议如何共同影响服务？所有性能数字均需先预热至少 5 次，再正式测量 3 次。

## 1. 实验环境

| 项目 | 填写 |
| --- | --- |
| GPU / driver / CUDA | |
| PyTorch | |
| 模型与 dtype | |
| prompt 长度与并发 | |
| 服务参数 | |
| 代码版本/文件摘要 | |

## 2. 可复现命令

```bash
# 在课程工程根目录执行
python -m microllm.benchmarks.grader week14
# 真实服务 benchmark 命令：
# python -m microllm.benchmarks.bench_server ... --json
```

## 3. 准入决策

| queue_depth | prompt_chars | max_queue | max_prompt | accepted | reason |
| ---: | ---: | ---: | ---: | :---: | --- |
| | | | | | |

解释队列满和 prompt 超长时的行为，并说明为什么拒绝原因必须可观察。

## 4. 三次性能测量

| 配置 | warmup | run 1 (s) | run 2 (s) | run 3 (s) | mean (s) | median (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline |  |  |  |  |  |  |
| optimized |  |  |  |  |  |  |

固定输入、计时边界和随机种子。不要只填平均值，平台会保存三次原始样本。

## 5. 结论与边界

区分 kernel 时间、传输时间、排队时间和端到端时间。指出优化在哪种输入下有效、在哪种输入下可能退化，并给出下一步验证。
