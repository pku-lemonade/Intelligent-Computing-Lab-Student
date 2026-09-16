# Week 13 Continuous Batching 实验记录

## 1. 实验目标

记录请求在线到达时的 prefill/decode 迭代调度、token budget 和资源释放。请用事件顺序说明调度器做出的每个决定。

## 2. 负载与约束

| 参数 | 值 |
| --- | --- |
| max_num_seqs | |
| max_num_batched_tokens | |
| chunked prefill | |
| 请求 A prompt/token 上限 | |
| 请求 B 到达时刻 | |

## 3. 迭代时间线

| step | waiting | prefill batch (tokens) | decode batch | 新到达请求 | 释放的资源 |
| ---: | --- | --- | --- | --- | --- |
| | | | | | |

至少包含 A prefill、A decode、B 到达、B prefill、A/B 共同 decode 的场景。

## 4. 观察与结论

说明 FIFO、token budget 和 chunking 如何影响公平性与吞吐；记录集成测试命令和结果。
