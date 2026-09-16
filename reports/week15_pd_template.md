# Week 15 Prefill/Decode Disaggregation 报告模板

## 1. 模型与假设

| 项目 | 填写 |
| --- | --- |
| layers | |
| kv heads | |
| head dim | |
| prompt tokens | |
| dtype bytes | |
| link bandwidth (B/s) | |
| prefill/decode worker 数 | |

## 2. 状态迁移

绘制并解释唯一合法顺序：`WAITING_PREFILL → PREFILLING → TRANSFERRING → DECODING → FINISHED`。记录每个事件时间，说明时间单调性和非法迁移处理。

## 3. KV 传输计算

```text
bytes = 2 × layers × kv_heads × head_dim × prompt_tokens × dtype_bytes
time  = bytes / link_bandwidth
```

填写一个具体算例，并核对程序输出。

## 4. 路由时间线与分析

列出 prefill worker、共享链路和 decode worker 的可用时间。改变带宽或 worker 数，观察瓶颈变化。明确哪些是确定性模拟结果，哪些需要真实多机实验才能验证。
