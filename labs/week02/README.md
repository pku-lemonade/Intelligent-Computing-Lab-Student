# Week 02 Profiler 资料包

| 文件 | 打开工具 | 用途 |
| --- | --- | --- |
| `week02-sample.nsys-rep` | Nsight Systems (`nsys-ui`) | 分析 CPU/CUDA API、stream、H2D/D2H、NVTX 和 kernel 时间线 |
| `week02-sample-stats.txt` | 文本编辑器 | 无法启动 GUI 时核对 Systems 汇总 |
| `week02-attention.ncu-rep` | Nsight Compute (`ncu-ui`) | 分析 FlashAttention kernel 的 launch、资源和采集 counter |
| `week02-attention-ncu-details.txt` | 文本编辑器 | 无法启动 Compute GUI 时核对报告中的 kernel 与 counter |
| `analysis-sheet.md` | 文本编辑器 | 填写 Systems 与 Compute 两部分观察 |

两个二进制报告是不同的独立实验：`.nsys-rep` 来自课程无模型 workload，
`.ncu-rep` 来自公开的 FlashAttention sequence-length-512 实验。不要把两个
报告中的 kernel 或时间区间强行对应，也不能把课程/第三方采集写成个人实测。

`week02-attention.ncu-rep` 的固定来源、commit、SHA-256 与 Apache-2.0 许可见
`THIRD_PARTY_NOTICES.md` 和 `LICENSE-AI-INFRA-BOOK-APACHE-2.0.txt`。
`sample_workload.py` 只用于课程 Systems 资料更新，学生不需要 GPU 或模型。
