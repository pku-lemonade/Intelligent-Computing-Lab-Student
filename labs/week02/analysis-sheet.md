# Week 02 分析工作表

Systems 来源：`week02-sample.nsys-rep`（课程采集，不是个人 GPU 实测）。

Compute 来源：`week02-attention.ncu-rep`（`bojieli/ai-infra-book` Apache-2.0
公开报告，不是课程 Systems workload，也不是个人 GPU 实测）。

## Systems

| 项目 | 记录 |
| --- | --- |
| 报告版本 |  |
| NVTX 范围名称 |  |
| 选择区间起点/终点与单位 |  |
| CUDA API |  |
| GPU stream |  |
| kernel 名称与 GPU 时长 |  |
| 前后空档及可能依赖 |  |

## Compute：FlashAttention kernel

| 项目 | 记录 |
| --- | --- |
| kernel 名称 |  |
| grid/block |  |
| 执行时间 |  |
| registers/thread |  |
| shared memory/block |  |
| occupancy 的限制因素 |  |
| L2 (`lts__t_bytes.sum`) |  |
| 报告未包含的指标 |  |

先分析 `flash_fwd_splitkv_kernel`，再与
`flash_fwd_splitkv_combine_kernel` 比较 grid、执行时间和 L2 流量。报告只采集了
一组指定 counter；没有出现的 SpeedOfLight、warp-stall 或 DRAM 字段填写
“资料未包含该指标”，不要从设备属性推算实测值。

## 四句结论

观察：

候选解释：

证据：

证伪实验：
