# 智能计算系统编程与优化学生使用指南

学生公开仓库入口：[pku-lemonade/Intelligent-Computing-Lab-Student](https://github.com/pku-lemonade/Intelligent-Computing-Lab-Student)。课程平台会公告当前发布版本和变更说明。本仓库提供 starter、公开测试、练习数据和报告工作表，不包含参考答案、隐藏测试、平台服务代码、账号数据或模型权重。

## 课程实验主线

课程围绕一条可测量的 LLM serving 主线递进：先定义服务指标，再观察 GPU 时间线；随后实现 CUDA 算子，分析访存、矩阵乘、归约和数值稳定性；最后把 attention、KV cache、连续批处理、系统优化和 Prefill/Decode 解耦组装成可验收的推理系统。

| 周次 | 实验主题 | 主要学习任务 | 主要材料或交付 |
| --- | --- | --- | --- |
| Week 01 | Serving 指标 | 从请求事件计算 TTFT、ITL、TPOT、E2E 与吞吐 | trace、观察工作表 |
| Week 02 | Profiling | 在 Nsight Systems/Compute 中关联 API、stream、kernel 和计数器 | 课程 profiler 资料、分析工作表、阶段标记任务 |
| Week 03 | CUDA 基础 | 线程索引、边界保护、错误检查和基础 kernel | CUDA starter、公开测试 |
| Week 04 | 优化框架 | 用 APOD、Amdahl 和融合分析定位优化收益 | CUDA starter、性能记录 |
| Week 05 | 全局访存 | 比较连续、跨步和转置访问的地址与带宽 | CUDA starter、访存记录 |
| Week 06 | Matmul 与 Linear | 用 shared memory 分块矩阵乘并核对 shape | CUDA/Python starter、数值测试 |
| Week 07 | Reduction | 处理单位元、warp shuffle、同步和浮点顺序 | CUDA starter、归约记录 |
| Week 08 | Softmax 与 RMSNorm | 保持数值稳定，明确 CPU/CUDA dtype 契约 | CUDA/Python starter、误差记录 |
| Week 09 | Attention | 实现 causal attention 的 prefill/decode 路径 | Python/CUDA starter、集成测试 |
| Week 10 | Host/Device Transfer | 用 pinned memory、stream 和 event 证明异步传输 | Python starter、传输生命周期记录 |
| Week 11 | Engine 与 KV cache | 管理 request、sequence、生成状态和 KV 所有权 | Python starter、引擎事件测试 |
| Week 12 | Paged KV | 把逻辑 token 映射到物理 block，保持分配原子性 | Python/CUDA starter、分页映射记录 |
| Week 13 | Continuous Batching | 调度 waiting/running 请求并控制 prefill token 预算 | Python starter、在线调度报告 |
| Week 14 | System Optimization | 统一 warmup、统计、admission 和 copy/compute overlap | Python starter、三次原始样本 |
| Week 15 | Prefill/Decode Disaggregation | 明确 PD 状态、KV 传输和路由依赖 | Python starter、PD 报告 |
| Week 16 | Final 综合验收 | 汇总前 15 周实现，提交可复现、可诊断的系统证据 | 最终报告、展示材料、代码归档 |

课程内部有 45 个 TODO，其中 43 个是学生可见的代码任务；每个代码任务都有唯一 ID、模板文件和公开契约测试。课程网站指导书负责解释概念、手算和实验边界，平台题卡负责给出当前任务的下载文件、允许的上传类型和评测阶段。

## 仓库目录

| 路径 | 用途 |
| --- | --- |
| `templates/` | 带文件名、函数签名和 TODO 标记的学生 starter |
| `tests/` | 可公开运行的契约测试；不替代平台隐藏或集成评测 |
| `kernels/` | 课程使用的 CUDA 示例与算子接口 |
| `microllm/` | 逐周演进的指标、模型、引擎、KV 和调度代码 |
| `labs/` | 观察数据、profiler 资料、工作表和复采脚本 |
| `reports/` | Week 13–16 报告空白模板 |
| `RELEASE.md` | 当前公开包版本说明与边界 |

## 开始实验

1. 打开课程平台的课程公告，确认当前版本、周次指导书和任务题卡。
2. 克隆公开仓库并进入目录：

   ```bash
   git clone https://github.com/pku-lemonade/Intelligent-Computing-Lab-Student.git
   cd Intelligent-Computing-Lab-Student
   ```

3. 若需要本地运行 Python 公开测试，使用 Python 3.10–3.12 建立环境：

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install -e ".[dev]"
   python -m pytest -q
   ```

   Windows PowerShell 的激活命令是 `.venv\\Scripts\\Activate.ps1`。CUDA 测试会依据设备和课程 worker 的能力执行；没有对应设备时，以平台返回的阶段摘要为准，不要用 CPU 数字替代 GPU 结论。

## 每个任务怎么提交

1. 阅读本周指导书的“任务契约”，先写出输入、输出、边界条件和计时范围。
2. 从平台题卡下载完整模板，保留原文件名、扩展名、函数签名和 `TODO_BEGIN/END` 标记。
3. 只修改目标 TODO 区域；不要复制参考实现、删除检查代码或改动无关接口。
4. 在本地运行与环境匹配的公开测试，检查格式、边界输入和错误路径。
5. 将完整文件上传到“实验提交”页面，阅读 `public`、`integration`、`hidden` 和启用时的 `performance` 阶段摘要。
6. 按题卡要求提交观察工作表或阶段报告。性能数据必须注明设备、输入、dtype、预热次数、重复次数、同步边界和每次原始样本；课程采集、个人实测、论文数字和构造算例分开记录。

本课程的评测入口是文件上传，不要求学生部署 Django、启动课程服务器或提交 Git 仓库。模型权重、密码、私有课件、其他学生代码、真实学生数据和包含本机绝对路径的日志都不能上传。

## 读结果和排错

- 先区分“代码正确”“已接入调用路径”和“性能已测量”三个结论。独立 kernel 通过不等于 Engine 已使用它。
- 遇到失败时保留完整阶段输出，先定位 canonical 测试文件和输入 shape，再修改一个变量复测。
- CPU/fake backend 适合检查状态机、shape、所有权和数值参考；真实 CUDA、stream、带宽和 overlap 结论必须来自课程评测 worker 或课程发布的 profiler/benchmark 资料。
- 报告中出现 `not_run`、`not_integrated` 或 `blocked` 时，写清原因、证据和对后续结论的影响，不填写估计数字。

## 资料边界

课程网站提供每周讲稿、课程大纲、实验与课件主题对照以及官方资料索引。课堂课件下载和源代码不作为学生仓库的一部分；请以课程平台公告和本仓库的 `RELEASE.md` 为版本依据。
