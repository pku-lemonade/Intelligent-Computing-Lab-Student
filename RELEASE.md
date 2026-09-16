# Student Release 2026.09

这是《智能计算系统编程与优化》的学生公开材料包，覆盖课程 Week 01–16 的实验入口、代码 starter、公开测试、练习数据和报告工作表。

## 公开入口

[pku-lemonade/Intelligent-Computing-Lab-Student](https://github.com/pku-lemonade/Intelligent-Computing-Lab-Student)

课程平台会公告采用的版本号、截止时间和本次发布的变更。旧讲义中的版本链接与平台公告不一致时，以平台公告为准。

## 包含内容

- `templates/`：带文件名、函数签名和 TODO 标记的学生 starter；只修改题卡指定区域。
- `tests/`：可公开运行的契约测试，用于本地快速反馈；平台仍会运行隐藏和集成阶段。
- `kernels/`、`microllm/`：从 CUDA 算子到 serving engine、KV cache 和调度器的课程代码主线。
- `labs/`：无模型练习数据、profiler 资料和分析工作表；Week 02 同时提供可直接打开的 `.nsys-rep` 与 `.ncu-rep`。
- `reports/`：综合阶段报告模板，要求记录实验边界、原始样本和证据来源。

## 使用边界

本发布包用于阅读、修改、运行公开测试和向课程平台上传完整文件。它不包含参考答案、隐藏测试、评测服务、账号数据、私有课件或模型权重；学生不需要部署课程平台或启动本地课程服务器。

## 提交规则

1. 按课程平台公告确认当前版本和本周题卡。
2. 下载完整模板，保留文件名、扩展名、函数签名和 `TODO_BEGIN/END` 标记。
3. 只修改目标 TODO，运行与环境匹配的公开测试后上传完整文件。
4. 阅读平台返回的 `public`、`integration`、`hidden` 和（启用时）`performance` 阶段摘要。
5. 报告中的性能数字必须注明设备、输入、dtype、预热、重复、同步边界和原始样本，并区分课程资料、个人实测、论文数字和构造算例。

## 环境

Python 公开测试支持 Python 3.10–3.12；依赖和开发测试命令见顶层 `README.md`。CUDA、stream、带宽和 overlap 结论以课程评测 worker 或课程发布的 profiler/benchmark 资料为准，不能用 CPU 结果替代。
