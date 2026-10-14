# π0.5 TensorRT 量化部署：两页汇报

文档版本：`quantization_slides_20261008_r1`。本版本仅整理已完成的量化部署工作，未运行新的训练、推理或闭环测试。

[打开两页PPT](output/π0.5_TensorRT量化部署_2页.pptx)。文字、技术路线和两张表均为可编辑的原生幻灯片对象。

第1页展示原始 pi05_libero 权重 → 任务观测校准 → AWQ／GPTQ INT4 → ONNX／TensorRT → 完整推理与 LIBERO 闭环的技术路线，并整理 T0–T4、全 Linear 扩展及问题处理。第2页展示 A100 上 TensorRT float、旧 AWQ／GPTQ、全 Linear AWQ／GPTQ 的统一对比和验收边界。

数据采用[A100综合对比](../a100_comparison_20261008_r1/comparison.json)与[项目总结证据](../project_summary_20261008_r1/summary_evidence.json)，详细来源列在幻灯片备注。没有混入历史4090的PyTorch测量。全 Linear 表示451个图可达 Linear，非 Linear 参数与激活保留浮点。

此版本为文档冻结，不代表20Hz、实机或全部数值门槛已通过。[冻结清单](../../releases/quantization_slides_20261008_r1/manifest.json)。
