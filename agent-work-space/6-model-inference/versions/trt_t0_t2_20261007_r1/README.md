# TensorRT T0–T2 实施版本

用户于 2026-10-07 授权实施 T0、T1、T2，解除这三个阶段的“暂不实施”限制。原 r4 方案和旧冻结文件保持不变；本版本建立独立源码、结果、运行镜像与校验清单。

**文件冻结不等于部署验收。** T0 通过；T1 三个浮点子图已经构建，但 128 条观测的扩展裁剪门槛未通过。T2 三种 INT4 导出及逐层字节核验通过，其中 AWQ 完整策略离线误差通过，GPTQ/RTN 未通过。量化引擎构建、实际 4070 的 8GB 显存与速度、闭环成功率均未验收。

先读 [T0–T2实施记录](T0-T2实施记录.md)，实际数值、诊断、复现命令和后续准入条件均在其中。状态字段见 [protocol.json](protocol.json)。当前结果目录为 `../../runs/trt_t0_t2_20261007_r1/`，完整服务器文件清单见 [冻结 manifest](../../releases/trt_t0_t2_20261007_r1/manifest.json)。

使用官方 pi05_libero 的 BF16 数值对齐权重，未使用 W8 权重再量化。首轮量化范围是语言 Transformer 的 126 个 attention/MLP Linear；实际 prefix KV 子图中有 121 个有效层。视觉、动作专家、嵌入与归一化保留浮点。INT4 为 signed symmetric、zero point=0、group=128，scale 与 AWQ 输入乘子为 BF16。

源码按用途分为：

- 环境与浮点：`common.py`、`subgraphs.py`、`runtime.py`、`onnx_export.py`、`t0.py`、`t1.py`、`validate_float.py`。
- 真实校准：`teacher.py`、`collect.py`、`collect_run.py`、`calibrate.py`、`audit_calibration.py`。
- 量化与核验：`codec.py`、`gptq.py`、`t2.py`、`validate_quant.py`、`codec_probe.py`、`verify_exports.py`。
- 留痕与冻结：`execute.py`、`capture_environment.py`、`freeze_release.py`。`salvage.py` 和 `sdpa_probe.py` 仅解释本次诊断与失败恢复。

`execute.py` 在运行前保存独立源码快照，输出名称必须全新。离线比较脚本的退出码 0 表示比较已完成，是否通过必须读取 `offline_accuracy_passed`，不能只看退出码。冻结后的受管文件不可覆盖；修复或调整需建立新版本。

本地仅同步源码、文档、日志和小型 JSON，完整观测、激活、权重、ONNX 外部数据及引擎保存在服务器。实际运行镜像为 `pi05_trt:20261007-r1`，完整指纹和包清单见实施记录；`Dockerfile` 是从固定基础镜像补齐依赖的配方，最终已安装镜像指纹另行保存。
