# π0.5 TensorRT INT4量化部署方案

唯一方案：[π0.5 TensorRT INT4量化部署方案](versions/w4_plan_20261007_r4/π0.5实机部署4bit量化压缩方案.md)，版本 `w4_plan_20261007_r4`。

部署主线：ModelOpt AWQ校准 → 显式INT4 ONNX → TensorRT engines。GPTQ作为同条件对照；首轮语言Transformer INT4、视觉及动作专家保留浮点。目标设备为RTX4070，推理总显存上限8GB。

**当前仅更新方案，暂不实施。** 没有安装依赖、量化、导出、构建engine、模型性能测试、仿真、训练或机器人部署；全部新方案验收项未执行。

| 路径 | 用途 |
|---|---|
| [versions/w4_plan_20261007_r4](versions/w4_plan_20261007_r4/README.md) | 唯一现行方案、协议和冻结清单 |
| weights/ | 原有模型权重及元数据，保持不变 |
| [plan_update_20261007_r1](plan_update_20261007_r1/变更记录.md) | TensorRT路线变更、前版退役和原清理结果 |

工作目录不恢复旧脚本、日志或结果。前版方案和原清理记录在项目外备份中原样保留。本地是轻量资产镜像，完整权重保留在服务器；采集、训练和控制器不作改动。
