# TensorRT T3 实施版本

日期：2026-10-07。版本：`trt_t3_20261007_r1`。详细结果见 [T3实施记录](T3实施记录.md)，当前入口见根目录 `当前实施索引_20261007_r2.md`。

T3 的 AWQ/GPTQ 原生引擎、实际执行覆盖与独立进程重载验证已通过。AWQ 离线数值门槛通过，作为主候选；GPTQ 因夹爪方向分歧超过门槛保留为对照。T1 原始失败项保持失败，用户的临时允许仅用于继续实施本阶段。闭环、目标 4070/8GB 与实机适配尚未验收。

服务器工作根目录为 `/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference`；容器内对应 `/work`。模型使用原始 `pi05_libero` 权重衍生的量化产物，未使用 ECO65+ZhiXing 微调权重；校准和验证观测来自 Panda+PandaGripper。

## 交付文件

| 文件 | 用途 |
|---|---|
| `protocol.json` | 授权范围、原门槛、运行契约与阶段状态 |
| `build_engine.py` | 强类型 prefix INT4 引擎构建、cache、详细层报告 |
| `make_bundle.py` | 三个冻结引擎的路径、SHA256 与环境绑定 |
| `runtime.py` | 单策略三引擎、串行共享工作空间、只读 prefix KV |
| `common.py` | 原始预处理和动作反归一化；不实例化源模型 |
| `infer.py` | 独立进程冷加载、完整 10 步推理、输出和显存记录 |
| `inspect_coverage.py` | 源 INT4 节点到优化后 DQ/GEMM 的追踪 |
| `confirm_execution.py` | 把源层映射与实际 IProfiler 回调关联 |
| `compare.py` | 同后端误差、累计误差、同引擎独立重载与原 T1 重放 |
| `capture_environment.py` | 与 T0–T2 比较包版本、原代码、补丁与 tokenizer 哈希 |
| `execute.py` | 本阶段执行时保存每次独立源码快照、日志和真实退出码 |
| `freeze_release.py` | 新清单生成及完整服务器／部分本地镜像校验 |

输出位于 `runs/trt_t3_20261007_r1`。主 bundle 为 `bundles/awq.json`，对照为 `bundles/gptq.json`，float 为 `bundles/float.json`。vision/action 引擎复用已冻结 T1 文件，prefix 引擎位于 `awq_build_r1/prefix.engine` 与 `gptq_build_r1/prefix.engine`，没有额外复制源权重。

当前有效结果为 `awq_infer_r2`、`awq_reload_r1`、`gptq_infer_r2`、`gptq_reload_r1`、`float_infer_r1`、`awq_comparison_r1.json`、`gptq_comparison_r1.json`、`awq_coverage_r2.json`、`gptq_coverage_r3.json`、两个 `*_execution_r1.json` 与 `environment_r1`。前期失败、旧诊断尝试及源码快照全部保留；不能用构建时的 `reload_accuracy: pending` 代替后续独立验证记录。

## 同服务器复现

运行镜像必须为 `pi05_trt:20261007-r1`，完整 ID 为 `sha256:cf96556ab8b55eb3f80770d2a684d6961c0128c8100660a8cc4c7e61ecc60369`。本阶段专属容器 `pi05-trt-t3-20261007-r1` 已在空闲时停止，未动其他采集、训练或控制进程。

以下命令重载相同 AWQ 引擎，写入一个**新的、事先不存在的**复现目录。不要把输出写回冻结版本，也不要在冻结后继续使用硬编码本阶段输出目录的 `execute.py` 创建新尝试。

```bash
docker start pi05-trt-t3-20261007-r1
docker exec pi05-trt-t3-20261007-r1 /.venv/bin/python \
  /work/versions/trt_t3_20261007_r1/infer.py \
  --out /work/runs/trt_t3_replay_NEW_TAG/awq \
  --bundle /work/runs/trt_t3_20261007_r1/bundles/awq.json \
  --pilot /work/runs/trt_t0_t2_20261007_r1/calibration_r1/pilot.json
```

把 `NEW_TAG` 换为新版本标识。GPTQ 只需换 bundle；质量判定仍使用原门槛。默认不启用 profiler；`--profile-warmup --limit 1` 仅用于诊断层执行，不能用作正式速度测试。

当前引擎的硬件兼容级别为 `NONE`，绑定开发端 4090 与当前 TensorRT 运行环境。到目标 4070 时应建立新版本，在目标设备重建并验收；不能直接把当前引擎的文件大小或 4090 显存结果当作目标部署通过。完整执行文件在服务器，本地仅镜像可审阅的小文件。
