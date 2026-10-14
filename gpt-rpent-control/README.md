# RPent + 当前 OpenPI 的 Panda / LIBERO 验证

工作目录：`/mnt/data/vla-policy/openpi/gpt-rpent-control`。新增版本 `versions/panda_libero_20261007_r1`；旧控制、数据采集、量化实验和 RPent 源码只读复用。

## 实施方案

1. 在独立运行环境加载 RPent commit `068cd64f4f175179a13c67d429a4d00ca56fb196`，使用其原生 ApiAgentLoop、LiberoToolkit、LiberoPrimitives、Env/VLA 客户端、RPC 与进程生命周期。
2. 环境适配层直接使用当前项目的 OffScreenRenderEnv，显式固定 Panda + PandaGripper。官方 LIBERO Spatial 的初态按 seed % 初态数选取，初始化后 10 步静置；20 Hz、OSC_POSE 和 Panda 的 legacy 附加 PD 保持现有基线。
3. VLA RPC 桥接已有冻结原精度 pi0.5 PyTorch 模型服务，双视角 180 度旋转、224 尺寸、8 维 state，原 10 轮去噪与 10×7 动作输出；只执行前 5 步后重新观测。保存原始与实际限幅动作。
4. 上层采用用户选择的现有 DeepSeek 文本规划，通过原生 RPent 工具选择/调用 VLA，再读反馈并 finish。文本模式不宣称图像推理能力。首轮仅暴露可实际执行且有预算的工具，不安装或伪造 SAM3。
5. 先运行真实组件与闭环试点，再运行 Spatial 10 个任务各 1 个初态。成功以原生 `_check_success()` 为准；LLM finish 与退出码独立记录。保存模型请求、工具调用、逐步动作、双视角视频、初态与源文件校验、进程退出。

## 验证口径

接口、真实策略闭环、原生任务成功和正式 benchmark 分开。10×1 为工程验证，不替代正式多初态统计验收；失败和异常完整保留。模型/控制器参数不根据结果临时修改。每次运行使用新目录，冻结后建立新版本才能改源。

凭据仅运行时读取已有凭据文件，不进入源码、参数、日志或发布清单。只在规划进程提供凭据，仿真/模型子进程不继承。

结果与复现命令完成后记入当前实施索引和版本验证记录。
