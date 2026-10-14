# ECO65＋ZhiXing 原生模型闭环验证 r2

本版承接 Panda 基线，使用真实 DeepSeek 文本规划、RPent ApiAgentLoop/LiberoToolkit/LiberoPrimitives 和冻结原精度 pi0.5；不加载 LoRA、不使用脚本教师，不修改模型、控制器、资产或 LIBERO 成功谓词。仅扩展项目外部薄 Env 适配器，不改 RPent 注册、CLI 或 Dashboard。

预注册本次验证：Spatial task0–9，各 seed=0、官方初态 index0，只运行一次正式批次。预算220动作步，10步静置；原10轮去噪、10×7预测、执行前5步重新观察；双RGB预处理与Panda相同。上层模型与工具提示保持原基线，只有机器人名称替换。两路CPU环境共享一个串行原模型服务。

ECO65使用现有按夹爪分派的 ZhiXing 初始关节角和既有安装/相机。保留 Panda 基线的 `legacy` 额外PD配置与现有 OSC 参数，ZhiXing 默认速度0.01及 +1闭合/-1张开语义。专家采集中的 `eco65_no_extra_pd` 是另一个配置，本轮不切换；本轮结论限定为 legacy 下的跨机械臂/夹爪零样本闭环，不解释为该候选配置的结果。

官方初态的机械臂维数不同，不能整段复制。参照已有初始位姿对齐方法，在参考Panda环境中读取每个任务官方初态，按关节名复制所有非机器人关节 qpos/qvel；ECO65机械臂和夹爪保留其现有 reset。保存原始官方状态、复制前后核对、ECO状态和10步静置后的图像。物体在静置期间的物理变化不修正。

模型state8沿用2-task-collection：原观测TCP xyz、轴角、按名字选择的 `Left_1_Joint/Right_1_Joint` 实测角度。保留6联动关节完整数据；不伪造Panda两指位移或隐藏新夹爪图像。7维动作直接交给原 OSC/夹爪实现，只保留基线既有限幅。

每集保存初始＋逐动作双视角MP4（512×256、20FPS）、RPent主视角MP4（256×256、20FPS）、最新观测/预测归档、DeepSeek反馈、逐步实际动作/qpos/qvel/ctrl及只读物体/夹爪/接触诊断。私有物体坐标只进诊断文件，不反馈给模型，不用于动作。失败完整保留；任务成功与轨迹审计分开判定。

先执行真实环境预检和10步实际模型链，再运行正式10任务；开发/预检不计入正式成功率。视频逐帧解码、尺寸、帧数与FPS检查；源码、旧冻结清单前后哈希核对。最终仅报告本批10×1仿真结果，不代表多初态统计、完整RPent工具集或实机适配。

复现使用新的run-id：

```bash
cd /mnt/data/vla-policy/openpi/gpt-rpent-control
python3 versions/eco65_zhixing_20261007_r2/workflow.py --run-id 新批次名 --workers 2
```

环境/模型启动复用原镜像与runtime。凭据仍只从stdin交给规划进程，证据不得包含凭据值。旧Panda实现、视频及冻结文件均保留。

开发r1预检在初始化证据JSON写入时因NumPy int32失败，未完成预检，不计成功；r2只转换关节索引整数类型，r1源码和失败证据保留。
