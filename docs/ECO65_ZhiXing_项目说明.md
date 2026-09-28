# OpenPI π0.5 复现与 ECO65 / ZhiXing 适配项目说明

整理日期：2026-09-19  
阅读对象：`/Users/yuyu/Documents/openpi` 本地代码副本  
项目定位：基于 OpenPI 的 π0.5 复现，以及 ECO65 六轴机械臂、ZhiXing CRT-CTAG2F90 夹爪的模型、仿真环境和策略接口适配。

## 1. 文档范围与结论依据

根据项目维护者说明，π0.5 的复现及主要运行工作在远程服务器上进行，当前目录是项目代码副本。本说明整理副本中能够确认的工程实现、数据接口、运行流程和后续核验事项。

本文采用以下证据区分：

| 标记 | 含义 |
| --- | --- |
| 用户说明 | 项目维护者提供的背景，例如已在远程服务器复现 π0.5 |
| 代码确认 | 本地源代码、配置、模型文件直接支持的事实 |
| 静态推断 | 沿调用链推导出的行为，未通过本轮运行实验验证 |
| 待远程核验 | 需要远程环境、实际 checkpoint、日志或硬件才能确认 |

本次整理未启动模型训练、MuJoCo 仿真、Docker 容器或真实硬件，未修改现有业务代码。项目中保留了多个实验版本，因此目录名、注释和旧 README 不一定代表实际执行行为。本文优先依据 Python 调用链与实际配置。

项目工作可以概括为：复用 OpenPI 的模型、训练、归一化和 WebSocket 推理框架，将 ECO65 的六轴模型与 ZhiXing 的联动夹爪接入 robosuite，再通过 LIBERO 的任务环境和观测接口连接 π0.5。副本还包含 RethinkGripper 对照实验、LIBERO 数据转换、自定义微调配置，以及 ECO 数据的离线远程推理代码。

## 2. 已有工作与工程贡献

| 工作方向 | 副本中的实现 | 当前能够确认的范围 |
| --- | --- | --- |
| π0.5 复现 | 保留 OpenPI 的 JAX / PyTorch 模型、训练脚本、策略服务 | 复现背景来自用户说明；实际运行后端、权重与指标需远程记录佐证 |
| ECO65 模型接入 | 六轴 MJCF、机器人 Python 类、机器人注册 | 模型资产、关节、执行器及框架入口已定义 |
| LIBERO 机器人接入 | `MountedECO65`、桌面偏移、独立初始姿态 | 桌面任务可沿代码路径选择 ECO65 模型 |
| ZhiXing 夹爪接入 | MJCF、联动约束、碰撞体、单维动作包装、夹爪注册 | 夹爪机构与控制接口已实现 |
| 观测与动作适配 | 双相机、轴角姿态、夹爪状态选取、7 维动作 | 已建立 LIBERO 客户端与策略服务的协议对应关系 |
| 控制稳定性调整 | 在 `SingleArm.control()` 中增加关节 PD 项 | 代码已存在，效果需仿真验证 |
| 数据处理 | TFRecord 直接解析、LeRobot 转换、统计量计算入口 | 包含数据转换脚本和本地数据元信息 |
| 本地数据微调配置 | `pi05_libero_custom` | 定义了训练参数，但不能仅凭配置判定训练已经完成 |
| 推理实验与记录 | 多个 ECO65 示例、双视角视频、离线预测保存 | 实验入口和输出逻辑存在 |
| 真实夹爪基础接口 | ROS `joint_states` 到 Modbus 寄存器控制 | 独立脚本存在，未发现与 OpenPI 执行链路集成 |

这些适配主要发生在机器人模型、环境、控制和数据接口层。不能把 OpenPI 原有的网络结构、训练器或 WebSocket 实现全部算作本项目新增内容；也不能把上游 README 中的 benchmark 成绩视为 ECO65 的实验结果。

## 3. 项目结构与模块职责

以下路径均以本地副本根目录 `/Users/yuyu/Documents/openpi` 为基准。

| 位置 | 作用 |
| --- | --- |
| [模型模块](/Users/yuyu/Documents/openpi/src/openpi/models) | JAX 模型、π0 / π0.5 配置、视觉和语言相关模块 |
| [PyTorch 模型模块](/Users/yuyu/Documents/openpi/src/openpi/models_pytorch) | PyTorch 实现及 Transformers 替换文件 |
| [训练配置](/Users/yuyu/Documents/openpi/src/openpi/training/config.py) | 数据转换组、模型配置、训练超参数、checkpoint 路径 |
| [策略模块](/Users/yuyu/Documents/openpi/src/openpi/policies) | 输入输出适配、模型加载、统一推理接口 |
| [推理服务](/Users/yuyu/Documents/openpi/src/openpi/serving/websocket_policy_server.py) | 对外提供 WebSocket 推理服务 |
| [客户端包](/Users/yuyu/Documents/openpi/packages/openpi-client) | 图像预处理、消息编解码、WebSocket 客户端和动作块执行支持 |
| [本地 robosuite](/Users/yuyu/Documents/openpi/robosuite) | ECO65 / ZhiXing 模型及实际仿真控制代码 |
| [本地 LIBERO](/Users/yuyu/Documents/openpi/third_party/libero) | 任务、场景、环境包装器、MountedECO65 |
| [当前 ZhiXing 客户端](/Users/yuyu/Documents/openpi/examples/libero/main.py) | 直接选择 ECO65，构造 8 维状态并执行策略动作 |
| [模型准备与调试](/Users/yuyu/Documents/openpi/my-robot) | URDF/MJCF 转换、早期模型副本、GUI、ROS 和串口资源 |
| [ECO 数据与推理](/Users/yuyu/Documents/openpi/ECO_data) | CSV、单路图像、离线远程推理脚本和预测文件 |
| [LIBERO 数据副本](/Users/yuyu/Documents/openpi/libreo_raw_data) | 原始 TFRecord 分片、提取图像及转换数据元信息 |

### 3.1 三条不同的工作链路

```text
仿真闭环：
LIBERO 任务与语言指令
  → MountedECO65 + ZhixingGripper
  → 图像与状态适配
  → WebSocket 策略服务 → π0.5 动作预测
  → OSC 位姿控制 + 夹爪动作转换
  → MuJoCo 推进 → 新观测

训练链路：
LIBERO TFRecord
  → LeRobot 格式
  → 归一化统计量
  → pi05_libero_custom 训练配置
  → checkpoint 与 assets
  → 策略服务加载

ECO 离线推理：
CSV + 单路图像
  → 截取状态、复用图像、设置任务提示
  → WebSocket 推理
  → 保存每帧动作块的第一步
```

第三条链路没有调用机械臂驱动执行预测动作。ROS/Modbus 夹爪脚本则是另一项独立资源，尚未在副本中形成统一真机闭环。

## 4. π0.5 模型与服务复用

### 4.1 模型配置

[Pi0Config](/Users/yuyu/Documents/openpi/src/openpi/models/pi0_config.py) 通过 `pi05=True` 选择 π0.5。当前 LIBERO 配置沿用默认 `gemma_2b` 主干和 `gemma_300m` 动作专家配置，内部动作维度为 32，LIBERO 的动作预测长度为 10。

[JAX 模型](/Users/yuyu/Documents/openpi/src/openpi/models/pi0.py) 中，图像与文本形成前缀，动作专家通过 flow matching 预测动作；π0.5 分支使用时间条件 MLP 和 adaRMS。模型内部的 32 维动作空间由数据变换填充，返回环境时再截取前 7 维。

这里有三个不同的“步数”：

| 参数 | 当前相关默认值 | 含义 |
| --- | --- | --- |
| `action_horizon` | 10 | 一次返回的动作序列长度 |
| `sample_actions(num_steps)` | 10 | JAX 模型动作采样的积分步数 |
| `replan_steps` | 5 | 客户端实际执行几步后重新获取观测并推理 |

三者不能互相替代。

### 4.2 状态是否实际进入网络

这是理解当前实现的重要细节：`pi05_libero` 与 `pi05_libero_custom` 均显式设置了 `discrete_state_input=False`。

沿代码追踪：

1. [ModelTransformFactory](/Users/yuyu/Documents/openpi/src/openpi/training/config.py:108) 将该参数传给 `TokenizePrompt`。
2. [TokenizePrompt](/Users/yuyu/Documents/openpi/src/openpi/transforms.py:247) 在该值为 False 时，不把状态数值交给 tokenizer。
3. [Pi0.embed_suffix](/Users/yuyu/Documents/openpi/src/openpi/models/pi0.py:151) 仅在 `not self.pi05` 时创建连续状态 token。
4. PyTorch 模型也使用相同的 π0.5 分支条件。

因此，按副本中这两个配置和当前模型实现，状态仍经过接口传输、归一化、补齐，并用于维度或批次处理，但**状态数值没有作为动作预测的网络条件输入**。当前配置下的预测条件主要来自图像与语言。

这不意味着可以任意删除状态字段：现有预处理和采样接口仍依赖它。也不应直接修改此开关后继续使用旧 checkpoint 而不检查训练配置是否一致。远程运行若采用不同配置，应以实际启动参数为准。

### 4.3 策略服务

[serve_policy.py](/Users/yuyu/Documents/openpi/scripts/serve_policy.py) 提供统一启动入口，默认监听端口 8000。选择 `--env LIBERO` 时，默认使用：

```text
配置：pi05_libero
权重：gs://openpi-assets/checkpoints/pi05_libero
```

[create_trained_policy()](/Users/yuyu/Documents/openpi/src/openpi/policies/policy_config.py:16) 根据 checkpoint 中是否存在 `model.safetensors` 判断 PyTorch 路径，否则加载 JAX 参数。归一化统计量默认从 checkpoint 的 `assets` 中读取。

服务接口继续使用 `WebsocketClientPolicy.infer()`；数组通过项目自带的 MessagePack/Numpy 编解码传输。环境端不需要安装完整训练依赖。

## 5. ECO65 六轴机械臂适配

### 5.1 机器人类与框架注册

[ECO65 类](/Users/yuyu/Documents/openpi/robosuite/models/robots/manipulators/eco65_robot.py) 继承 `ManipulatorModel`，加载 ECO65-B 的 MJCF，并设置：

| 属性 | 当前值 |
| --- | --- |
| 关节数量 | 6 |
| 关节 damping | 每个关节 0.1 |
| 默认安装底座 | `RethinkMount` |
| 默认夹爪 | `ZhixingGripper` |
| 默认控制器名称 | `default_eco65` |
| 初始关节角 | `[0.0, -0.5, -1.5, -2.0, -1.5, -1.5]` |
| 机械臂类型 | `single` |
| 普通桌面 X 偏移 | `-0.16 - table_length / 2` |

机器人模型通过 manipulators 包导入参与注册，同时在 [ROBOT_CLASS_MAPPING](/Users/yuyu/Documents/openpi/robosuite/robots/__init__.py) 中将 `ECO65` 映射到 `SingleArm`。前者解决模型选择，后者解决运行时机器人封装选择。

### 5.2 MJCF 模型

[ECO65 robot.xml](/Users/yuyu/Documents/openpi/robosuite/models/assets/robots/ECO65-B/robot.xml) 定义了底座、六级连杆、关节轴、关节限位、质量与惯性、视觉与碰撞网格以及六个力矩执行器。

| 关节 | XML 角度范围，rad | motor 控制范围 |
| --- | --- | --- |
| `joint_1` | `[-3.106, 3.106]` | `[-60, 60]` |
| `joint_2` | `[-3.106, 2.356]` | `[-60, 60]` |
| `joint_3` | `[-2.792, 2.53]` | `[-30, 30]` |
| `joint_4` | `[-3.106, 3.106]` | `[-10, 10]` |
| `joint_5` | `[-3.106, 3.106]` | `[-10, 10]` |
| `joint_6` | `[-6.28, 6.28]` | `[-10, 10]` |

这些是仿真文件中的参数，不能直接作为真实硬件额定参数的证明。

末端通过 `right_hand` 挂接夹爪，该 body 相对最后一节连杆的位置为 `0 0 -0.02`。腕部相机 `eye_in_hand` 位于该 body 下，位置为 `0.05 0 0`，`fovy=75`。经 robosuite 命名加前缀后，客户端读取 `robot0_eye_in_hand_image`。

机械臂和夹爪 XML 引用的 mesh 文件均已做本地路径检查，未发现缺失。此次未检验网格碰撞效果或动力学参数精度。

### 5.3 LIBERO 中的 MountedECO65

[MountedECO65](/Users/yuyu/Documents/openpi/third_party/libero/libero/libero/envs/robots/mounted_eco65.py) 也继承 `ManipulatorModel`，复用 ECO65-B XML，并将初始姿态改为：

```text
[0.0, -0.3, -1.6, 0.3, 1.5, -1.5]
```

该类补充 `study_table`、`kitchen_table` 的基座偏移，默认夹爪仍为 `ZhixingGripper`。

[桌面任务构造器](/Users/yuyu/Documents/openpi/third_party/libero/libero/libero/envs/problems/libero_tabletop_manipulation.py:22) 会为输入机器人名添加 `Mounted` 前缀，因此典型链路为：

```text
OffScreenRenderEnv(robots=["ECO65"])
  → 桌面任务构造器
  → robots=["MountedECO65"]
  → SingleArm
  → MountedECO65 模型与默认 ZhixingGripper
```

这解释了为什么只修改普通 `ECO65.init_qpos` 不一定改变 LIBERO 桌面任务的初始姿态。

部分地面、茶几和客厅场景使用 `OnTheGround` 前缀。副本没有发现对应的 `OnTheGroundECO65` 类，因此不能默认所有 LIBERO 场景均已覆盖。

## 6. ZhiXing CRT-CTAG2F90 夹爪适配

### 6.1 模型来源与使用位置

[my-robot/Grapper](/Users/yuyu/Documents/openpi/my-robot/Grapper) 保存 URDF、Xacro、mesh、ROS launch 和厂商说明等资源。[convert.py](/Users/yuyu/Documents/openpi/my-robot/convert.py) 使用 MuJoCo 加载 URDF 并导出 XML，脚本当前保留 Windows 绝对路径。

运行时实际由 [zhixing_gripper.py](/Users/yuyu/Documents/openpi/robosuite/models/grippers/zhixing_gripper.py) 加载 [robosuite 下的夹爪 XML](/Users/yuyu/Documents/openpi/robosuite/models/assets/grippers/zhixing_gripper.xml)。修改模型准备目录中的同名 XML，不会自动替换运行时文件。

### 6.2 关节、联动和闭环约束

运行时夹爪的关节顺序为：

| 索引 | 关节名 | 角色 |
| --- | --- | --- |
| 0 | `Left_1_Joint` | 主动驱动关节 |
| 1 | `Left_Support_Joint` | 左侧支撑/指垫关节 |
| 2 | `Left_2_Joint` | 左侧联动关节 |
| 3 | `Right_2_Joint` | 右侧联动关节 |
| 4 | `Right_1_Joint` | 右侧联动关节 |
| 5 | `Right_Support_Joint` | 右侧支撑/指垫关节 |

XML 用五条 `equality/joint` 约束规定主从关节角度关系；另用左右各一条 `connect` 约束连接支撑端点与联动连杆。其设计意图是通过单个输入带动两侧机构，并保持指端机构的配合运动。

`Left_2_Joint` 与主关节同号，其余四个从关节在等式约束中与主关节反号。具体空间运动还取决于各 body 的变换和关节轴方向，不能只按角度正负判断左右开合。

### 6.3 执行器与动作映射

夹爪使用一个 position actuator：

```xml
<position name="gripper" joint="Left_1_Joint"
          ctrlrange="-0.91 0" kp="100"/>
```

Python 类对外声明 `dof=1`，动作转换为：

```text
u_next = clip(u_current + 0.2 × sign(action), -1, 1)
q_target = -0.455 + 0.455 × u_next
```

第二步来自 [Manipulator.grip_action()](/Users/yuyu/Documents/openpi/robosuite/robots/manipulator.py:16) 对执行器控制范围的线性映射。

**（2026-09-27 实测修正）** `q_target = -0.455 + 0.455 × u_next` 这一步与原文一致：`u=-1` → `q_target=-0.91`、`u=+1` → `q_target=0`。但**原文把物理效果写反了**：

| u | q_target | 实测指尖开口 | 实际动作 |
|---|---|---|---|
| `+1` | `0.00` | **11.26 cm** | **张开** |
| `-1` | `-0.91` | **2.28 cm** | **闭合（近闭）** |

即 `u=+1` 实际**张开**、`u=-1` 实际**闭合**，与 robosuite/LIBERO「+1=闭合」的统一约定相反（PandaGripper 在同一机器人、同一协议下实测 `+1` → 0.10 cm 闭合、`-1` → 7.94 cm 张开）。

该极性反转已在 `ZhixingGripper.format_action` 中修正（累积项取负），修正后 `+1=闭合`、`-1=张开`。证据：`agent-work-space/zhixing_adaptation_20260927/results/gripper_polarity.json`。零输入保持当前累加目标；非零输入主要提供方向，其绝对值大小被 `sign()` 消去。

需要区分“每次 format_action 调用增加 0.2”和“每次策略预测增加 0.2”：当前底层控制在仿真子步中也调用夹爪动作转换，因此实际闭合速度应结合仿真步长观察。

### 6.4 接触与末端参考点

左右指垫使用命名碰撞体 `left_fingerpad_col`、`right_fingerpad_col`，配置了摩擦参数和 `condim=4`。Python 类通过 `_important_geoms` 将这些名称暴露给框架。

`grip_site` 定义在夹爪局部 `z=0.12` 的 eef body 下，是控制和观测使用的重要末端参考点。XML 还定义了 `ft_frame`、末端力和力矩传感器。因此控制中的末端位置应理解为框架使用的抓取参考点，而不能直接等同于机械臂法兰中心。

### 6.5 初始化的待核验点

`ZhixingGripper.init_qpos` 返回单元素 `[-0.91]`，但 XML 包含六个关节。[SingleArm.reset()](/Users/yuyu/Documents/openpi/robosuite/robots/single_arm.py:178) 将它赋给全部夹爪关节位置索引，按 NumPy 赋值规则会广播。

**（2026-09-27 已核实并修复）** 运行验证确认这**不是无害的**：XML 的 5 条 joint equality 要求 Left_Support / Right_1 / Right_2 / Right_Support = -Left_1 = **+0.91**，因此 4 个关节符号错误，**最大 equality 残差 1.82 rad（≈104°）**，且**左侧四连杆 connect 残差 34.73 mm**（reset 瞬间即被撑开 3.5 cm）。

现已改为自洽的 6 元素初始化 `[-0.91, 0.91, -0.91, 0.91, 0.91, 0.91]`，实测全部 joint equality 残差 **0.0**、四连杆 connect 残差 **7e-8 m**。

注意：`-0.91` 是执行器行程的**接近闭合**端（并非全开）；全开对应 `Left_1 = 0`。证据：`agent-work-space/zhixing_adaptation_20260927/results/gripper_audit.json`。

## 7. 仿真控制层

### 7.1 实际使用的 OSC 配置

[LIBERO ControlEnv](/Users/yuyu/Documents/openpi/third_party/libero/libero/libero/envs/env_wrapper.py:47) 显式调用 `load_controller_config(default_controller="OSC_POSE")`，并把结果传给机器人。

因此，当前主链路实际使用 [osc_pose.json](/Users/yuyu/Documents/openpi/robosuite/controllers/config/osc_pose.json)，关键参数为：

| 参数 | 值 |
| --- | --- |
| 输入范围 | 每维 `[-1, 1]` |
| 平移输出范围 | 每轴 `[-0.05, 0.05]` |
| 旋转输出范围 | 每轴 `[-0.5, 0.5]` |
| `kp` | 150 |
| `damping_ratio` | 1 |
| `impedance_mode` | `fixed` |
| `control_delta` | True |
| `uncouple_pos_ori` | True |

环境接收的前六维动作是经过归一化范围约定的末端位姿增量指令，控制器缩放后计算目标，再转换成六关节力矩。不能把这六维直接解释成六个关节的目标角度。

### 7.2 default_eco65.json 的兼容性

[default_eco65.json](/Users/yuyu/Documents/openpi/robosuite/controllers/config/robots/default_eco65.json) 使用 `BASIC → body_parts → arms → right` 的嵌套结构，还包含 `input_ref_frame="base"`。

但本地 robosuite 标识为 1.4.1，其 `SingleArm._load_controller()` 默认寻找 `controllers/config/default_eco65.json`，并直接读取顶层控制器类型。当前文件位于 `controllers/config/robots/`，结构也不同。

因此该文件不能视为当前 LIBERO 主链路的实际控制配置；直接创建 ECO65 且不显式提供控制器配置的入口也需要核验。尤其不能根据这个未被主链路加载的文件，断定实际 OSC 在基座坐标系解释动作。

### 7.3 额外关节 PD 稳定项

[SingleArm.control()](/Users/yuyu/Documents/openpi/robosuite/robots/single_arm.py:252) 在 OSC 输出上加入：

```text
tau = tau_OSC + 3.0 × (q_init - q) - 2 × sqrt(3.0) × q_dot
```

随后再按执行器限幅写入 MuJoCo。注释表明其目的在于抑制非冗余机械臂姿态漂移。

这项修改没有按机器人类型加条件，会影响所有使用 `SingleArm` 的机器人；它也未经过零空间投影，所以可能与末端任务控制产生竞争。其稳定性收益与跟踪误差需要通过对照实验评估。

## 8. LIBERO 客户端与观测/动作协议

### 8.1 当前 ZhiXing 入口

[examples/libero/main.py](/Users/yuyu/Documents/openpi/examples/libero/main.py) 显式创建 `OffScreenRenderEnv(robots=["ECO65"])`。因此这份副本中的该脚本已经承担 ECO65 适配入口职责，不能只按目录名把它视作原始 Panda 示例。

### 8.2 输入协议

| 字段 | 构造方式 | 当前形状/类型 |
| --- | --- | --- |
| `observation/image` | `agentview_image` 旋转 180°，等比例补边缩放 | `224×224×3`，uint8 |
| `observation/wrist_image` | `robot0_eye_in_hand_image` 同样处理 | `224×224×3`，uint8 |
| `observation/state` | 末端位置、轴角姿态、两个 Support 关节角拼接 | 8 维 |
| `prompt` | LIBERO 任务语言描述 | 字符串 |

状态的实际构造如下：

```python
state = np.concatenate((
    obs["robot0_eef_pos"],
    _quat2axisangle(obs["robot0_eef_quat"]),
    obs["robot0_gripper_qpos"][[1, 5]],
))
```

两个夹爪值是旋转关节角，不是夹爪张开宽度。它们与 Panda 两个滑动指关节状态具有不同物理含义；保持 8 维只能解决接口形状匹配，不能自动完成语义标定。当前配置对状态数值的实际使用情况见第 4.2 节。

### 8.3 模型适配和归一化

[LiberoInputs](/Users/yuyu/Documents/openpi/src/openpi/policies/libero_policy.py:29) 将输入改写为模型需要的键：外部图像对应 `base_0_rgb`，腕部图像对应 `left_wrist_0_rgb`；不存在的另一腕部图像填零，并在 π0.5 下屏蔽。

π0.5 数据配置默认采用分位数归一化。状态和动作经 `PadStatesAndActions` 补齐到 32 维；输出先反归一化，再由 `LiberoOutputs` 取前 7 维。

归一化统计量必须与训练使用的数据和输出动作约定一致。截短统计量或补齐数组本身不会转换单位、坐标系和夹爪机构含义。

### 8.4 动作与执行周期

返回动作排列为：

```text
[dx, dy, dz, d_rx, d_ry, d_rz, gripper_command]
```

客户端默认取动作块前 5 步加入队列，每个环境步取出一条，队列耗尽后重新推理。LIBERO 包装器默认 `control_freq=20`，因此 5 步对应 0.25 秒仿真时间；这不等于端到端真实时间，网络和模型推理会产生额外耗时。

### 8.5 初始化、成功判断与记录

客户端默认使用 `libero_spatial`，每任务 50 次 rollout，先执行 10 步等待动作。等待动作的机械臂部分为零，夹爪指令为 -1，因此也承担打开夹爪的作用。

ECO65 分支采用环境重新 reset，跳过 Panda 原有的完整 MuJoCo 初始状态恢复。这避免了不同关节结构之间直接恢复状态的问题，但物体初始状态也不再严格等同于标准 benchmark 的预置状态，评估结果应注明这一条件。

代码记录外部和腕部两种视频，并输出任务成功次数与成功率。默认视频文件名只包含任务描述和成功/失败后缀，不包含 episode 编号，同任务同结果的视频可能覆盖。此外，循环内会反复写入 `panda_agentview.png`、`panda_wrist.png`；文件名沿用旧名称，不代表实际机器人仍为 Panda。

## 9. 不同实验入口的区别

| 入口 | 机器人/夹爪 | 状态与初始化 | 用途及注意点 |
| --- | --- | --- | --- |
| [examples/libero/main.py](/Users/yuyu/Documents/openpi/examples/libero/main.py) | 桌面任务为 MountedECO65 + ZhixingGripper | 8 维状态；ECO65 使用 reset | 当前与 ZhiXing 适配直接对应的入口 |
| [examples/libero_eco65/main.py](/Users/yuyu/Documents/openpi/examples/libero_eco65/main.py) | 显式 ECO65，默认夹爪取模型定义 | 拼接全部夹爪 qpos；仍恢复原初始状态 | 当前 ZhiXing 下状态会成为 12 维，需要与统计量和初始化机制核对 |
| [examples/eco65_sim/main.py](/Users/yuyu/Documents/openpi/examples/eco65_sim/main.py) | monkey patch 强制 ECO65 + RethinkGripper | 拼接 Rethink 全部 qpos；仍尝试原始状态恢复 | Rethink 实验入口，带视频保存选择逻辑 |
| [examples/eco65_libero/main.py](/Users/yuyu/Documents/openpi/examples/eco65_libero/main.py) | monkey patch 强制 ECO65 + RethinkGripper | 类似上项 | 包含推理预热、异常处理和视频保存逻辑 |
| [examples/eco65_lift/main.py](/Users/yuyu/Documents/openpi/examples/eco65_lift/main.py) | ECO65 + RethinkGripper | 外部图像、零腕部图像；默认提示 lift the cube | 通过 ActionChunkBroker 驱动 Lift 环境 |
| [my-robot/main.py](/Users/yuyu/Documents/openpi/my-robot/main.py) | 直接创建 ECO65 Lift | 零动作循环 | 模型查看和初步仿真调试；保留 Windows 渲染设置 |

RethinkGripper 虽然动作自由度为 1，但有两个关节状态，因此这些脚本拼接出的状态通常是 `3+3+2=8` 维。部分注释和 README 写成 7 维，不应作为接口依据。

Lift 包装器中 `apply_action()` 更新观测但未更新 `_done`；reset 后 `_done=False`，所以按当前实现通常由最大步数结束循环，不能直接把该入口视作完整的成功率评测程序。

`my-robot/models/` 中的 ECO65 定义还使用 `default_base` 和按 `right` 返回字典的接口，且默认 RethinkGripper；它与运行时 robosuite 模型不同，属于需要区分的模型副本。

## 10. 数据处理与自定义微调

### 10.1 LIBERO 数据转换

[convert_tfrecord_to_lerobot.py](/Users/yuyu/Documents/openpi/scripts/convert_tfrecord_to_lerobot.py) 直接读取 TFRecord，绕过 TFDS 的完整数据集构建过程，允许处理本地已有分片。

转换逻辑包括：按 7 维动作还原时间序列、按 8 维状态还原观测、解码两路 JPEG 为 RGB、提取语言任务、按 episode 保存 LeRobot 数据。输出声明 `robot_type="panda"`、`fps=10`，默认 repo ID 为 `local/libero_spatial`。

脚本默认输出到 LeRobot 的缓存根目录，而不是自动使用副本中已有的 `libreo_raw_data/local/`。若指定输出目录，需要确认训练加载器能从同一根目录读取。脚本在目标 repo 目录已存在时会先删除该目录，再重新转换，重跑前应核对目标路径。

### 10.2 本地可见数据规模

[数据元信息](/Users/yuyu/Documents/openpi/libreo_raw_data/local/libero_spatial/meta/info.json) 声明：

| 项目 | 元信息中的值 |
| --- | --- |
| 格式版本 | LeRobot v2.1 |
| robot_type | panda |
| episodes | 432 |
| frames | 52,970 |
| tasks | 10 |
| fps | 10 |
| state / actions | 8 / 7 维 |
| 图像 | 两路 256×256 RGB |

本地该转换目录仅发现 meta 文件，未发现其声明的 episode parquet 数据文件；以上数字是元信息记录，不等于已验证本地完整数据集可用于训练。原始数据目录仅发现名称为 `00000-of-00016` 的一个 TFRecord 分片，不能据此宣称副本包含全部原始分片。

### 10.3 pi05_libero_custom 配置

[自定义配置](/Users/yuyu/Documents/openpi/src/openpi/training/config.py:970) 使用本地 LIBERO 数据进行 π0.5 微调：

| 参数 | 当前配置 |
| --- | --- |
| 名称 | `pi05_libero_custom` |
| 模型 | `Pi0Config(pi05=True)` |
| action_horizon | 10 |
| discrete_state_input | False |
| repo_id | `local/libero_spatial` |
| prompt_from_task | True |
| extra_delta_transform | False |
| 初始权重 | `gs://openpi-assets/checkpoints/pi05_base/params` |
| batch_size | 16 |
| warmup_steps | 500 |
| peak_lr / decay_lr | `5e-5 / 5e-5` |
| decay_steps | 5,000 |
| 优化器 | AdamW，clip_gradient_norm=1.0 |
| EMA | 0.999 |
| 训练步数 | 5,000 |
| 日志 / 保存间隔 | 50 / 500 |
| keep_period | 1,000 |

虽然配置使用 `CosineDecaySchedule` 类，峰值与末尾学习率相同，因此 warmup 后的调度参数没有设置实际的学习率下降幅度。

该配置没有选择 LoRA variant 或额外冻结过滤器，按配置意图属于全参数微调。此处“自定义”指本地数据源和训练参数，并不意味着数据已经改成 ECO65 采集数据。

### 10.4 数据变换与归一化

`LeRobotLiberoDataConfig` 将 `image / wrist_image / state / actions` 映射到策略接口，训练时从 task 生成 prompt，再执行与推理兼容的数据变换。由于 LIBERO 动作已经使用增量表示，配置关闭额外 delta transform。

[compute_norm_stats.py](/Users/yuyu/Documents/openpi/scripts/compute_norm_stats.py) 针对状态和动作计算统计量。自定义配置的统计量默认写入：

```text
<服务器项目根目录>/assets/pi05_libero_custom/local/libero_spatial/
```

训练 checkpoint 默认位于：

```text
<服务器项目根目录>/checkpoints/pi05_libero_custom/<实验名称>/<保存步数>/
```

推理需要同时对应正确的模型配置、参数文件和统计量资产，不能仅替换权重路径而忽略其训练数据约定。

## 11. ECO 数据的离线远程推理

### 11.1 数据组成

[ECO_data](/Users/yuyu/Documents/openpi/ECO_data) 包含两个图像/CSV 数据块。本次逐行读取 CSV 得到：

| 数据块 | CSV 行数 | observation.state 长度 | action 长度 |
| --- | --- | --- | --- |
| chunk-000 | 136 | 12 | 13 |
| chunk-001 | 151 | 12 | 13 |

CSV 还包含采样时间、状态时间、帧序号、图像文件名和任务列。副本未包含明确解释全部 12/13 维物理含义的数据采集程序，因此不能仅凭数值推断完整字段语义。

### 11.2 当前推理处理方式

[remote_inference_client.py](/Users/yuyu/Documents/openpi/ECO_data/remote_inference_client.py) 默认读取 chunk-001 的前三帧，处理过程为：

1. 从 CSV 读取图像路径和状态，跳过图像文件不存在的行。
2. 将单路图像等比例缩放并补边为 224×224。
3. 同一图像同时作为外部视图和腕部视图。
4. 用 `state[:8]` 构造策略状态。
5. 使用命令行任务提示，默认 `pick up the object`。
6. 调用策略服务，将每个动作块的第一条动作保存到 `inferred_actions.npy`。

该程序没有读取 CSV 的 action 作为误差评测目标，也没有把预测动作发送给机械臂。所以它适合验证数据读取、网络推理与输出保存，不等同于离线精度评估或真机任务执行。

后续若用于训练或真实控制，需要先说明 12 维状态与 13 维 action 的字段、单位、坐标系，以及如何映射到 LIBERO 的 8 维状态和 7 维动作。

## 12. 真机相关资源与集成边界

[GUI2Robot_90.py](/Users/yuyu/Documents/openpi/my-robot/Grapper/scripts/GUI2Robot_90.py) 提供夹爪的独立 ROS/Modbus 控制接口：

| 项目 | 实现 |
| --- | --- |
| 通信库 | minimalmodbus、serial |
| 默认串口 | `/dev/ttyUSB0` |
| 波特率 / 从站号 | 115200 / 1 |
| ROS 输入 | `joint_states` |
| 控制项 | 使能、位置、速度、力、加减速、运动触发、回零 |
| 位置构造 | `int((1 - data.position[2]) * 9000)` |
| 去抖逻辑 | 最近 5 个值中至少 3 次相同，且与上次发送不同才写入 |

该脚本在模块加载时创建串口对象，在主程序中设置使能与运动参数。本文只阅读它，没有执行。

本地未发现 ECO65 专用 SDK 驱动、策略动作到真机控制命令的转换、相机实时采集与机械臂反馈组成的完整控制程序。因此真机整体部署是否已在远程或其他仓库完成，需要另行补充材料。

## 13. 环境与运行流程说明

### 13.1 环境分离

[pyproject.toml](/Users/yuyu/Documents/openpi/pyproject.toml) 要求模型侧 Python ≥3.11；本地依赖记录包含 JAX 0.5.3、Torch 2.7.1、Transformers 4.53.2。

[LIBERO Dockerfile](/Users/yuyu/Documents/openpi/examples/libero/Dockerfile) 使用 Python 3.8 的独立虚拟环境；[模型服务 Dockerfile](/Users/yuyu/Documents/openpi/scripts/docker/serve_policy.Dockerfile) 使用 Python 3.11.9。两个环境通过 WebSocket 通信。

LIBERO 容器设置：

```text
PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero
```

其中 `/app` 位于前面，使项目中的本地 robosuite 优先于环境中安装的包。这是自定义机器人和夹爪生效的重要条件。远程部署应核对 `robosuite.__file__` 的实际位置。

### 13.2 运行命令模板

以下命令依据副本接口整理，供在配置完成的远程服务器上使用；本次未执行。将 `/path/to/server/openpi` 替换为实际项目根目录，并使用对应的模型或 LIBERO 环境。

模型环境中的数据检查、转换和训练：

```bash
cd /path/to/server/openpi

# 只检查原始分片结构
uv run --group rlds scripts/convert_tfrecord_to_lerobot.py \
  --data-dir libreo_raw_data --dry-run

# 正式转换会重建同名输出 repo 目录，先确认目录中没有需要保留的数据
uv run --group rlds scripts/convert_tfrecord_to_lerobot.py \
  --data-dir libreo_raw_data --repo-id local/libero_spatial

uv run scripts/compute_norm_stats.py --config-name pi05_libero_custom

uv run scripts/train.py pi05_libero_custom --exp-name eco65_libero_documented_run
```

最后一个实验名只是命令模板，不表示远程已有该实验。本地配置默认启用 W&B，训练环境和记录方式需与远程已有设置保持一致。

使用默认 LIBERO checkpoint 启动策略服务：

```bash
cd /path/to/server/openpi
uv run scripts/serve_policy.py --env LIBERO
```

使用自定义 checkpoint 时，配置必须匹配其训练来源：

```bash
cd /path/to/server/openpi
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi05_libero_custom \
  --policy.dir /path/to/checkpoints/pi05_libero_custom/experiment/saved_step
```

在独立 LIBERO 环境中运行当前 ZhiXing 入口：

```bash
cd /path/to/server/openpi
PYTHONPATH="$PWD:$PWD/packages/openpi-client/src:$PWD/third_party/libero" \
  python examples/libero/main.py \
  --host 127.0.0.1 --port 8000 \
  --task-suite-name libero_spatial --num-trials-per-task 1
```

客户端与模型分属不同机器时，将 `--host` 替换为模型服务器的可访问地址。

离线 ECO 数据推理：

```bash
cd /path/to/server/openpi
PYTHONPATH="$PWD/packages/openpi-client/src" \
  python ECO_data/remote_inference_client.py \
  --host 127.0.0.1 --port 8000 \
  --chunk ECO_data/vla_dataset/chunk-001 --max-frames 3 \
  --task "pick up the object"
```

### 13.3 Compose 文件的已见差异

| 文件 | 副本中的情况 |
| --- | --- |
| [examples/libero/compose.yml](/Users/yuyu/Documents/openpi/examples/libero/compose.yml) | 内容为双服务部署；文件带 UTF-16 编码，工具兼容性需核对 |
| [examples/libero_eco65/compose.yml](/Users/yuyu/Documents/openpi/examples/libero_eco65/compose.yml) | Dockerfile 路径写成 `examples/libero_coe65/Dockerfile`，与实际目录拼写不同 |
| [examples/eco65_lift/compose.yml](/Users/yuyu/Documents/openpi/examples/eco65_lift/compose.yml) | command 实际启动 eco65_sim 脚本，不是同目录 Lift 脚本 |
| [examples/eco65_libero/compose.yml](/Users/yuyu/Documents/openpi/examples/eco65_libero/compose.yml) | 实际 command 直接启动 main.py；旧 README 所述先运行 patch_libero.py 与当前文件不同 |

因此本文使用独立服务/客户端命令说明流程，没有把这些 Compose 文件描述成已经重新验证可一键运行的部署方案。

## 14. 当前实现的待核验事项

以下是阅读副本得到的检查项，不等于远程服务器已经出现相应故障。本轮仅记录，没有修改。

| 项目 | 依据与影响 | 下一步核验方式 |
| --- | --- | --- |
| 状态输入开关 | 两个 π0.5 LIBERO 配置关闭离散状态输入，π0.5 也无连续状态 token | 核对远程模型配置与 checkpoint；保持图像/提示和随机噪声一致，改变状态观察输出 |
| 夹爪状态语义 | 两个旋转关节角被放到 LIBERO 两指状态槽位 | 定义实际关节角、开口宽度和训练字段关系 |
| 夹爪初始化 | 单元素 init_qpos 广播到六个关节 | 检查 reset 后关节角、约束残差与前几步抖动 |
| 夹爪累加速度 | format_action 在底层控制调用中更新 | 记录每个环境步内的目标变化与实际开合速度 |
| 全局关节 PD | 对全部 SingleArm 生效，并非零空间项 | 对比开关该项前后的末端跟踪误差和姿态稳定性 |
| 控制配置不一致 | default_eco65 文件位置、格式与本地加载器不一致 | 分别检查显式 OSC 路径和默认机器人创建路径 |
| 实验入口混用 | Rethink monkey patch 会覆盖默认 ZhiXing | 启动后记录实际模型、夹爪类和观测维度 |
| 标准初始状态 | 主入口改为随机 reset，其他入口仍使用原状态 | 区分稳定运行验证与严格 benchmark 复现 |
| 场景覆盖 | 未见 OnTheGroundECO65 | 按实际任务场景逐项检查机器人名称注册 |
| 数据完整性 | 转换目录只见 meta；原始 TFRecord 只见一个分片 | 在远程核对 parquet、分片与实际训练 episode 数 |
| 视频覆盖 | 主入口视频名缺少 episode 编号 | 核对每次 rollout 的独立录像与日志对应关系 |
| 真机闭环 | ECO 离线推理与 Modbus 脚本尚未联通 | 补充实际 ECO65 驱动和执行桥接程序 |

## 15. 实验材料与后续整理建议

为了使项目说明进一步成为可复现的实验报告，建议从远程服务器补充以下信息：

| 材料 | 建议记录内容 |
| --- | --- |
| 运行版本 | 代码 commit 或快照日期、实际导入模块路径、Python/CUDA/MuJoCo 版本 |
| 模型来源 | 配置名、checkpoint 路径、JAX/PyTorch 后端、归一化资产路径 |
| 训练记录 | 数据集来源与规模、训练命令、实际结束步数、loss、训练时长、显存 |
| 仿真条件 | 机器人与夹爪类、初始姿态、控制频率、PD 项、相机参数、随机种子 |
| 评测记录 | 每任务成功次数/总次数、初始化方式、独立视频、失败原因 |
| 推理性能 | 冷启动与稳定阶段延迟、动作块长度、重规划间隔 |
| 真机接口 | 机械臂控制 API、动作单位与坐标系、夹爪协议、图像和状态时间对齐 |

建议把后续实验分为模型加载与通信验证、ECO65 单体控制验证、ZhiXing 开合与接触验证、LIBERO 任务评测、真实数据语义适配、真机闭环六个阶段，分别保留配置与结果，便于定位问题来自模型、控制、观测还是硬件执行。

## 16. 可用于项目介绍的文字

本项目基于 OpenPI 开源框架开展 π0.5 视觉语言动作模型复现，并面向 ECO65 六轴机械臂与 ZhiXing CRT-CTAG2F90 双指夹爪进行了工程适配。项目构建了 ECO65 与夹爪的 MuJoCo 模型及 robosuite 注册接口，通过关节联动约束、单维夹爪动作转换、末端位姿控制和 LIBERO 环境扩展，建立了语言任务、双视角观测、远程策略推理与仿真执行之间的接口链路。同时整理了 LIBERO 数据转换、自定义微调配置和 ECO 数据离线推理工具，为后续跨机器人评测、数据适配与真实硬件集成提供基础。

其中，π0.5 复现背景依据项目维护者说明；ECO65/ZhiXing 的具体效果、微调完成情况、任务成功率和真机执行结果，应由远程运行日志与实验材料补充，不从代码存在本身推导。
