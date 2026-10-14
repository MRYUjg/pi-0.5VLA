# PyRUA-Lean 接入 ECO65 的便捷性分析

> 分析时间：2026-10-02
> 分析依据：PyRUA-Lean 源码（`_robot.py` / `libero.py` / `_backend.py` / `hosts/rpent_libero.py` / `robots.py`）+ RPent 源码（`robots/libero/env_server.py` / `robot_spec.py` / `tools.py`）+ 你工作区已有资产（`examples/eco65_libero/`、本地 `robosuite/` 源码）
> 结论性质：源码级判断。未实际运行，故对「已跑通/未跑通」不做声明。

---

## 一、结论速览

| 场景 | 便捷性 | 工作量 |
|---|---|---|
| **A. ECO65 在 LIBERO 仿真里跑通 PyRUA-Lean** | ⭐⭐⭐⭐⭐ 极便捷 | **1 个环境变量 + 1 处注册**，可做到**零改 PyRUA-Lean 代码** |
| **B. ECO65 加上 agentic 能力（Astra 写代码）** | ⭐⭐⭐⭐ 便捷 | 走 A 即可获得，无需额外工作 |
| **C. 让 Astra 驱动真实 ECO65 硬件** | ⭐⭐ 不便捷 | PyRUA-Lean 是**仿真框架**，真机需自写 backend；PyRUA-Lean 本身帮不上 |

**一句话**：如果你要的是「在 LIBERO 仿真里让 GPT-6 Astra 写代码操作 ECO65」，**这个项目接入成本极低，几乎白送**；如果你要的是「驱动真实 ECO65 机械臂」，**PyRUA-Lean 不是那条路**，它没有真机 backend。

---

## 二、为什么仿真接入几乎零成本：三个源码级证据

### 证据 1：RPent 的 env server 已内置机器人替换钩子

`robots/libero/env_server.py` 第 85–97 行：

```python
"init_params": {
    "camera_heights": 256,
    "camera_widths": 256,
    "camera_depths": True,
    "horizon": max_episode_steps,
    **(
        {"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
        if os.environ.get("LIBERO_ROBOT_BASE")
        else {}
    ),
},
```

`robots` 就是 robosuite `MujocoEnv` 的标准参数（由 `SingleArmEnv.__init__(robots=...)` 消费）。
**设 `LIBERO_ROBOT_BASE=ECO65` 就能让 RPent 的 LIBERO 环境用 ECO65 而非 Panda**，不需要改 RPent，也不需要改 PyRUA-Lean。
RPent 的 LIBERO 栈构建在 `rlinf.envs.sim.libero.libero_env.LiberoEnv` 之上，底层是同一套 robosuite + LIBERO。

### 证据 2：接口维度完全兼容（这是最关键的，也是最容易翻车的地方）

PyRUA-Lean 与 RPent 都硬编码了「**7 维动作 + 2 关节夹爪 qpos**」。我核对了你 ECO65 的实际配置：

| 契约 | 要求 | ECO65 + RethinkGripper 实际 | 兼容 |
|---|---|---|---|
| 动作维度 | **7**（6 OSC_POSE + 1 夹爪） | ECO65 6-DOF 臂 + `RethinkGripper.dof == 1` → 7 | ✅ |
| `robot0_eef_pos` | 3 维 | robosuite 标准观测键，同前缀 | ✅ |
| `robot0_eef_quat` | 4 维 xyzw | 同上 | ✅ |
| `robot0_gripper_qpos` | **2** 维 | `RethinkGripper.init_qpos` 为 `[0.020833, -0.020833]` → 2 关节 | ✅ |

核对细节：
- `RethinkGripper` 源码（`robosuite/models/grippers/rethink_gripper.py`）明确 `dof = 1`，`format_action` 断言 `len(action) == 1`，且**双指 qpos 仍是 2 维** —— 这正是 `gripper_qpos` 需要 2 维的地方。
- 你的 `examples/eco65_libero/main.py` 第 28 行 `LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]`、第 167–175 行 state 拼成 7 维 —— 说明你**已经在 LIBERO 里跑通了 7 维 ECO65**。
- PyRUA-Lean `libero.py` 第 890–906 行从 `robot0_eef_pos` / `robot0_eef_quat` / `robot0_gripper_qpos` 三个键装配 `State`。**键名是 robosuite 按机器人 idn 生成的，与具体机器人无关**，只要 idn 保持 `robot0` 就不变。
- RPent `tools.py` 同样硬编码 `action[6] = gripper`、读 `robot0_eef_quat`（第 364/413/423 行），并注释说明是 **OSC_POSE** 语义 —— 与 ECO65 在 robosuite 注册时用的 `OSC_POSE` 控制器一致。

> 换句话说：**你已经做过的 ECO65 robosuite 注册工作，正好把最难的一步解决了**。PyRUA-Lean / RPent 对 Panda 的假设是「7 维 OSC_POSE 单臂 + 标准 robot0 观测键」，ECO65 满足这套假设。

### 证据 3：注册脚本可以直接复用

你的 `examples/eco65_libero/register_eco65.py` monkey-patch `SingleArmEnv.__init__`：

```python
def _patched_init(self, robots=None, gripper_types=None, **kwargs):
    return _original_init(self, robots=["ECO65"], gripper_types="RethinkGripper", **kwargs)
```

它会覆盖 `robots` 参数 —— 也就是说**它和 `LIBERO_ROBOT_BASE` 是两条等效的路**（一个在 Python 层拦截，一个在 config 层传入），可以互为备份。
`patch_libero.py` 还负责把 ECO65 注册进 `ROBOT_CLASS_MAPPING` 并替换 LIBERO 源码里的 Panda 字符串。

---

## 三、接入方案：最小改动路径

### 路径 1（推荐，零改 PyRUA-Lean 代码）

```bash
# 1) 装 PyRUA-Lean 到 RPent 的 Python 环境
cd PyRUA-Lean && pip install -e /path/to/RPent-env

# 2) 让 RPent 的 env server 用 ECO65
export LIBERO_ROBOT_BASE=ECO65

# 3) 确保 env server 进程里 ECO65 已注册（复用你的 patch）
export PYTHONSTARTUP=...   # 或把 patch 挂进 env_server 的启动路径

# 4) 按原命令跑
pyrualean play --robot libero --suite libero_spatial --task 7 --seed 2 \
    --model gpt-6-astra --max-turns 40 --out runs/eco65_demo --cuda-device 0
```

**关键工程点**：`register_eco65` 的生效位置必须是 **env server 子进程**。
PyRUA-Lean 的 `hosts/rpent_libero.py` 里 `boot()` 调用 `spec.init_runtime(...)` 会 spawn env server（`robot_spec.py` 第 391 行指向 `robots/libero/env_server.py`）。你的 patch 必须在这个子进程的 Python 环境里生效，否则环境还是 Panda。

### 路径 2（更稳，约 5 处小改动）

| # | 改动 | 位置 | 工作量 |
|---|---|---|---|
| 1 | 加 `"eco65"` 到 `ROBOTS` 白名单 | `src/pyrualean/robots.py` 第 25 行 | 1 行 |
| 2 | 写 `src/pyrualean/eco65.py`：照抄 `libero.py`，改 `NAME`/`CAMERAS`/`IMAGE_ARTIFACTS`/`ADAPTER` | 新文件 | 大部分是常量 |
| 3 | 写 `src/pyrualean/hosts/rpent_eco65.py`：照抄 `rpent_libero.py`，把 `robots.libero.robot_spec` 换成你的 | 新文件 | 中 |
| 4 | 写 `src/pyrualean/knowledge/eco65.md`：改写坐标系/单位/夹爪约定/运动上限 | 新文件 | **最容易写错，也最影响效果** |
| 5 | `--primitives no-vla` 或提供 ECO65 的 VLA 端点 | 命令行 | 0 |

`RobotAdapter` 有合理默认值（`guides_subdir=""`、`guide_names=()`、`show_examples` 有默认），所以最小可用版本只需 5 个字段：`name` / `robot_cls` / `host` / `knowledge` / `blurb`。

---

## 四、硬约束与坑（按严重程度排序）

### ⚠️ 坑 1：机器人自述与实际不符会误导 Astra

`libero.py` 第 331 行的类 docstring 是**直接进提示词的**：

> "A LIBERO tabletop **Franka** arm with a two-finger gripper."

第 946 行 `blurb="a simulated **Franka** arm in the LIBERO benchmark"`，第 338 行注释里写着 `CLOSE`（+1）闭合、`OPEN`（-1）张开。

**Astra 读到的机器人描述如果写着 Franka，它就会按 Franka 的几何/行程去规划**。参考 `astra-robot-sim2real` 的实测：完整本体几何信息能带来 57.4% 的时间下降 —— 反过来，**错误的本体描述是负资产**。所以路径 2 的第 4 步不是可选项。

### ⚠️ 坑 2：`open` / `close` 的物理语义可能正好反过来

PyRUA-Lean 走 robosuite 惯例：`-1 = 张开`、`+1 = 闭合`（`libero.py` 第 349–350 行）。
`RethinkGripper.format_action` 也是 `-1 => open, 1 => closed`。
你 `main.py` 里 `LIBERO_DUMMY_ACTION` 的夹爪位是 **`-1.0`**（张开），与此一致。
但**真实 ECO65 夹爪驱动的符号约定很可能相反**，跨到真机时必须实测确认，否则会出现「以为在抓、其实在松」。

### ⚠️ 坑 3：`pi0_pick` / `pi0_doubled` 的域差距没有消失

PyRUA-Lean 的 LIBERO 机器人暴露两个 π₀.₅ 动作原语（`libero.py` 第 369 行 `VLA_PRIMITIVES = ("pi0_pick", "pi0_doubled")`），由 RPent spawn 一个 Pi0.5 server 支撑。
换成 ECO65 后：
- **好处**：整条链路（图像 + 7 维 state → π₀.₅）**自动变成 ECO65 的 state，不再需要你现在的 `stats[:7]` 截断 hack** —— 因为环境本身就是 ECO65。
- **但不解决**：π₀.₅ 的权重/归一化统计仍来自 Panda 数据，域差距只是从「状态拼接」转移成了「模型分布」。
- **建议**：第一次对比就用 `--primitives no-vla`，把 VLA 变量摘掉，纯看 Astra 用分析式原语能做到什么程度，得到干净的基线。

### ⚠️ 坑 4：`LIBERO_ROBOT_BASE` 是单点、无文档、未验证的钩子

全 RPent 仓库里这个环境变量**只出现一次**（`env_server.py` 第 93 行），没有文档、没有测试。
它看起来是为此类需求新加的（或实验性的）。**别把它当成有保障的公共接口**：用之前先在 `--capture-only`/`card` 模式下验证环境里真的是 ECO65（例如打印 `raw_obs` 的关节名或比对 qpos 维度）。

### ⚠️ 坑 5：`world_map` 依赖深度渲染与相机标定

`LiberoRobot` 提供 `WORLD_MAP_ARTIFACTS`（`agentview_world_high.npz` 等），是**逐像素世界坐标**，Astra 用它做几何推理（PyRUA-Lean 示例代码里 `robo.world_map()` 是关键）。
RPent 侧 `build_env_cfg` 已打开 `camera_depths: True` 并注释说明「用于 depth + 相机标定反投影到世界系」。
**若 ECO65 改变了相机安装位姿而标定没同步更新，`world_map` 会给出错误的世界坐标**，Astra 的几何推理会系统性地偏。这是接入后**最值得先验证**的一项。

### ⚠️ 坑 6：wrist 相机与产物名

`IMAGE_ARTIFACTS` 期望 `wrist_high.png` / `wrist.png`（腕部相机）。LIBERO/Panda 的 `robot0_eye_in_hand_image` 提供它；你的 `main.py` 也确实读取 `obs["robot0_eye_in_hand_image"]`，所以 ECO65 侧应该也有。但若腕部相机缺失或名字不同，需改 `IMAGE_ARTIFACTS` —— 改错会直接 `ValueError`，属于**会立刻报错、不会静默失败**的类型，风险低。

---

## 五、能不能驱动真实 ECO65？—— 这是你需要知道的边界

**不能直接用。** 依据：

1. `_backend.py` 的 `Backend` 协议只有 5 个方法（`execute_tool` / `solved` / `artifact` / `env_steps` / `cancel`），**理论上可以自己实现一个真机 backend** —— 这是它设计上唯一的「扩展口」。
2. 但 `RobotBase` 的所有原语都假设 backend 有**录制回放**语义：`state(step=-1)`、`image(camera, step=...)`、`artifact(name, step=...)` 全都要按**历史步**取数据。真机驱动需要自己实现一套「轨迹缓冲 + 图像落盘 + 产物命名」。
3. 更要命的是**任务判定**：`done` 依赖 `backend.solved()`，即**基准自己的成功谓词**。真机上没有 BDDL 判定器。
4. 仓库里没有 ROS / 硬件抽象层。你的真机资产在 `my-robot/`（`Grapper/`、`main.py`）和 `ECO_data/`（`remote_inference_client.py`），与 PyRUA-Lean 没有接口重叠。

> **判断**：如果你要走真机，PyRUA-Lean 的价值是**提供一套可借鉴的 API 设计范式**（结果化返回值、docstring 即提示词、沙箱、调用台账），而不是提供可直接运行的代码。真正的真机参考是 `astra-robot-sim2real`（华为，XLeRobot 真机 + 完整标定材料）。

---

## 六、建议的下一步（按性价比排序）

1. **先验证钩子**（半天）：只跑 `pyrualean card --robot libero --suite libero_spatial --task 0 --seed 0`，设好 `LIBERO_ROBOT_BASE=ECO65`，确认 task card 里的 `eef_pos` / `gripper_opening` / 关节数与 Panda 不同 —— **证明环境确实换成了 ECO65**。这一步不花钱、不调模型。
2. **验证 `world_map` 正确性**（半天）：取一个已知物体位置，用 `world_map` 反算世界坐标与仿真真值比对。坑 5 是静默失败，必须显式验证。
3. **跑一次 `--primitives no-vla` 的纯 Astra 回合**（1 天）：得到 ECO65 上「Astra 只用分析式原语」的干净基线。
4. **再打开 VLA 原语**对比：观察 π₀.₅ 在 ECO65 state 下的表现，与你现在 `examples/eco65_libero/main.py` 的遥控式评测结果对比。

---

## 七、遗留不确定项（如实列出）

- **未实际运行验证**：以上判断基于源码比对，`LIBERO_ROBOT_BASE=ECO65` 是否真能在 RPent 的 `LiberoEnv` 里无副作用生效，需要按第六节第 1 步实测。robosuite 的 `robots` 参数在 LIBERO 的 `OffScreenRenderEnv` 上是否存在与 `SingleArmEnv` 完全一致的传递路径，也建议一并确认。
- **ECO65 的 `ROBOT_CLASS_MAPPING` 注册来源**：我未能在工作区定位到 ECO65 的 robosuite 模型定义文件（`find` 未命中 `ECO65` 目录）。你的 `patch_libero.py` 注释写着「Expecting ECO65 to be pre-registered in robosuite」，说明它来自镜像内的 robosuite 安装，工作区源码树里看不到 —— 我**无法核实**其 `OSC_POSE` 控制器配置与初始位姿，第 2 节表格中「动作维度 7」的推断基于 `main.py` 的实证用法（`[0.0]*6 + [-1.0]`）而非模型文件。
- **Python 版本**：RPent 与 PyRUA-Lean 的环境要求（`docs/setup.md`）我未逐条核对与你现有 uv 环境的兼容性。
- **`LIBERO_ROBOT_BASE` 的引入时间与稳定性**：无法确认它是 RPent 的长期接口还是临时补丁，请勿在生产流程上依赖。

---

## 附：本次分析引用的源码位置

| 文件 | 关键行 | 内容 |
|---|---|---|
| `RPent/robots/libero/env_server.py` | 85–97 | `LIBERO_ROBOT_BASE` → `init_params.robots` 钩子 |
| `RPent/robots/libero/env_server.py` | 103–125 | `make_env` 构建 `LiberoEnv` |
| `RPent/robots/libero/robot_spec.py` | 391 | env server 启动路径（子进程边界） |
| `RPent/robots/libero/tools.py` | 357–370, 413–423 | `action[6] = gripper`、`robot0_eef_quat` 硬编码，注释标明 OSC_POSE |
| `PyRUA-Lean/src/pyrualean/libero.py` | 331–350 | 类 docstring（进提示词）与 OPEN/CLOSE 语义 |
| `PyRUA-Lean/src/pyrualean/libero.py` | 353–369 | CAMERAS / IMAGE_ARTIFACTS / WORLD_MAP_ARTIFACTS / VLA_PRIMITIVES |
| `PyRUA-Lean/src/pyrualean/libero.py` | 890–906 | 从 `robot0_*` 键装配 State（7 维假设） |
| `PyRUA-Lean/src/pyrualean/libero.py` | 941–952 | `ADAPTER` 定义（blurb 写死 Franka） |
| `PyRUA-Lean/src/pyrualean/_backend.py` | 28–45 | Backend 五方法协议（真机扩展点） |
| `PyRUA-Lean/src/pyrualean/robots.py` | 25, 63–72 | `ROBOTS` 白名单与 `get_adapter` |
| `PyRUA-Lean/src/pyrualean/hosts/rpent_libero.py` | 99–176 | `boot()`：spawn RPent 栈、构建 toolkit |
| `robosuite/models/grippers/rethink_gripper.py` | 38–66 | `RethinkGripper.dof == 1`，双指 qpos 2 维 |
| `examples/eco65_libero/register_eco65.py` | 28–36 | 你的 ECO65 monkey-patch（覆盖 robots 参数） |
| `examples/eco65_libero/main.py` | 28, 167–175 | 7 维动作与 7 维 state 的实证用法 |
