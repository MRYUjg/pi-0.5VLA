# GPT-6 Astra 具身机械臂开源项目调研

> 调研时间：2026-10-02
> 调研方法：GitHub API 逐仓库核实（存在性、star、创建时间、文件树、README 正文）+ arXiv abs 页可达性校验 + 关键实现文档下载
> 可信度标注：下文每个项目均标注【已核实代码】/【仅 README 声称】

---

## 0. 一句话结论

GPT-6 Astra 做具身机械臂，**当前开源项目的主流范式不是「训练一个 VLA」，而是把冻结的 GPT-6 Astra 当作策略本身**：
用「当前图像 + 本体状态 + 历史经验」拼上下文，让模型直接输出**动作序列（ee_delta）或直接写控制程序**，再交给底层控制器 / 冻结的 VLA（如 π₀.₅）执行。

这一点对你手上的 openpi / π₀.₅ + ECO65 极其关键：**Astra 不是替代 π₀.₅，最实用的路线是让 Astra 当 π₀.₅ 的「审查者 / 纠错层」或「高层规划器」**。

---

## 1. 三条技术路线（先分清，再选学）

| 路线 | 做法 | 代表项目 | 是否需要训练 |
|---|---|---|---|
| **A. 直接策略（Direct Policy）** | Astra 看到三路相机 + 本体状态 → 输出 `ee_delta` 动作块 → 闭环执行 | RoboICL、RoboDojo 评测、GPT-as-Policy 的 Direct 分支 | 不需要，权重全冻 |
| **B. 程序即动作（Code-as-Policy）** | Astra **写 Python 代码**调用低层 API（移动/夹爪/读图），代码在沙箱里跑 | PyRUA-Lean、astra-robot-sim2real | 不需要，且 token 消耗降 65% |
| **C. 审查 / 混合（Hybrid）** | 冻结 VLA（π₀.₅）出动作，Astra 在关键步审查并纠错 | GPT-as-Policy 的 hybrid 分支、RPent（Harness VLA） | 不需要，改动最小 |

**给 ECO65 的建议顺序：C → B → A。** 理由见第 5 节。

---

## 2. 项目清单（按对「学习实现」的价值排序）

### ⭐⭐⭐ 2.1 RoboICL —— 最适合学「动作接口怎么设计」

- 仓库：<https://github.com/Mosi-AI/RoboICL>｜【已核实代码】
- 论文：<https://arxiv.org/abs/2609.34261>（HTTP 200 已确认）
- 核实数据：star 155、fork 5、创建 2026-09-19、最后推送 2026-10-01、MIT、Python、96 个文件
- 机构：Mosi-AI（作者含 Li Yong-Lu、Tang Yehui 等）

**核心思路**：冻结的通用 VLM + 上下文学习，**不做任何机器人相关参数更新，也不训练 VLA**。
- **演示上下文**：录制的轨迹作为可执行的 few-shot 示例
- **交互记忆**：模型自己执行过的动作、控制器回执、结果观测，滚动保留
- **共享交互语法**：把上面两者统一成 `observation -> action -> receipt -> observation` 记录
- **有界锚定记忆**：保留早期/中期的关键交互 + 最新的完整交互，用于立即纠错

**最值得抄的配置文件**（`configs/shared_harness.json`，我已完整下载）：
```json
{
  "predict_horizon": 15, "execute_horizon": 15, "feedback_frames": 1,
  "feedback_cameras": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
  "action_space": "ee_delta",          // 末端增量动作空间 → 换机械臂时最省事
  "delta_frame": "world",
  "max_translation_m": 0.03,           // 单步位移硬限幅
  "max_rotation_rad": 0.15,            // 单步旋转硬限幅
  "max_joint_increment_rad": 0.15,
  "ik_continuity_guard": true,         // IK 连续性保护，防跳变
  "max_action_corrections": 2,         // 动作越界后允许模型自我修正的次数
  "image_layout": "triptych",          // 三相机拼成一张图给模型
  "image_profile": "native_jpeg", "image_detail": "original",
  "reasoning_effort": "xhigh",         // 与 GPT-as-Policy 一致：xhigh 推理档
  "max_context_images": 50, "live_anchor_count": 5
}
```
> **学习要点**：这一份 JSON 就是「把 VLM 变成机器人策略」的工程答案——动作限幅、IK 保护、图像布局、记忆裁剪、推理档位，全是踩过坑的参数。换到 ECO65 时这套约束框架可以直接复用。

**运行成本**：需要 Linux + NVIDIA GPU + Vulkan/CUDA、Isaac Sim 5.1 兼容环境、RoboDojo 资产（Git LFS）。仿真与策略进程**故意用两个 Python 解释器**（websockets 依赖冲突）。`configs/upstream.lock.json` 锁定 RoboDojo/XPolicyLab/IsaacLab/cuRobo 的确切 commit。
- API key 环境变量：`ASTRA_API_KEY`
- 调试友好：`--dry-run`（不调 API 校验环境）、`--capture-only`（只验证三路相机流）

### ⭐⭐⭐ 2.2 GPT-as-Policy —— 与你的 π₀.₅ 场景重合度最高

- 仓库：<https://github.com/anonymous-report-421/GPT-as-Policy>｜【已核实代码】
- 核实数据：**star 565、fork 25**（本批最热）、创建 2026-09-13、MIT、852 个文件
- 归属：Galbot（银河通用），技术报告作者含 Li Yi、He Wang 等

**做法**：在 10 个 RoboDojo 任务上（每任务 5 个对齐用例）评测两个策略：
1. **GPT 6 Astra Direct**：成功率 26%，平均 Score 37.81（48 个有原生分的 episode）
2. **Hybrid**：Astra **审查并可选修正 π₀.₅ 的动作** → 平均 Score **62.60**、成功率 **48%**，且**只在 14.4% 的控制步上做修正**

> **这是全批次里对你最有价值的一条结论**：不改 π₀.₅ 权重、只在少数控制步让 Astra 介入，就能把成功率从 26% 提到 48%。你遇到「换 ECO65 后效果变差」，正是这类「底座模型 + 轻量纠错」方案的目标场景。

**实现骨架**（`hybrid_rollout/`）：
```text
robodojo/   仿真、policy server、控制器、skills、scheduler
robolab/    独立的 RoboLab 集成
report_site/ 中英双语报告源码 + 打包工具
public_results/evaluation_cases.json   ← 跨策略对齐 task/scene/seed
```
- 模型固定为 `gpt-6-astra` + `xhigh` 推理，**无 provider fallback**
- 切换开关：`ROLLOUT_EVALUATION_METHOD=pi05_plus_gpt`（混合）或 `gpt_only`（纯 Astra）
- 关键提示：需要 **Isaac Sim 5.1 + RoboDojo 源码/资产 + π₀.₅ 的 OpenPI/JAX checkpoint**；仓库明确声明**不是独立可分发的仿真器**
- ⚠️ 诚实提示：仓库声明「**本次发布未重跑付费模型或 GPU 实验**」，且需要你自己的授权账号/网关；网关 URL 是 `.invalid` 占位符

### ⭐⭐⭐ 2.3 PyRUA-Lean —— 学「程序即动作」+ 真实机器人接入契约

- 仓库：<https://github.com/DAGroup-PKU/PyRUA-Lean>｜【已核实代码】
- 论文：<https://arxiv.org/abs/2610.01939>（HTTP 200 已确认）
- 核实数据：star 5、创建 2026-09-25、**最后推送 2026-10-02（今日，仍活跃）**、Apache-2.0
- 有中文 README（`README.zh-CN.md`）和中文指南（`docs/guide.zh-CN.md`）

**核心思路**：把机器人变成 Python 对象 `robo`，agent 只拿到**一个工具 `python(code)`**，在一个持久命名空间里写代码。一次 LLM 调用可以完成「找物体 → 移到上方 → 抓取 → 检查夹爪 → 重试」。

**效果（README 声称）**：九个子套件上持平或更高（整体 63.1% → 71.7%），每个已解决 episode 的输入 token 少 1.5–4.5 倍，费用少 1.2–3.1 倍。
> ⚠️ 一处不一致我如实记录：仓库标题写「14% Higher Success」，但 GitHub API 返回的 description 与某些索引里写「12%」；star 仅 5，属**新仓库、低社区验证**，效果请自行复现确认。

**支持的机器人**（`--robot`）：
| 值 | Benchmark | 机器人 / 技能栈 |
|---|---|---|
| `libero`（默认） | LIBERO-PRO | Franka 臂；脚本伺服 + **Pi0.5 VLA** + SAM3 感知 |
| `robotwin` | RoboTwin 2.0 | 双臂；LingBot-VLA |
| `robocasa` | RoboCasa365 | 移动机械臂；RLDX-1 VLA + 导航 |

**给新机器人接入的三件套**（`docs/multi-robot.md`，**这是把 Astra 接到 ECO65 的最佳模板**）：
1. `src/pyrualean/<robot>.py` —— 面向策略的 API 类：每个 RPent 工具一个方法，**docstring 就是提示词**
2. `src/pyrualean/hosts/rpent_<robot>.py` —— 启动 RPent 服务、构建 toolkit、导出 task card
3. `src/pyrualean/knowledge/<robot>.md` —— 操作知识（坐标系/单位、夹爪约定、运动上限、VLA 规则、恢复策略）

**值得学的工程细节**：
- **结果而非异常**：运动调用返回实际发生了什么（`Move.reached`、`Pick.success`、`found`），误用才抛异常
- **提示词不会漂移**：agent 读的 API 参考是**从活类自动生成**的
- **沙箱**：agent 代码固定可导入集合、文件访问限定在其 workspace、**拿不到 backend 句柄**
- **无 VLA 模式**：`--primitives no-vla` 可把学到的 VLA 原语从 API 参考中摘掉（做消融很关键）
- **双运行时**：`--runtime codex|claude`，Codex CLI 0.155.1 或 Claude Code
- **可审计产物**：episode 视频、每个 cell 的代码+输出、原语调用台账、token 用量、benchmark 自己的判定

**依赖**：底层驱动依赖 [RPent](https://github.com/RLinf/RPent)（star 1337、Apache-2.0、2026-10-01 仍活跃），需先装好你要的机器人栈；`pip install -e .` 装进 RPent 的 Python 环境。

### ⭐⭐ 2.4 astra-robot-sim2real —— 唯一「真实硬件 + sim2real」完整数据开源

- 仓库：<https://github.com/hesd10/astra-robot-sim2real>｜【已核实代码】
- 论文：<https://arxiv.org/abs/2609.31770>（HTTP 200 已确认）
- 机构：华为（Sida He、Lingxi Xie、Qi Tian 等）
- 核实数据：star 7、创建 2026-09-23、1086 个文件、**无 license** ⚠️

**任务**：操纵**真实 XLeRobot**（改装四轮麦轮底盘 + 双臂 + 云台头 + 三相机）按电梯按钮。

**做法**：Astra 读取**提供的材料**（机器人几何、机械 XML、mesh、关节映射、相机标定、历史经验），看当前相机图像，然后**针对低层控制 API 写程序**，权重保持冻结。

**关键实验结论（可直接指导你的提示词工程）**：
| 条件 | 相对基线的时间下降 |
|---|---|
| 完整机器人几何 + 相机信息（I3） | **57.4%** |
| 图像 + 同步的动作与状态（E3） | **68.6%** |
| 原起点的经验迁移到新起点（9 个点，10–100cm） | **58–63%**，且**每个点都更快** |
| 提供可复用局部技能 STEP / LOOP | **29.0% / 30.6%** |
| sim2real：仿真资产 / 仿真经验 | **53.0% / 49.9%** |

> **学习要点**：这说明**「喂给模型多少机器人本体知识」比模型本身更决定成败**——正好能解释你「换 ECO65 后效果变差」的一部分原因（几何/运动学/相机标定与训练分布不一致）。

**可直接复用的资产**：
- `docs/REAL_ROBOT.md`：真实机器人**标定与调试流程**（命名 profile、备份电机寄存器、逐关节测正方向与量程、轮径/轮距/轴距、相机角色识别、速度加速度 profile、收尾恢复参数并释放扭矩）
- `docs/METHODS.md`：信息/经验条件分级（I0–I3、E0–E3）与计时口径定义
- `real/`：硬件控制、标定、实验面板源码（用到 `scservo_sdk`）
- `analysis/`：从逐 trial JSON 离线重建所有图表（`python analysis/build_results.py`）
- 还有真实机器人视频（Git LFS，6× 加速 + 原始速度）

### ⭐⭐ 2.5 权威评测数据 —— 校准预期用

- **RoboDojo GPT-6 Astra 评测**：<https://robodojo-benchmark.com/report/gpt-6-astra-eval>
  42 个仿真任务、2100 trials、单 seed，RoboProbe 固定非学习 harness：**平均 Score 28.97、平均成功率 22.48%**；真实机器人结果因安全原因中止测试，仅作诊断。
- **RoboCurve GPT-6 Astra 评测**：<https://openai.robocurve.org/gpt-6-astra/>
  受控 YAM 臂对比，声称 20 次碗任务完成 19 次，输出 token 减少 80%。（该站疑似为 OpenAI 相关）
- **RPent 排行榜**：<https://github.com/RLinf/RPent#leaderboard>
  Astra 在 LIBERO-PRO 全部八个套件 **92.63%（741/800）**，对比 RPent/Claude Opus 4.7 的 82.4%、冻结 π_RLinf VLA 的 50.0%；RoboCasa365 Target50 上 **59.20%**。
- **Awesome-Astra-Embodied-AI**：<https://github.com/zjwzcx/Awesome-Astra-Embodied-AI>（star 1071）
  37 个案例的持续清单，按「零样本控制（仿真 12 / 真实 10）」「Agentic Policy Calls」「Real-to-sim 回放」「Astra 搭 RL 环境」分类，**含失败案例章节**。我已下载全文到本目录 `README_Awesome-Astra.md`。
  - 其中真实机器人案例值得注意：`cheng-haha/GPT-Policy-Eval`（单次视频演示 → 真实接触密集型插头插入，无 VLA/RL/DAgger）、Piper 臂 + RealSense 直接控制、Loop-ROS 切黄瓜、ENPIRE harness 上下文学习等。⚠️ 清单里的 `cheng-haha/GPT-Policy-Eval` 我实测 GitHub API 返回 **301**（已重命名或转移），请以重定向后的地址为准。

---

## 3. 通用技术骨架（所有项目共有的模式）

```
        ┌─────────────────────────────────────────────────┐
        │  GPT-6 Astra (frozen, reasoning=xhigh)          │
        │   输入: 三路相机拼图 + 本体状态 + 历史交互 + 参考演示   │
        │   输出: 动作块(ee_delta) 或 控制程序(Python)        │
        └─────────────────────────────────────────────────┘
                 ↑ 观测                      ↓ 动作/代码
        ┌─────────────────────────────────────────────────┐
        │  安全层: 限幅/IK连续性/越界重试(max_corrections)   │
        ├─────────────────────────────────────────────────┤
        │  执行层: 控制器 或 冻结 VLA(π₀.₅) 或 分析式原语      │
        ├─────────────────────────────────────────────────┤
        │  记忆层: 有界锚定记忆(早期锚点 + 最新完整交互)        │
        └─────────────────────────────────────────────────┘
```

**六个必须自己实现的工程点**：
1. **动作空间选 `ee_delta` 而不是关节角** —— 换机械臂时模型输出语义不变（RoboICL 的选择）
2. **硬限幅 + IK 连续性保护** —— `max_translation_m` / `max_rotation_rad` / `ik_continuity_guard`
3. **结果化 API** —— 返回 `reached/success/found` 而不是抛异常，让模型能自我纠错
4. **图像布局统一为 triptych** —— 三相机拼一张，原生 JPEG、original 精度
5. **记忆有界** —— 锚点 + 最新，`max_context_images` 上限，否则上下文爆炸
6. **API 参考自动生成** —— 从活代码生成提示词，避免文档漂移（PyRUA-Lean）

---

## 4. 学习路径建议（4 步，从便宜到贵）

**第 1 步（半天，零硬件）—— 只读代码，理解接口设计**
- 精读 `Mosi-AI/RoboICL` 的 `configs/shared_harness.json`（本目录已存）与 `roboicl/policy/` 目录
- 精读 `DAGroup-PKU/PyRUA-Lean` 的 `docs/guide.zh-CN.md` + `docs/multi-robot.md`（本目录已存后者）
- 目标：能画出「观测→提示词→动作→回执」的数据流

**第 2 步（1–2 天，零硬件）—— 跑通 PyRUA-Lean 的无仿真模式**
```bash
git clone https://github.com/DAGroup-PKU/PyRUA-Lean.git && cd PyRUA-Lean
pip install -e .        # 装进你 RPent 机器人栈的 Python 环境
pyrualean api           # 只起 agent 能看到的机器人 API，不需要仿真器
```
- 这一步能让你**看到 Astra 实际写的代码长什么样**，成本最低、信息量最大
- 需要 Codex CLI 0.155.1（或 Claude Code）+ 你自己的 Astra 授权账号

**第 3 步（1 周，需 GPU）—— 用 RoboICL 在仿真里闭环**
- 环境重（Isaac Sim 5.1 + RoboDojo 资产 + 双解释器），但流程规范：
  `--dry-run` → `--capture-only` → `--shots 0` 零样本 → `--shots 1` 单次演示
- 目标：亲手验证「动作限幅 + 记忆裁剪」这些参数对成功率的影响

**第 4 步（对接你的 ECO65）—— 走混合路线**
- 参考 `anonymous-report-421/GPT-as-Policy` 的 `hybrid_rollout/`：让 π₀.₅ 出动作，Astra 只在关键步审查
- 参考 `PyRUA-Lean` 的三件套把 ECO65 注册成一个新 robot
- 参考 `hesd10/astra-robot-sim2real` 的 `docs/REAL_ROBOT.md` 做 ECO65 标定与安全边界

---

## 5. 与你 openpi / ECO65 现状的结合点

你当前状态（据工作区）：openpi + π₀.₅，机械臂换成 **ECO65 + panda 夹爪**，`agent-work-space/` 下有 `换机械臂为何变差_分析.md`、`夹爪对照实验结论.md`、`examples/eco65_libero/`。

**为什么这批项目对你有用**：
1. **你问的「π₀.₅ 输入输出不是与机械臂无关吗」**——π₀.₅ 学习的是「图像+语言 → 动作 token」的映射，但**动作 token 的语义、关节顺序、行程范围、相机视角全部与具体机械臂强耦合**。GPT-as-Policy 的 hybrid 结果（26% → 48%）恰好说明：底座策略在换本体后掉性能时，**外挂一层 VLM 审查是性价比最高的补救**，不需要重训。
2. **Astra 的 `ee_delta` 动作空间天然跨本体**：RoboICL 用世界坐标系末端增量，换臂时只需保证 IK 与限幅正确——这比重新对齐关节空间动作 token 省事得多。
3. **astra-robot-sim2real 的 I3/E3 结论**可以直接用来诊断你的 ECO65 问题：先确认提示词/配置里是否给了**完整的机器人几何 + 相机标定 + 同步的动作状态记录**。缺这些，模型表现会差 50%+。

**成本与风险提示（务必先看）**：
- **API 价格**：多方报道 GPT-6 Astra 定价约为上一代 **2.5 倍**（<https://m.eepw.com.cn/article/202609/483318.html>）。RoboICL 用 `reasoning_effort: xhigh` + 最多 50 张上下文图，**单次决策的 token 成本很高**；真实机器人闭环的推理延迟也需要实测。
- **真实硬件项目极少**：本轮核实到有真实机械臂代码/数据的只有 `astra-robot-sim2real`（XLeRobot）和依赖 RPent 栈的 `PyRUA-Lean` / RPent（LIBERO/RoboTwin/RoboCasa 仿真 + 部分真机 demo）。**Astra 直控真实 ECO65 目前没有现成开源项目**，需要自己接。
- **活跃度差异大**：RoboICL（155★）、GPT-as-Policy（565★）、RPent（1337★）社区验证较多；PyRUA-Lean（5★）、astra-robot-sim2real（7★）很新，**效果数据基本只有作者自述**。
- **许可**：`astra-robot-sim2real` **无 license**，商用前需联系作者；RoboICL 为 MIT，PyRUA-Lean 为 Apache-2.0，GPT-as-Policy 为 MIT。
- **复现成本**：多数项目依赖 Isaac Sim 5.1 + 大量仿真资产 + 自有 GPU 集群；GPT-as-Policy 明确说「未重跑付费模型或 GPU 实验」，其发布快照**不是一键可跑的配方**。

---

## 6. 本次调研的未决项（如实列出）

- OpenAI 官方 GPT-6 Astra 页面 <https://openai.com/index/gpt-6-astra/> 抓取返回 **403**，我未能直接阅读官方系统卡；关于模型能力/定价的描述来自二手报道，请以官方文档为准。
- <https://developers.openai.com/api/docs/models/gpt-6-astra> 未直接验证；调用前请确认**官方是否提供机器人/具身专用端点**，还是需要像这些项目一样自己封装。
- 各项目的成功率数字均为**作者自述**，RoboDojo 官方评测只有 22.48% 平均成功率，与部分项目声称的高成功率口径不同（任务集/交互轮数/是否带记忆），不要横向直接比较。

---

## 附：本目录已下载的原始材料

| 文件 | 来源 |
|---|---|
| `README_RoboICL.md` | Mosi-AI/RoboICL |
| `CFG_RoboICL_harness.json` | Mosi-AI/RoboICL `configs/shared_harness.json` |
| `README_GPT-as-Policy.md` | anonymous-report-421/GPT-as-Policy |
| `README_PyRUA-Lean.md` | DAGroup-PKU/PyRUA-Lean |
| `DOC_PyRUA_multi-robot.md` | 同上 `docs/multi-robot.md` |
| `DOC_PyRUA_setup.md` | 同上 `docs/setup.md` |
| `README_astra-sim2real.md` | hesd10/astra-robot-sim2real |
| `DOC_sim2real_REAL_ROBOT.md` | 同上 `docs/REAL_ROBOT.md` |
| `DOC_sim2real_METHODS.md` | 同上 `docs/METHODS.md` |
| `README_Awesome-Astra.md` | zjwzcx/Awesome-Astra-Embodied-AI（37 案例清单全文） |
