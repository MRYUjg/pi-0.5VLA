# PyRUA-Lean 实现逻辑讲解：大模型如何理解并控制机械臂

> 面向 ECO65 接入的源码级讲解
> 依据：PyRUA-Lean 源码（`prompt.py` / `arms.py` / `sandbox.py` / `runner.py` / `_robot.py` / `hosts/rpent_libero.py` / `knowledge/libero.md`）+ RPent 源码
> 所有行号均对应当前版本的实际源码，可在 `gpt6-astra-调研/pyrua-src/` 下核对

---

## 0. 总纲：一个反常但关键的设计决定

普通 VLA 范式是「**图像 → 模型 → 动作向量**」。PyRUA-Lean 不是。

它的范式是：

```
图像 + 状态 + 知识  →  大模型  →  Python 代码  →  沙箱执行  →  原语调用  →  机械臂动作
                        ↑                              ↓
                        └────── 执行回执（文本+状态+图像）──┘
```

**大模型不输出关节角，也不输出末端位姿。大模型写代码。**

这一句话决定了整个项目的所有设计。理由是：机械臂控制的难点不在「单步该往哪动」，而在「**一个持续几分钟、需要反复纠错的闭环**」。让 LLM 在每一步都做一次实时决策，既贵又慢又不稳；让它写一段**能自己循环、自己判断、自己重试**的程序，就把 LLM 从「实时控制器」降级成了「策略作者」——这正好是它擅长的事。

论文标题里的「65% fewer tokens」就来自这里：一次代码调用可以覆盖「找物体 → 移到上方 → 抓取 → 检查夹爪 → 失败重试」整段，而工具调用范式要为此花 4 次 LLM 调用、每次还回传 3 张图。

---

## 1. 五层架构

```
┌───────────────────────────────────────────────────────────────┐
│ 第 5 层  审计层   sandbox.py  —— 能碰到什么、做了什么               │
│   ALLOWED_MODULES / REMOVED_BUILTINS / BLOCKED_NAMES / audit     │
├───────────────────────────────────────────────────────────────┤
│ 第 4 层  协议层   arms.py + mcp_server.py  —— 模型能调什么         │
│   只有两个工具: python(code) / run_program + finish              │
├───────────────────────────────────────────────────────────────┤
│ 第 3 层  提示层   prompt.py  —— 模型看到什么（理解从这来）          │
│   契约 CONTRACT + 自动生成的 API 参考 + knowledge/*.md + TaskCard  │
├───────────────────────────────────────────────────────────────┤
│ 第 2 层  语义层   _robot.py + libero.py  —— 模型的语言是什么        │
│   robo 对象: move_to / segment / pi0_pick ... 结果化 dataclass    │
├───────────────────────────────────────────────────────────────┤
│ 第 1 层  执行层   _backend.py + RPent toolkit  —— 谁去动电机       │
│   Backend 五方法协议 → RPent ToolKit → env server → robosuite    │
└───────────────────────────────────────────────────────────────┘
        ▲                                              │
        │            ECO65 替换点在这一层最底下           │
        └──────────────────────────────────────────────┘
```

**关键认知**：ECO65 的接入只影响第 1 层最底下的环境侧（`LIBERO_ROBOT_BASE=ECO65`）。**上面四层完全不用改** —— 这正是它接入便捷的根本原因。

---

## 2. 模型为什么能「理解」：提示词是拼出来的

这是整个项目最精妙的部分。模型对机器人的理解**不是靠训练，而是靠提示词在每次运行时现场构建**。

### 2.1 提示词的四个组成部分

`prompt.py:418-465` 的 `render_prompt()` 把 system 拼成三块、user 一块：

```python
system = "\n\n".join([
    contract.rstrip(),                                      # ① 契约：你要交什么
    "# API reference (generated from the library the host injects)\n\n" + reference,  # ② API 参考：有什么能力
    know.strip(),                                           # ③ 操作知识：怎么用才对
])
user = card.render()                                        # ④ 任务卡：这一次的具体场景
images = tuple(card.images.values())                        # ⑤ 附带的图像
```

### 2.2 ① 契约（`CONTRACT`，`prompt.py:382-415`）

告诉模型「你不是在聊天，你是在写程序」：

```
You control {robot_blurb} by WRITING A PROGRAM, not by chatting with the robot.
Reply with one complete Python file that defines
    def run(robo):
...
- Success = the benchmark's own task predicate fires (`terminated`) at any
  moment while `run` executes.  The simulator checks it, not you.
```

注意 `{robot_blurb}` 这个占位符 —— 它来自 `ADAPTER.blurb`（`libero.py:946` 写的是 `"a simulated Franka arm in the LIBERO benchmark"`）。
**换 ECO65 时必须改这里**，否则模型从第一句话起就以为自己在操作 Franka。

### 2.3 ② API 参考：从活代码自动生成（`prompt.py:164-225`）

**这是防「提示词漂移」的核心机制。** API 参考不是手写的文档，而是用 `inspect` 从**真实会被注入的那个类**上现场反射出来的：

```python
def _render_callable(owner, name, fn):
    sig = inspect.signature(fn)          # 从真实签名生成
    ...
    doc = inspect.getdoc(fn) or ""       # 从真实 docstring 生成
    return head + "\n" + textwrap.indent(doc, "    ")
```

覆盖范围（`api_reference()`）：
- **类 docstring** —— 机器人的本体描述（坐标系、单位、夹爪语义）
- **Constants** —— 大写类属性，如 `robo.OPEN = -1.0`、`robo.CLOSE = 1.0`
- **Methods and attributes** —— 按 MRO 顺序，基类在前；`_robot.py:106-123` 的 `_public_members()` 只取非下划线开头的
- **Result types** —— 结果 dataclass 的字段名、类型、docstring、property
- **Exceptions** —— `ToolError` / `ValueError` / `EpisodeFinished` 的语义

**设计铁律**（`_robot.py:22-24` 的注释明说）：任何不想让模型看到的成员**必须以下划线开头**。`_backend`、`_ledger`、`_guard`、`_absorb` 全部是下划线，所以模型看不到后门。

> **对 ECO65 的含义**：你只要把 `eco65.py` 的类 docstring 和 docstring 写准，模型看到的 API 参考就自动是 ECO65 的。**不需要维护两份文档**。

### 2.4 ③ 操作知识（`knowledge/libero.md`）

这是**手写的 markdown**，承载「老师傅经验」。它不是 API 文档，而是**关于这个物理世界的判断力**。以 LIBERO 为例，我把它归纳为六类：

| 类别 | 例子（原文摘录） |
|---|---|
| **坐标系与语义** | "「左/右」是机器人自我中心的：+y 是机器人左侧，在 agentview 图像里出现在**右边**" |
| **夹爪物理** | "`gripper_opening` ~0.08 全开，~0 空闭，**0.01–0.05 说明指间有物体**" |
| **运动约束** | "单次 move_to 在 x/y 最多 0.30 m（超了库会抛 ValueError）；长距离拆成 2–3 个途经点" |
| **分档参数** | "`step_clip`：空手或盒子 0.025 / 易拉罐 0.015 / 高瓶子 0.012" |
| **策略顺序** | "**先高位接近，再垂直下降；先垂直抬起，再水平移动**" |
| **失败恢复** | "`Move.reached` 为 false 说明伺服卡住了（IK 极限/接触/预算耗尽）" |

还有**任务级的具体数值**（第 87–93 行），比如"客厅场景夹爪归位 z≈0.68、厨房 1.17、物体场景 0.26"、"BOWL: `eef_y = plate_y + 0.045`"。

> **这是全项目最值得你投入的部分**。参考 `astra-robot-sim2real` 的实测：**给全本体几何能省 57.4% 时间、给同步动作状态能省 68.6%**。知识文件就是这个结论的工程落地——而且它**不含任务配方**（每种物体的答案），只给通用判断力，所以能跨任务迁移。

### 2.5 ④ 任务卡（`TaskCard`，`prompt.py:287-325`）

每次运行现场生成，内容是**与 RPent 工具调用 agent 完全对等的信息**（`prompt.py:250-254` 强调"nothing privileged such as object poses"）：

```
# Task card
- suite: libero_spatial
- task: 7
- seed: 2
- task instruction (authoritative): 'put the bowl on the plate'
- movable objects in the scene: akita_black_bowl_1, plate_1
- initial gripper position [x, y, z]: [0.123, 0.456, 0.789] m, opening 0.080 (open, pointing down)
- simulator step budget: 10000
- attached images: agentview (agentview_high.png), wrist (wrist_high.png). They are step 0 ...
```

⚠️ **注意 object_names 的陷阱**（`knowledge_libero.md:11-13`）：名字带 `_1`/`_2` 后缀，但**没有位姿，视觉上也区分不出来**。所以知识文件明确要求：「靠外观和位置识别目标，并**严格照任务指令执行**」。

还有 **transfer 模式**的警告（`prompt.py:304-315`）：如果程序要在其他 seed 上跑，必须显式告诉模型「图只是示例布局，**不要硬编码像素或世界坐标**」。

### 2.6 ⑤ 图像

`card.images` 来自 `dump_card()`：以 `step=0` 渲染 agentview 和 wrist 两张图，作为多模态输入附在 user 消息里（`hosts/rpent_libero.py:46` 的 `CARD_IMAGES`）。

---

## 3. 模型「理解」得到的状态是什么

`libero.py:52-73` 的 `State` 是冻结 dataclass，**只有这些**：

| 字段 | 内容 | 用途 |
|---|---|---|
| `eef_pos` | 夹爪世界坐标 (x,y,z)，米 | 几何推理的基准点 |
| `eef_quat` | 夹爪姿态四元数 (x,y,z,w) | 判断朝向是否朝下 |
| `gripper_qpos` | 两个手指关节位置 | 细分判断抓没抓住 |
| `gripper_opening` | 开合度标量 | 主判据：0.01–0.05 = 夹住了 |
| `object_names` | 场景中可移动物体名 | **无位姿**，必须靠视觉定位 |
| `task` | 任务指令（权威） | 语义目标 |
| `terminated` / `truncated` | 回合标志 | 判定成功/超预算 |

**关键点：物体位姿从不直接给。** 模型必须通过**感知工具自己算**：
- `segment(prompt)` —— 用 SAM3 在图像上做文本分割，返回 `world_xyz`（可见表面中位数）
- `back_project(row, col)` —— 把像素反投影成世界坐标（依赖深度图 + 相机标定）
- `region_center(...)` —— 容器内部中心（因为 mask 中位数会偏到边缘）
- `world_map(camera)` —— **整张逐像素世界坐标数组**，让模型自己做几何推理

`knowledge_libero.md:72-77` 对这块的教学非常具体：
> "`back_project`：在物体**顶面**上采几个像素取中位数；采到边缘、阴影或到桌面的空隙上会返回几米外的背景点。"

> **这就是「理解」的技术本质**：不是模型"看见"了物体，而是**模型把视觉测量结果和世界坐标做了闭环推理**，再用这些数字去算运动目标。

---

## 4. 「控制」怎么发生：`robo` 对象与原语

### 4.1 模型手里只有两个工具（`arms.py:203-261`）

```python
if self.arm == "cells":
    main = {"name": "python", "description": "Execute Python code in a persistent session where `robo` ..."}
else:
    main = {"name": "run_program", "description": "Run a complete Python program that defines `run(robo)` ..."}
return [main, finish]
```

- `python(code)` —— **交互式逐格模式**（`--arm cells`）：持久命名空间，变量/函数/导入跨次保留
- `run_program(code|path)` —— **整程序模式**（`--arm program`）：每次全新命名空间，程序里定义 `run(robo)`
- `finish(status, summary)` —— 结束回合，`status ∈ {success, failure, stuck}`

**没有第三个工具。** 所有能力都从 `robo` 这一个 Python 对象暴露出去。

### 4.2 `robo` 的方法就是「动作词汇表」

以 `libero.py` 为例，模型可用的原语（部分）：

| 原语 | 语义 | 结果类型（关键字段） |
|---|---|---|
| `move_to(xyz, gripper, step_clip)` | 直线移到世界坐标 | `Move.reached`, `final_eef_pos` |
| `move_pose(...)` | 联动 xyz + 俯仰/偏航（绕过 IK 奇点） | `MovePose.reached` |
| `rotate_wrist(...)` / `rotate_pitch(...)` | 单独调腕部 | `Rotation` |
| `set_gripper(+1, steps=8)` | 夹紧 | `Gripper` |
| `release()` | 松开 | `Release.terminated` |
| `segment(prompt)` | SAM3 文本分割 | `Segment.found`, `.score`, `.world_xyz` |
| `back_project(row, col)` | 像素 → 世界坐标 | `BackProjection` |
| `world_map(camera)` | 逐像素世界坐标 | ndarray |
| `pi0_pick(...)` | **调 π₀.₅ 做一次抓取** | `Pick.success`, `.min_gripper_opening` |
| `pi0_doubled(...)` | π₀.₅ 接触密集型技能（抽屉/门/插入） | `Contact` |

### 4.3 三个设计决定，直接决定 LLM 能不能用得好

**① 结果化，而非异常**（`_backend.py:47-54` 的注释）

> "Measured outcomes (a grasp that did not lift, a servo that stopped short, a segmentation with no mask) are **not** errors: they are returned as result fields so a policy can branch on them."

所以 `Move.reached == False` 是**正常返回值**，模型可以写 `if not m.reached: 换 move_pose 重试`。
如果是抛异常，模型的代码就会被打断，必须靠外层 try/except 兜——**结果化让「重试」变成一等公民**，这正是自主纠错的基础。

**② 动作是阻塞的、带物理反馈的**

`_robot.py:303-330` 的 `_stateful()`：每个有副作用的调用前后都取
- `steps_before = backend.env_steps()` / `steps_after`
- `record.env_steps = steps_after - steps_before`

于是模型能知道「这次移动真实消耗了多少仿真步」，而不只是「我发了指令」。

**③ 回合结束由环境判定，不由模型自报**（`_robot.py:158-160`）

```python
@property
def done(self) -> bool:
    """True once the benchmark's success predicate has fired (sticky)."""
    return self._done or bool(self._backend.solved())
```

成功与否是**仿真器的谓词**说了算。模型只是执行者，不能自称成功——这堵住了「模型自欺」的漏洞。

---

## 5. 闭环的引擎：一格代码执行时到底发生了什么

这是「控制」真正落地的地方。以 `--arm cells` 的 `python(code)` 为例，`arms.py:383-432`：

### 第 1 步：审计（跑之前）

```python
self._last_flags = audit_source(code, workspace=self.workspace)   # 静态标记可疑模式
blocked = blocked_names(code)                                     # 读了 __globals__ 之类？
if blocked:
    extra["exception"] = _blocked_message(blocked, "cell")
    return self._feedback(before, extra)                          # 直接拒绝，不执行
```

### 第 2 步：快照（为算增量）

```python
before = self._snapshot()   # (len(ledger), backend.env_steps())
```

### 第 3 步：执行 + **尾表达式回显**

```python
tree = ast.parse(code)
if tree.body and isinstance(tree.body[-1], ast.Expr):
    tail = tree.body.pop()           # 把最后一行表达式摘出来
...
exec(compile(tree, ..., "exec"), self._globals)
if tail is not None:
    value = eval(compile(ast.Expression(tail.value), ..., "eval"), self._globals)
    if value is not None:
        extra["value"] = _trim(repr(value), 4000)
```

**尾表达式回显**是交互体验的关键：模型写 `robo.state().gripper_opening` 单独一行，就能直接看到数值，**不用 print**。这让它像在用 Jupyter，而不是在发指令。

### 第 4 步：异常捕获（含 `EpisodeFinished`）

```python
except EpisodeFinished as exc:
    extra["episode_finished"] = exc.reason      # 任务成功/超时——正常结束，不是错误
except Exception:
    extra["exception"] = _trim(traceback.format_exc(), 4000)
```

注意 `EpisodeFinished` 继承自 **`BaseException`**（`_backend.py:63-70`），刻意避开 `except Exception`：

> "This derives from `BaseException` (like `KeyboardInterrupt`) so that a policy's `except Exception` blocks do not swallow it."

**模型写的 `try/except Exception` 兜不住"任务已完成"信号**——避免它吞掉终止事件继续乱动。

### 第 5 步：组装回执（`_feedback`，`arms.py:316-343`）

```python
result["primitive_calls"] = [ {tool, kwargs, ok, error, **summary} for r in records ][-40:]
result["moved"]     = moved                                  # 是否有副作用调用
result["env_steps"] = steps_after - steps_before
result["state"]     = _state_dict(self.robo)                 # 最新状态 + gripper_opening
result["episode_done"] = bool(self.robo.done)
```

再叠加 `_attach_images()`：

```python
if self.images == "on-motion" and moved:      # 默认策略
    wanted = list(type(self.robo).ON_MOTION_IMAGES)   # ("agentview_high.png", "wrist_high.png")
elif self.images == "on-demand":
    wanted = self.robo._show_artifacts(requested)     # 只在模型显式 robo.show() 时给
```

**这里省钱的机制**：默认 `on-motion` —— **只在机器人真的动了之后才附图像**。没动的那些格（纯计算、纯推理）不回传任何图。这就是「按需看图」的工程实现。

### 第 6 步：状态持久化（`arms.py:411-415`）

```python
exec(compile(tree, f"<cell {index}>", "exec"), self._globals)
```

`self._globals` 是 `ArmToolkit` 的**实例属性**（`arms.py:178-193`），跨调用存活：

```python
self._globals = {
    "__name__": "__policy__",
    "__builtins__": self._builtins,
    "robo": robo, "math": math, "time": time,
    "display": self._display, "log": print,
    "np": np, "numpy": np,
}
```

所以模型第一次写 `plate = robo.segment("the plate")`，下一次可以直接 `plate.world_xyz` —— **它自己的中间结果就是它的记忆**。论文 README 的示例代码里那句 `plate was located by an earlier cell` 就是这个机制。

---

## 6. 完整数据流（一次成功回合的时序）

```
[预检]  pyrualean card / --dry-run
   └─ host.boot(env_only=True) → dump_card() 只起仿真器，不调模型

[启动]  play_episode()  arms.py/play.py:548-583
   ├─ host.boot(cell, out_dir)                    → RPent env server + Pi0.5 + SAM3 三进程
   ├─ dump_card(toolkit, cell, out_dir)           → step0 图像 + card.json
   ├─ robo = adapter.robot_cls(backend, ...)      → 模型将要操作的那个对象
   ├─ ArmToolkit(robo, arm="cells", ...)          → 包成 MCP 工具面
   ├─ ArmMcpServer(...).start()                   → 本地 HTTP MCP 服务
   └─ build_prompt(...) → prompt.txt              → 契约 + API参考 + 知识 + 任务卡 + 2图

[循环]  Codex / Claude Code 作为 agent runtime
   │
   ├─ LLM 写 cell_001.py:  plate = robo.segment("the plate"); print(plate.world_xyz)
   │    └─ exec → robo.segment → Backend.execute_tool → RPent toolkit → SAM3 server
   │    └─ 回执: value/primitive_calls/state/episode_done   （未移动 → 不附图）
   │
   ├─ LLM 写 cell_002.py:  算目标坐标 + robo.move_to(...)  （用了 cell_001 的 plate 变量）
   │    └─ 回执: moved=True → 附 agentview + wrist 两张图 + 新 state
   │
   ├─ LLM 看图 → 写 cell_003.py: 判断是否到位，不满足则 robo.move_pose(...) 重试
   │    └─ 若 Move.reached == False → 模型读字段，改写策略（知识文件教过它怎么办）
   │
   ├─ ... 直到 robo.done 变 True（仿真器谓词触发）
   │
   └─ LLM 调 finish(status="success", summary="...")

[收尾]  play.py
   ├─ result["environment_success"] = bool(toolkit.solved())   ← 环境说了算
   ├─ 落盘: result.json / calls.jsonl / prompt.txt / cell_*.py
   │        policy_output.txt / episode.mp4 / states.json / codex_events.jsonl
   └─ 关闭 toolkit + 停所有 daemon
```

**注意最后一行**：`environment_success` 来自 `toolkit.solved()`，**不是模型自报的 `finish.status`**。模型说 success 而环境说没成功，记录的是后者。

---

## 7. 三种「失败」的边界与预算控制

### 7.1 三重预算

| 预算 | 机制 | 位置 |
|---|---|---|
| **决策预算** | `--max-turns N`：最多 N 次工具调用 | `arms.py:374-381` `_budget_left()` |
| **墙钟预算** | `budget_s`：`threading.Timer` 触发 `robo._stop_episode("timeout")` | `runner.py:118-128` |
| **仿真步预算** | 环境自己的 `max_episode_steps` | env server |

决策预算耗尽时的处理很讲究（`arms.py:376-381`）：不只是拒绝下一次调用，而是**主动结束回合并让所有后续运动调用抛 `EpisodeFinished`**：

```python
def _budget_left(self, used: int) -> bool:
    if self.max_turns is None or used < self.max_turns:
        return True
    if not self.turn_budget_exhausted:
        self.turn_budget_exhausted = True
        self.robo._stop_episode("call_budget")   # ← 让机械臂停下来，而不只是拒绝消息
    return False
```

**这是安全设计**：预算耗尽的瞬间，机器人不能再动，避免模型在"还能动"的情况下继续瞎试。

### 7.2 硬杀兜底

`runner.py:159-205` 的 `run_host()` 把 host 放在**独立进程组**里跑，超时先 SIGTERM、宽限 30 s 后 SIGKILL；调用者收到 SIGTERM/SIGHUP 会**转发给整个进程组**：

> "so killing the CLI never leaves an orphaned simulator behind."

对真机场景，这条尤其重要。

---

## 8. 沙箱：模型能碰到什么（`sandbox.py`）

设计哲学开门见山（`sandbox.py:11-19`）：

> "**In-process Python cannot be sealed**, so the sandbox does three things: ..."

既然进程内无法真正密封，就做三件务实的事：

### ① 拿掉便利的越狱路径

```python
REMOVED_BUILTINS = {"exec", "eval", "compile", "__import__", "open", "breakpoint", "input", "help", "exit", "quit"}
```
`open` 和 `__import__` **不是删掉，而是换成带守卫的版本**：
- `guarded_import`：白名单 `ALLOWED_MODULES`（numpy/scipy/cv2/PIL/math... 纯计算），外加工作区里的 `.py` 文件
- `guarded_open`：`resolved != root and root not in resolved.parents` → `PermissionError`；连文件描述符（int）都拒绝

### ② 禁止命名内省路径

```python
BLOCKED_NAMES = {"__builtins__", "__closure__", "__code__", "__func__", "__globals__",
                 "__import__", "__loader__", "__self__", "__spec__", "__subclasses__", "__wrapped__"}
```
属性访问、裸名字、**甚至字符串常量**都算——这堵住了 `getattr(obj, "__globals__")` 这条经典逃逸。注意注释里的实情：**`__globals__`/`__func__` 曾被用来在运行时改库**。

### ③ 记录「尝试」而不只是「结果」

`audit_source()` 用 AST 静态扫描，产出 `sandbox_flags`，分类为 `import:` / `host-attr:` / `private-attr:` / `escape:` / `probe:` / `path:`。**这些标记只是信息性的**（运行时守卫才决定什么真能跑），但会写进 `result.json`：

> "so an episode that reached for the simulator's internals is **visible in result.json instead of silently counting as a success**"

**这条设计对科研极重要**：一个"成功"的回合如果动了后门，你事后能查出来。

---

## 9. 这套逻辑映射到 ECO65 时，哪些要改、哪些不用改

| 层 | 是否需要改 | 说明 |
|---|---|---|
| 第 5 层 审计 | **不改** | 与机器人无关 |
| 第 4 层 协议/MCP | **不改** | `python`/`finish` 与机器人无关 |
| 第 3 层 提示 | **必须改 3 处** | ①`CONTRACT` 的 `{robot_blurb}`←`ADAPTER.blurb` ②类 docstring（本体描述）③`knowledge/eco65.md`（操作知识） |
| 第 2 层 语义 | **多数不改** | `move_to`/`segment`/`back_project`/`world_map`/`pi0_pick` 语义通用；`CAMERAS`/`IMAGE_ARTIFACTS`/`WORLD_MAP_ARTIFACTS` 常量需与 ECO65 实际产物对齐 |
| 第 1 层 执行 | **改环境侧一处** | `LIBERO_ROBOT_BASE=ECO65` + robosuite 注册（你已有 `register_eco65.py`） |

### 换成 ECO65 后，提示词里必须重写的具体内容

`knowledge_libero.md` 里这些**都是 Panda 专属**，照抄会误导模型：

| 原文 | 问题 |
|---|---|
| `gripper_opening`: ~0.08 全开 | Panda 指行程。ECO65+panda 夹爪要实测 |
| `step_clip` 0.025 / 0.015 / 0.012 | 与控制器增益和臂长有关，需重测 |
| 夹爪归位 z ≈ 0.68（客厅）/ 1.17（厨房）/ 0.26（物体） | Panda 的臂长与安装高度 |
| `BOWL: eef_y = plate_y + 0.045` | Panda 夹爪宽度导致的偏移 |
| "单次 move_to 最多 0.30 m" | 来自 `MAX_PLANAR_MOVE_M`，是库级常量，通常可保留 |
| `+y 是机器人左侧` | 与基座朝向绑定，**必须重新确认** |

而 `libero.py:331` 的类 docstring 第一句就是：

> "A LIBERO tabletop **Franka** arm with a two-finger gripper."

**这句话会原封不动进提示词。** 这是接 ECO65 时第一个要改的字符串。

---

## 10. 一页速查：机制 → 源码位置

| 机制 | 位置 |
|---|---|
| 提示词四段拼接 | `prompt.py:418-465` `render_prompt()` |
| 契约模板（含 `{robot_blurb}`） | `prompt.py:382-415` `CONTRACT` |
| API 参考从活类反射生成 | `prompt.py:164-225` `api_reference()` |
| 只暴露非下划线成员 | `_robot.py:106-123` `_public_members()` |
| 结果类型渲染 | `prompt.py:88-103` `_render_result_type()` |
| 任务卡渲染（含 transfer 警告） | `prompt.py:287-361` |
| 操作知识加载 | `prompt.py:236-244` `knowledge_text()` |
| 两个工具 + finish | `arms.py:203-261` `get_tools_spec()` |
| 持久命名空间 | `arms.py:178-193` + `:411-415` |
| 尾表达式回显 | `arms.py:409-422` |
| 回执组装（增量/状态/图像） | `arms.py:316-343` `_feedback()` |
| 按需/动后附图 | `arms.py:345-372` `_attach_images()` |
| 决策预算与急停 | `arms.py:374-381` `_budget_left()` |
| 结果化而非异常 | `_backend.py:47-54` `ToolError` 文档 |
| `EpisodeFinished` 继承 BaseException | `_backend.py:63-70` |
| 成功由环境判定 | `_robot.py:158-160` `done` |
| 副作用调用的步数记账 | `_robot.py:303-330` `_stateful()` |
| 沙箱三件事 | `sandbox.py:11-19, 112-148, 237-261` |
| 墙钟看门狗 + 进程组硬杀 | `runner.py:104-156, 159-205` |
| MCP 工具发布（自动批准注解） | `mcp_server.py:52-105, 577-583` |
| 回合编排与落盘 | `hosts/rpent_libero.py:217-338` / `play.py:416-605` |

---

## 11. 一句话总结这套实现逻辑

> **它把「控制机械臂」重新定义成了「写一个操作机器人 Python 对象的程序」，然后把四件事做扎实：**
> **① 让模型通过自动生成的 API 参考 + 手写操作知识准确知道有什么能力、这个物理世界什么脾气；**
> **② 让每个动作都返回可分支的物理结果（而不是抛异常），使重试成为一等公民；**
> **③ 让每格代码的中间变量沉淀成模型的记忆，并把图像按需回传；**
> **④ 把成功判定交给环境、把越狱路径堵死、把预算耗尽变成急停。**
>
> **模型负责"想"，代码负责"跑"，环境负责"判"。** 三者职责分明，这就是它能用大模型驱动机械臂的原因。
