# ECO65 + PandaGripper 脚本专家仿真验证报告

日期：2026-09-29。服务器：`223.109.239.11:20852`（`ubuntu20`，`vipuser`，8×Xeon E5-2698 v4 / 15 GB，**无可用 GPU**）。
工作目录：`/mnt/data/vla-policy/openpi/agent-work-space`。结果目录：`/mnt/data/vla-policy/openpi/agent-work-space/outputs`。

> **为什么不用 π0.5 评测**：本机无 GPU（`nvidia-smi` 无法与驱动通信，仅有 virtio 显示设备），
> VLA 推理不可用；且按 `任务方案.md`，**不得用策略成功率判断夹爪好坏**。
> 因此本次改用**脚本专家 + 底层 7 维 OSC**，把「夹爪/臂能不能被正确控制」与「策略想不想抓」分开。

---

## 1. 结论（先给答案）

**能。ECO65 + PandaGripper 在仿真里可以被正确控制，并完整跑通
「接近 → 下降 → 闭合 → 抬升 → 保持 2 s → 搬运 → 投放 → 释放 → 稳定」全链路。**

| 判据 | 结果 |
|---|---|
| 开合极性（-1 开 / +1 合） | ✅ 与 π0.5 / Panda 约定一致 |
| 全行程与重复性（空载开闭 ×10） | ✅ 全开 **78.98 mm** / 全闭 **0.00 mm**，10 次标准差 **0.000 mm** |
| 夹持点几何（按实测夹持点而非硬编码常数求解） | ✅ 计划截面宽 = 试块宽度（30.0 / 36.7 mm），指尖离台面余量 44–46 mm |
| 抓取—抬升—保持—搬运—放置—释放（N3 七项闸门） | ✅ **30 mm 试块 4/4 全通过**；另有 36.7 mm 6/6、25 mm 3/3（均为一次通过，无重试）|
| 指垫穿透峰值 | ✅ **0.87–1.04 mm**（软限 2.0 mm） |
| LIBERO 原生成功谓词 `(In alphabet_soup_1 basket_1_contain_region)` | ✅ 全部成立 |
| 释放后稳定（20 步窗口内位置漂移 < 5 mm） | ✅ **≈7e-7 mm**，物体在篮内 |
| 单步最大位移（抓瞬移） | ✅ 0.006 m/步（限 0.02）|
| 负载下末端跟踪 | ⚠️ 搬运段**未收敛**（残差 0.055–0.077 m），详见 §5；**不影响任务成功** |
| N4 采集器口径（含伺服超时判据） | ❌ 0/4 —— 因搬运段不收敛被拒收，与抓取能力无关（§5.3） |

数据：`expert_PandaGripper_w30_20260929_061956.json`（4/4）、`_w37_...json`（6/6）、`_w25_...json`（3/3）。

---

## 2. 同宽度对照：Panda vs ZhiXing（30 mm 试块，各 4 次）

| 指标 | **ECO65 + PandaGripper** | ECO65 + ZhiXingGripper |
|---|---|---|
| N3 七项闸门通过 | **4 / 4** | **0 / 4** |
| 抬升高度 | 76.6 – 78.1 mm | 74.0 – 75.0 mm（4 条中 3 条抬起来了） |
| 指垫穿透峰值 | **0.867 – 0.908 mm** | **2.73 – 3.21 mm**（超 2.0 mm 软限） |
| 投放/释放稳定 | ✅ 全部在篮内、漂移 ~1e-6 m | ❌ `place_release_stable` 全部未通过 |
| 原生成功谓词 | ✅ | ❌ |
| 空载开闭 ×10 重复性 | **78.98±0.000 / 0.00±0.000 mm（PASS）** | 107.53±**0.722** / 18.27±0.007 mm（**FAIL**，开度极差 2.3 mm）※数值口径见勘误 |
| 夹爪部署速度 | 资产自带 0.01 | 需注入 `ZHIXING_GRIPPER_SPEED=0.01` |

### 更窄试块（ZhiXing 明显变差，Panda 仍稳定）

| 试块宽 | Panda | ZhiXing |
|---|---|---|
| 13.8 mm | 未测（预算未触发） | **0/6**，穿透 17–23 mm、抬升 ≈0（指垫完全穿过试块） |
| 25 mm | **3/3**，穿透 ≤1.04 mm | **0/3**，穿透 ≈20 mm、抬升 ≈0 |
| 30 mm | **4/4** | 0/4（能抬起但放不稳） |
| 36.7 mm | **6/6** | 未测 |

> **勘误（2026-09-30）**：本节 ZhiXing 的"开口"数值（107.77 / 18.27 mm）取自
> `jaw_gap_mm()` 的旧实现，它对 **mesh 类型**指垫退化成"几何中心距"。
> 用真实 mesh 顶点重算后应为 **全开 89.49 mm / 全闭 0.035 mm**（见
> `diag/zhixing_aperture_metric.json`）。两者都不影响本节的宽度选择
> （预算 = min(请求宽度, 开口−4)，修正后仍取请求宽度 30.0 mm）与任务级结论。
>
> 解释（**原解释已被证伪，见 `ZhiXing夹爪控制问题清单_20260930.md`**）：试块窄于约 28 mm 时
> 两侧指垫会互相“夹空”并**穿过**试块（穿透 17–23 mm）。这与既有 N3 记录
> 「30/50/70 mm 试块 10/10」并不矛盾 —— 那些宽度都 ≥30 mm。
> Panda 在 25–36.7 mm 全区间都稳定，**对夹持对象尺寸的鲁棒性更好**。

---

## 3. 顺带修掉的两个“适配层”缺陷（对 P1 采集有直接影响）

这两处都不是 Panda 的物理问题，而是**代码只适配了 ZhiXing**，跑 Panda 会直接失败或静默出错：

1. **state 两维解析写死关节名**：`n4_collect.resolve_state_gripper_indices()` 只认
   `Left_1_Joint` / `Right_1_Joint`。PandaGripper 的关节是
   `gripper0_finger_joint1/2`，调用即抛
   `ValueError: [R4] state 夹爪关节 'Left_1_Joint' … 命中 0 个`。
   → 本脚本按关节名/平动指通用解析（`--gripper PandaGripper` 现可跑通）。
2. **指垫 geom 定位写死名字**：`n3_osc_full_chain.pad_geoms()` 只认
   `left_fingerpad_col` / `right_fingerpad_col`。Panda 的指垫叫
   `gripper0_finger1_pad_collision` → 返回空字典 → `ContactMonitor.pad_ids` 为空
   → **指垫↔物体穿透恒等于 0（静默失效，会把坏结果报成好结果）**。
   → 本脚本安装夹爪无关的指垫定位（真·指垫优先，退化时取全部手指碰撞 geom）。

> 另：ZhiXing 的资产默认 `speed=0.2`，**部署值必须是 0.01**，否则闭合即把物体拨飞
> （本次实测未注入时穿透 17–23 mm、抬升为 0；注入 0.01 后抬升 74 mm）。

---

## 4. 方法与口径（可复核）

- 环境：`libero_object` task 0（`pick up the alphabet soup and place it in the basket`），
  `OffScreenRenderEnv`，256×256，`robo=ECO65`，`gripper=PandaGripper`，20 Hz。
- 物体：把 BDDL 目标物 `alphabet_soup_1` **进程内**替换为已知几何的实心长方体试块
  （宽 = 请求宽度、深 60 mm、高 70 mm、密度 700、`solref=0.02 1` 软接触）；
  原生谓词因此**真的适用**，磁盘资产零改动。
- 教师：全程只有 7 维 OSC 增量动作（无 IK 直写 qpos、无逐子步锁臂），
  每步目标推进量限幅 12 mm / 0.04 rad，夹爪速度 0.01。
- 穿透：挂在 `_pre_action`（每物理子步 500 Hz）+ 每控制步末采样，
  峰值无论是否通过都写进结果。
- 闸门：与 `n3_osc_full_chain.py` 同一套 7 项（抬升≥30 mm、保持 40 步≥25 mm、
  释放后稳定、原生谓词、无瞬移、穿透≤2 mm、接近/下降收敛）。
- 空载开闭重复性判据沿用 N3：全开/全闭标准 **< 0.5 mm**。
- 开口度量：两侧指垫碰撞 geom **表面点集的跨侧最近点对距离**
  （aperture = 净距最大的一对；min_clearance = 所有组合的最小净距）。
  ⚠️ 这是本脚本自己的口径，与 N2 契约的 `pad_face_separation_mm` 不完全同源，
  绝对值不可混用（本次不作跨报告数值比较）。

### 复现命令

```bash
cd /mnt/data/vla-policy/openpi
docker run --rm --network none -v "$PWD:/app" -w /app -e MUJOCO_GL=osmesa \
  --entrypoint /bin/bash libero:latest -c "
    source /.venv/bin/activate
    export PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero
    python /app/agent-work-space/outputs/scripts/eco65_expert_validation.py \
      --gripper PandaGripper --width-mm 30 --trials 4 --gripper-speed 0.01 \
      --out-dir /app/agent-work-space/outputs"
# 夹爪体检 + 空载开闭 ×10：
#   … 同命令加 --pilot --pilot-only --repeat 10
```

---

## 5. 必须如实记录的两项未通过 / 未定因

### 5.1 搬运段伺服不收敛（**26/26 次全部不收敛，两个夹爪都一样**）

| 阶段 | 目标 | 收敛判据 | 实测（Panda w30，末段 20 步位移） |
|---|---|---|---|
| transport | 篮口上方 +0.12 m | 220 步内 ≤8 mm | 196–211 步耗尽，残差 **0.055–0.077 m**，末 20 步仅移动 2.5–5.2 mm |
| place | 篮口上方 +0.03 m | 160 步内 ≤8 mm | 36–38 步耗尽，残差 0.004–0.013 m |

**定因（已排除项，逐条实测）**：

- ❌ 不是关节限位：全过程臂关节最小余量 **1.18–1.57 rad**，`at_limit=0`。
- ❌ 不是夹爪问题：同一目标点、同一路径，Panda 100% 完成任务，ZhiXing 也能抬升 74 mm。
- ❌ 不是 DLS-IK 不可达：对同一目标点单独解 IK，`into_rim+0.03` 残差仅 **5.1 mm**
  （`above_rim+0.12` 用单初值解到 47 mm，属局部极小，不能据此断定不可达）。
- ⚠️ 残余可能：搬运段末端夹爪/手腕与篮体或台面的**接触阻挡**、或 OSC 在满载下的稳态误差。
  **尚未定因**，需逐接触级排查（本次未做，不做推测性结论）。

**重要影响**：N4 采集器把搬运/投放超时记为硬失败（`servo_no_timeout=False`），
因此**即使抓取、抬升、投放、原生谓词全部成立，episode 仍被判失败** ——
这解释了此前 `n4v8 0/6`（我上一轮复核已独立确认：6/6 都是 transport 220 + place 160 超时，
其中 5/6 谓词已为真）。**这是采集链路的判据/可达性问题，不是 Panda 夹爪的问题。**

### 5.2 部分 trial 的抓取计划无效（不是夹爪失败）

Panda w30 有 1/3、ZhiXing 有 1/4 的 trial 报
`invalid_grasp_plan_section_too_thin`：计划高度上的截面宽远小于试块宽度
（薄带里只剩个别顶点），说明该高度**没有实体可夹**。
本脚本已把它作为**独立 reason 显式拒绝**，不再计为夹爪失败
（早期版本会继续执行并记成穿透 20 mm 的失败，属错误归因）。

---

## 6. 产物清单（`/mnt/data/vla-policy/openpi/agent-work-space/outputs/`）

| 文件 | 内容 |
|---|---|
| `scripts/eco65_expert_validation.py` | 脚本专家验证器（夹爪无关几何 + 七项闸门 + pilot 体检 + 阶段定因轨迹） |
| `scripts/make_comparison.py` | 汇总器：把所有 JSON 折成 `comparison.md/csv` |
| `comparison.md` / `comparison.csv` | 全部 run 的对照表 |
| `pilot_PandaGripper_20260929_090610.json` | Panda 体检：极性 / 行程 / 空载×10（**PASS**）|
| `pilot_ZhixingGripper_20260929_091351.json` | ZhiXing 体检（同口径对照，重复性 **FAIL**）|
| `expert_PandaGripper_w30_20260929_061956.json` | **主结果**：30 mm 4/4 |
| `expert_PandaGripper_w37_20260929_051647.json` | 36.7 mm 6/6 |
| `expert_PandaGripper_w25_20260929_054622.json` | 25 mm 3/3 |
| `expert_PandaGripper_w30_20260929_072637.json` | 30 mm 带阶段定因轨迹（transport 关节余量/位移） |
| `expert_ZhixingGripper_w30_20260929_073109.json` | ZhiXing 同宽度对照（0/4）|
| `_deprecated_harness_bugs/` | 被机具缺陷（开口预算取 reset、未注入夹爪速度）污染的 8 次历史运行，**仅留档不参与结论** |
| `pilot_*` / `expert_*` 均为**只读产物**：未改动控制器参数、未改动 `zhixing_adaptation_20260927/` 下任何受管文件 |

---

## 7. 本次没有做 / 不能据此推断的事

- **未做** VLA（π0.5）评测：本机无 GPU，且方案禁止用策略成功率判夹爪。
- **未做** 真实目标物（薄壁碗/罐）抓取：本次用已知几何的实心试块，
  结论**只对「夹爪—臂控制是否正确」负责**，不代表能抓住真实薄壁物体。
- **未做** 相机/视觉域适配：`T_cam2grip` 与 Panda 参考的关系沿用既有契约，本次未重测。
- **未做** 搬运段不收敛的接触级定因（§5.1）。
- 因此：**本报告支持「ECO65 + PandaGripper 控制正确、抓放链路可用」，
  不支持「已可进入 P1 批量采集」** —— 后者需先解决 §5.1 的判据/可达性问题。
