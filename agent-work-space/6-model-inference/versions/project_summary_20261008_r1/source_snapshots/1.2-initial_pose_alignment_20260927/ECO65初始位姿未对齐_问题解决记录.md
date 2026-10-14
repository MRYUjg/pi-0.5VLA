# ECO65 初始末端位姿未对齐：问题解决记录

| 项目 | 内容 |
|---|---|
| 记录日期 | 2026-09-27 |
| 适配方案 | ECO65 + PandaGripper |
| 验证环境 | 服务器 galaxy-mobile，`libero` / `openpi_server` Docker 镜像，MuJoCo 3.2.3 + robosuite 1.4.1 + LIBERO |
| 工作目录 | `/mnt/data/vla-policy/openpi/agent-work-space/initial_pose_alignment_20260927` |
| 问题范围 | 基座自碰撞修复后，ECO65 初始末端位姿与训练分布不一致，仍有 4/10 任务失败 |
| 当前状态 | **原因已定位，IK 对齐已落地，端到端复评完成（6/10 → 9/10）；扩样统计与外观差异对齐尚未完成** |
| 前置记录 | [ECO65 基座自碰撞问题解决记录](../controller_validation_20260927/ECO65基座自碰撞_问题解决记录.md) |

---

## 1. 问题背景

前一轮修掉「基座—第一连杆持续自碰撞」后，同一套 `pi05_libero` 权重、同一评测协议下
ECO65 + PandaGripper 从 **0/10** 升到 **6/10**，但仍明显低于 Panda + PandaGripper 的 **10/10**。

| 配置 | 修复自碰撞前 | 修复自碰撞后 |
|---|---:|---:|
| ECO65 + PandaGripper | 0/10 | 6/10 |
| Panda + PandaGripper | 10/10 | 10/10 |

此时有两条互相竞争的假设：

1. **观测分布差异**：ECO65 的第三人称外观、腕部视角尺度、初始末端位姿与 Panda 不同；
2. **安装/相机几何差异**：ECO65 的 `right_hand` 参数与 Panda 不同（`pos="0 0 -0.02"`、
   `quat="1 0 0 0"` vs Panda 的 `pos="0 0 0.1065"`、`quat="0.924 0 0 -0.383"`），
   可能连相机相对夹持点的关系都不一致。

如果是第 2 条，就需要改安装几何；如果是第 1 条，则应只对齐运动学/观测。

**硬约束**：不允许直接把 Panda 的 `right_hand` 参数搬到 ECO65 —— 那是最后一级连杆坐标系，
改它会**同时改变夹爪安装几何与运动学链**，属于换机器人而不是对齐初始条件。

---

## 2. 如何定位

### 2.1 固定物体位姿，建立可比场景

初始条件对比必须在同一物体摆放上进行。做法：以 **Panda 官方 init state**（`libero_spatial` task0,
init_idx=0）为唯一基准，把所有**非机器人关节**（名字不以 `robot0_` / `gripper0_` 开头）的 qpos
按关节名抄进任意场景的 `data.set_joint_qpos()`，再 `sim.forward()`。

之所以能按名字抄：两个模型加载同一个 BDDL/arena，物体关节名一致；只有机器人关节数不同，
所以只同步物体、不碰机器人。

### 2.2 实测两台机器人的初始条件

在同一场景、同一物体摆放下，对每台机器人记录：`grip_site` 世界位姿、`eye_in_hand` 相机世界位姿、
两者的相对变换 `T_cam2grip`（夹持点在相机系）与 `T_grip2cam`、`right_hand` 世界位姿及其到夹持点/相机的变换。

对齐前实测：

```text
### Panda + PandaGripper
  grip_site 位置 : [-0.2085, 0.0, 1.1733]
  grip_site 姿态 : quat(wxyz) [-0.0201, 0.7066, 0.7070, -0.0201]
  相机位置       : [-0.1530, 0.0, 1.2673]

### ECO65 + PandaGripper
  grip_site 位置 : [-0.1647, 0.0478, 1.0860]
  grip_site 姿态 : quat(wxyz) [0.0159, 0.7308, 0.6815, -0.0348]
  相机位置       : [-0.1120, 0.0511, 1.1816]

grip_site 位姿差  : 108.69 mm / 6.01 deg
T_cam2grip 平移差 : 0.000 mm        ← 逐元素相同
```

`T_cam2grip` 两边完全一致：

```text
T_cam2grip = [[-1, 0, 0,  0    ],
              [ 0, 1, 0, -0.05 ],
              [ 0, 0,-1, -0.097],
              [ 0, 0, 0,  1    ]]
```

**结论：假设 2 不成立。** 差异只存在于末端位姿。

### 2.3 为什么相机↔夹持点必然一致（代码核对）

| 环节 | ECO65 | Panda | 是否相同 |
|---|---|---|---|
| `eye_in_hand` local 参数 | `pos="0.05 0 0"` `quat="0 0.707108 0.707108 0"` `fovy=75` | 同左 | **相同** |
| 相机挂载点 | `right_hand` 子节点 | `right_hand` 子节点 | 同结构 |
| 夹爪 | PandaGripper | PandaGripper | **同一资产** |
| 夹爪装配方式 | `add_gripper()` → `merge(gripper, merge_body="right_hand")`，保留夹爪根 body 自身 `pos/quat` | 同左 | **相同** |
| `right_hand` 自身定义 | `pos="0 0 -0.02"` `quat="1 0 0 0"` | `pos="0 0 0.1065"` `quat="0.924 0 0 -0.383"` | **不同（但无关）** |

相机与夹持点**都是 `right_hand` 的子节点**，二者的相对变换由"相机 local 参数"×"夹爪链路"决定，
与 `right_hand` 自身在连杆上是怎么定义的无关。因此：

- 照搬 Panda 的 `right_hand` 只会改变夹爪安装几何，对相机↔夹持点**没有任何帮助**；
- 需要对齐的只有**末端位姿**——把 `grip_site` 摆到与 Panda 一致，腕部相机自然一同对齐。

---

## 3. 解决方案

### 3.1 IK 对齐完整末端位姿

目标：Panda 初始 `grip_site` 的**完整 6D 位姿**（位置 + 朝向），
`pos=[-0.2085, 0, 1.1733]`，`quat(wxyz)=[-0.0201, 0.7066, 0.7070, -0.0201]`。

方法：阻尼最小二乘（DLS），在**真实 LIBERO 场景**中求解，天然计入 mount、基座偏移与夹爪链路。

```text
J = [Jp(site); Jr(site)] ∈ R^{6×6}      只取 6 个机械臂关节
dq = Jᵀ (J Jᵀ + λ²I)^{-1} e              e = [Δpos; Δrot(轴角)]
q ← clip(q + α·dq, 关节下限, 关节上限)
```

- 用 MuJoCo 原生 `mj_kinematics` / `mj_comPos` / `mj_jacSite`，不依赖渲染与动力学
- **多初值**：当前 `init_qpos` + 40 组扰动，避免落在坏分支
- 选解准则：误差小 **且** 关节离限位远；收敛容差 1e-5，验收容差 0.5 mm / 0.2°

结果（41 个初值全部收敛）：

```text
旧 init_qpos = [ 0.0,       -0.3,      -1.6,       0.3,       1.5,       -1.5     ]
新 init_qpos = [-0.128227, -0.080175, -1.482052, -0.064915,  1.563529,  -1.699308]

残差          ≤ 0.009 mm / 0.0005 deg
最小关节余量   1.310 rad（joint_3），无关节贴边
```

### 3.2 落地范围（严格限定）

| 文件 | 改动 | 备份 |
|---|---|---|
| `third_party/libero/libero/libero/envs/robots/mounted_eco65.py` | `init_qpos` → IK 解（LIBERO 桌面任务走这个类） | `*.pre_init_align.bak` |
| `robosuite/models/robots/manipulators/eco65_robot.py` | 同上（保持两个入口一致） | `*.pre_init_align.bak` |

**未改动**：`right_hand` 的 `pos/quat`、`eye_in_hand` 相机参数、PandaGripper 资产、mount、控制器源码。

工具内置条件性相机修正（`apply --align-camera`）：**仅当**实测 `T_grip2cam` 与 Panda 有实质差异时，
才会修改 `eye_in_hand` 的 local 参数（仍然不动 `right_hand`）。本次实测判定 **无需修改**，该分支未触发。

---

## 4. 解决结果

### 4.1 初始条件对齐精度

| 指标 | 对齐前 | 对齐后 |
|---|---:|---:|
| `grip_site` 位置差 | 108.694 mm | **0.001 mm** |
| `grip_site` 姿态差 | 6.0071 deg | **0.0003 deg** |
| `T_cam2grip` 平移差 | 0.000 mm | 0.000 mm（本来就一致） |
| `T_cam2grip` 旋转差 | 0.0000 deg | 0.0000 deg |
| **腕部画面平均像素差** | 24.30/255（9.5%） | **3.63/255（1.4%）** |

腕部渲染图（`results/wrist_Panda_PandaGripper.png` vs `results/wrist_verify_ECO65.png`）：
物体摆位、尺度、夹爪在画面中的位置与占比均已一致。

### 4.2 控制器复检（换了初始姿态后）

```text
ECO65_pdon_hold:        pos=0.000mm ori=0.000deg  pass=True
ECO65_pdon_pos_z_plus:  pos=1.080mm ori=0.226deg  pass=True
ECO65_pdoff_hold:       pos=0.000mm ori=0.000deg  pass=True
ECO65_pdoff_pos_z_plus: pos=0.039mm ori=0.014deg  pass=True
```

### 4.3 端到端评测

| 阶段 | ECO65 + PandaGripper（`libero_spatial` 10 任务 ×1，seed 7，随机初始化） |
|---|---:|
| 原始 | 0/10 |
| + 基座自碰撞修复 | 6/10（60%） |
| **+ 初始位姿对齐** | **9/10（90%）** |
| Panda + PandaGripper 参考 | 10/10 |

对齐后逐任务：

| 结果 | 任务 | 步数 |
|---|---|---:|
| ✅ | between the plate and the ramekin | 120 |
| ✅ | next to the ramekin | 129 |
| ✅ | from table center | 103 |
| ✅ | on the cookie box | 85 |
| ✅ | **in the top drawer of the wooden cabinet**（自碰撞修复后仍失败） | 201 |
| ✅ | on the ramekin | 149 |
| ✅ | **next to the cookie box**（自碰撞修复后仍失败） | 113 |
| ✅ | **on the stove**（自碰撞修复后仍失败） | 128 |
| ✅ | next to the plate | 106 |
| ❌ | on the wooden cabinet | 230 |

**自碰撞修复后残余的 4 个失败任务中有 3 个被初始位姿对齐救回。**

---

## 5. 当前是否已经解决

| 阶段 | 状态 |
|---|---|
| 发现初始末端位姿与训练分布不一致 | 已完成 |
| 固定物体位姿、建立跨机器人可比场景 | 已完成 |
| 实测相机↔夹持点变换 | 已完成（结论：本来就一致，无需改动） |
| 用 IK 对齐完整 6D 末端位姿 | 已完成，残差 0.001 mm / 0.0003 deg |
| 把对齐解写入正式 `init_qpos` | **已实施**（含备份） |
| 控制器复检 | 已完成（冒烟全过） |
| 端到端 π0.5 复评 | **已完成：6/10 → 9/10** |
| 扩样统计（每格 10 → 10~50 episode） | **未完成** |
| `agentview` 手臂外观差异对齐 | **未完成** |
| 用 ECO65 自己的数据微调 | **未完成** |

---

## 6. 证据边界

- 本轮只对齐**初始**位姿。轨迹执行过程中的动力学差异（6-DOF 非冗余 vs 7-DOF 冗余、力矩上限
  差异、`single_arm.py:252-259` 的关节 PD 回拉）未做任何处理。
- `agentview`（第三人称）里 ECO65 与 Panda 的**手臂外观仍然不同**，这不是初始位姿能解决的；
  腕部画面差异从 9.5% 降到 1.4% 但未归零，残余主要来自手臂本体渲染差异。
- 评测每格只有 **10 个 episode**（10 任务 ×1 次）。"0 → 6 → 9"方向明确、机制清楚，
  但 90% 与 100% 的差距**尚无统计意义**，正式指标需扩样重测。
- 物体位姿取自 **task0 / init_idx=0**。其他任务的物体分布可能不同，但**机器人初始姿态与之无关**
  （IK 目标只取决于 Panda 的初始关节角，不依赖物体）。
- 对齐只保证初始时刻的位姿一致；不保证 ECO65 在整条轨迹上都能复现 Panda 的末端轨迹。

---

## 7. 复现与证据索引

在服务器运行（使用已有 `libero` 镜像）：

```bash
cd /mnt/data/vla-policy/openpi
D=agent-work-space/initial_pose_alignment_20260927

bash $D/run.sh measure --gripper PandaGripper                       # 对齐前实测
bash $D/run.sh solve   --gripper PandaGripper --multistart 40       # IK 求解
bash $D/run.sh apply   --gripper PandaGripper                       # 写入 init_qpos（自动备份）
bash $D/run.sh verify  --gripper PandaGripper                       # 复测对比

# 端到端复评
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
  /bin/bash -c 'source /.venv/bin/activate && python /app/agent-work-space/scripts/eval_matrix.py \
    --robot ECO65 --gripper PandaGripper --tag H_eco65_panda_align \
    --task-suite-name libero_spatial --num-trials-per-task 1 --seed 7 \
    --out-root /app/agent-work-space/eval'
```

**回滚**：

```bash
cd /mnt/data/vla-policy/openpi
cp third_party/libero/libero/libero/envs/robots/mounted_eco65.py.pre_init_align.bak \
   third_party/libero/libero/libero/envs/robots/mounted_eco65.py
cp robosuite/models/robots/manipulators/eco65_robot.py.pre_init_align.bak \
   robosuite/models/robots/manipulators/eco65_robot.py
```

| 文件 | 用途 |
|---|---|
| [README.md](README.md) | 方法说明与结果汇总 |
| [align_initial_pose.py](align_initial_pose.py) | 主工具：measure / solve / apply / verify |
| [run.sh](run.sh) | libero 容器运行包装 |
| [results/initial_condition.json](results/initial_condition.json) | 对齐前两台机器人的初始条件实测 |
| [results/ik_solution.json](results/ik_solution.json) | IK 解、残差、关节余量、目标位姿 |
| [results/verify.json](results/verify.json) | 对齐后复测数据 |
| [results/report.txt](results/report.txt) | verify 输出存档 |
| `results/wrist_*.png` | 腕部视角渲染（对齐前后对比） |
| `eval/H_eco65_panda_align/summary.json` | 对齐后端到端逐任务结果 |
| `logs/eval_H_eco65_panda_align.log` | 端到端评测日志 |

---

## 8. 后续建议

| 优先级 | 动作 | 目的 |
|---|---|---|
| ★★★ | 扩样复测 H 配置（`--num-trials-per-task 10~50`） | 给 90% 一个可写进报告的置信区间 |
| ★★☆ | 对齐 `agentview` 外观差异（遮挡/替换实验） | 量化残余 90% → 100% 中"外观"占多少 |
| ★★☆ | 量化关节 PD 回拉（`single_arm.py:252-259` 做成开关）的残余影响 | 非冗余臂的姿态拉扯 |
| ★☆☆ | 用 ECO65 + PandaGripper 自己的数据微调（复现方案 M5） | 根本解；前几项只是缩小分布偏移 |
| ★☆☆ | 若重新启用知行夹爪：先解决腕部外观 OOD | 夹爪路线的前置条件（目前 0/10） |
