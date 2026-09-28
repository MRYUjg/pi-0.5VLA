# ECO65 初始条件对齐（IK 对齐末端位姿 + 相机↔夹持点变换实测）

| 项目 | 内容 |
| --- | --- |
| 日期 | 2026-09-27 |
| 目标 | 把 ECO65 + PandaGripper 的**初始条件**对齐到 π0.5 训练分布（Panda + PandaGripper） |
| 方案 | **固定物体位姿**，用 **IK 对齐 grip_site 的完整 6D 位姿**（位置 + 朝向），并**实测**相机↔夹持点变换 |
| 硬约束 | **不照搬 Panda 的 `right_hand` 参数** —— 那是最后一级连杆坐标系，改它会同时改变夹爪安装几何与运动学链 |
| 结果 | 初始末端位姿差 **108.69 mm / 6.01° → 0.001 mm / 0.0003°**；腕部画面像素差 **9.5% → 1.4%**；评测 **6/10 → 9/10** |

---

## 1. 目录内容

```text
initial_pose_alignment_20260927/
├── README.md                 # 本文档
├── align_initial_pose.py     # 主工具：measure / solve / apply / verify
├── run.sh                    # 在 libero 容器里运行的包装脚本
└── results/
    ├── initial_condition.json   # 对齐前：两台机器人的初始条件实测
    ├── ik_solution.json         # IK 解与残差、关节余量
    ├── verify.json              # 对齐后复测
    ├── wrist_*.png              # 腕部视角渲染（对齐前 / 对齐后）
    └── report.txt
```

## 2. 方法

### 2.1 固定物体位姿

以 **Panda 官方 init state**（`libero_spatial` task0, init_idx=0）为唯一场景基准：
读出所有**非机器人关节**（名字不以 `robot0_` / `gripper0_` 开头）的 qpos，按关节名抄进任意场景的
`data.set_joint_qpos()`，再 `sim.forward()`。这样 ECO65 与 Panda 看到的是**完全相同的物体摆放**。

> 之所以能按名字抄：两个模型的物体来自同一份 BDDL/arena，物体关节名一致；机器人关节数不同，
> 所以只同步物体、不碰机器人。

### 2.2 测量项

对每台机器人记录：

- `grip_site` 世界位姿（位置 + 旋转矩阵 + 四元数）—— robosuite 的末端参考点
- `eye_in_hand` 相机世界位姿
- **`T_cam2grip`**（夹持点表达在相机系）与 **`T_grip2cam`**（相机表达在夹持点系）
- `right_hand` 世界位姿，以及 `T_hand2grip` / `T_hand2cam`（用于判断"到底是安装几何不同还是末端位姿不同"）
- 腕部渲染图（256×256，robosuite obs 约定）
- 相机到各物体的距离（尺度代理量）

### 2.3 IK

阻尼最小二乘（DLS），目标为 Panda 初始 `grip_site` 的完整 6D 位姿：

```text
J = [Jp(site); Jr(site)]  ∈ R^{6×6}     只取 6 个机械臂关节
dq = Jᵀ (J Jᵀ + λ²I)^{-1} e              e = [Δpos; Δrot(轴角)]
q ← clip(q + α·dq, 关节下限, 关节上限)
```

- 用 MuJoCo 原生 `mj_kinematics` / `mj_comPos` / `mj_jacSite`，直接在**真实 LIBERO 场景**里求解，
  因此天然计入 mount、基座偏移、夹爪链路
- **多初值**（当前 init_qpos + 40 组扰动）避免落在坏分支；按"误差小 + 关节离限位远"选解
- 结束条件与判据分开：收敛容差 1e-5，验收容差 0.5 mm / 0.2°

### 2.4 落地与回滚

只改两处：

| 文件 | 改动 | 备份 |
| --- | --- | --- |
| `third_party/libero/.../envs/robots/mounted_eco65.py` | `init_qpos` → IK 解（LIBERO 桌面任务走这个类） | `*.pre_init_align.bak` |
| `robosuite/models/robots/manipulators/eco65_robot.py` | 同上（保持两个入口一致） | `*.pre_init_align.bak` |

`--align-camera` 是**条件性**的：只有实测 `T_grip2cam` 与 Panda 有实质差异时，才会去改
`eye_in_hand` 相机的 local `pos/quat`（**仍然不动 `right_hand`**）。本次实测判定**无需修改**。

---

## 3. 实测结果

### 3.1 对齐前

```text
### Panda + PandaGripper
  grip_site 位置 : [-0.2085, 0.0, 1.1733]
  grip_site 姿态 : quat(wxyz) [-0.0201, 0.7066, 0.7070, -0.0201]
  相机位置       : [-0.1530, 0.0, 1.2673]
  T_cam2grip = [[-1, 0, 0, 0], [0, 1, 0, -0.05], [0, 0, -1, -0.097], [0,0,0,1]]

### ECO65 + PandaGripper
  grip_site 位置 : [-0.1647, 0.0478, 1.0860]
  grip_site 姿态 : quat(wxyz) [0.0159, 0.7308, 0.6815, -0.0348]
  相机位置       : [-0.1120, 0.0511, 1.1816]
  T_cam2grip = [[-1, 0, 0, 0], [0, 1, 0, -0.05], [0, 0, -1, -0.097], [0,0,0,1]]

grip_site 位姿差  : 108.69 mm / 6.01 deg
T_cam2grip 平移差 : 0.000 mm      ← 完全相同
```

**关键事实：`T_cam2grip` 两台机器人逐元素相同（平移差 0.000 mm、旋转差 0.0000°）。**

原因（已核对代码）：两台机器人的 `eye_in_hand` 相机 local 参数完全相同
（`pos="0.05 0 0" quat="0 0.707108 0.707108 0" fovy=75`），且都挂在各自的 `right_hand` 下、
装的是**同一个 PandaGripper**（`add_gripper` 用 `merge(gripper, merge_body="right_hand")`，
保留夹爪根 body 自身的 `pos/quat`）。相机与夹持点都是 `right_hand` 的子节点，它们的**相对变换
与 `right_hand` 自身怎么定义无关**。

> 这正好印证了任务约束：**不该去照搬 Panda 的 `right_hand`**（`pos="0 0 0.1065"`,
> `quat="0.924 0 0 -0.383"`）。那只会改变夹爪安装几何，而对"相机看到的夹持点相对关系"毫无帮助。

所以真正需要对齐的只有 **末端位姿**：把 grip_site 摆到与 Panda 一致，腕部相机自然也就一致了。

### 3.2 IK 解

```text
目标(Panda grip_site): pos = [-0.2085, 0.0, 1.1733]
                       quat(wxyz) = [-0.0201, 0.7066, 0.7070, -0.0201]
旧 init_qpos = [ 0.0,     -0.3,     -1.6,      0.3,      1.5,      -1.5    ]
新 init_qpos = [-0.128227, -0.080175, -1.482052, -0.064915, 1.563529, -1.699308]

41 个初值全部收敛，残差 ≤ 0.009 mm / 0.0005 deg
最小关节限位余量 = 1.310 rad（joint_3），无关节贴边
```

### 3.3 对齐后复测

```text
初始 grip_site 位姿差异 : 位置 0.001 mm   姿态 0.0003 deg
                          （对齐前 108.694 mm / 6.0071 deg）
T_cam2grip 差异         : 平移 0.000 mm   旋转 0.0000 deg
腕部画面平均像素差       : 3.63/255 (1.4%)
                          （对齐前 24.30/255 = 9.5%）
```

腕部画面对比（`results/wrist_Panda_PandaGripper.png` vs `results/wrist_verify_ECO65.png`）：
物体摆位、尺度、夹爪在画面中的位置与占比均已一致。

### 3.4 控制器复检

换了初始姿态后重跑控制器冒烟，跟踪仍然正常：

```text
ECO65_pdon_hold:     pos=0.000mm ori=0.000deg  pass=True
ECO65_pdon_pos_z_plus: pos=1.080mm ori=0.226deg pass=True
ECO65_pdoff_hold:    pos=0.000mm ori=0.000deg  pass=True
ECO65_pdoff_pos_z_plus: pos=0.039mm ori=0.014deg pass=True
```

---

## 4. 端到端效果

| 阶段 | ECO65 + PandaGripper（`libero_spatial` 10 任务 ×1，seed 7，随机初始化） |
| --- | ---: |
| 原始 | 0/10 |
| + 基座自碰撞修复 | 6/10（60%） |
| **+ 初始位姿对齐** | **9/10（90%）** |
| Panda + PandaGripper 参考 | 10/10 |

H 格逐任务（对齐后）：

| 结果 | 任务 | 步数 |
| --- | --- | ---: |
| ✅ | between the plate and the ramekin | 120 |
| ✅ | next to the ramekin | 129 |
| ✅ | from table center | 103 |
| ✅ | on the cookie box | 85 |
| ✅ | **in the top drawer of the wooden cabinet**（修复自碰撞后仍失败） | 201 |
| ✅ | on the ramekin | 149 |
| ✅ | **next to the cookie box**（修复自碰撞后仍失败） | 113 |
| ✅ | **on the stove**（修复自碰撞后仍失败） | 128 |
| ✅ | next to the plate | 106 |
| ❌ | on the wooden cabinet | 230 |

也就是**修复自碰撞后剩余的 4 个失败任务里有 3 个被初始位姿对齐救回**。残余 1 个失败，
以及 90% → 100% 的差距，属于仍存在的分布差异（机械臂外观、6-DOF 非冗余、动力学）或随机性。

---

## 5. 复现命令

```bash
cd /mnt/data/vla-policy/openpi
D=agent-work-space/initial_pose_alignment_20260927

bash $D/run.sh measure --gripper PandaGripper          # 对齐前实测
bash $D/run.sh solve   --gripper PandaGripper --multistart 40
bash $D/run.sh apply   --gripper PandaGripper          # 写入 init_qpos（自动备份）
bash $D/run.sh verify  --gripper PandaGripper          # 复测对比

# 端到端评测
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
  /bin/bash -c 'source /.venv/bin/activate && python /app/agent-work-space/scripts/eval_matrix.py \
    --robot ECO65 --gripper PandaGripper --tag H_eco65_panda_align \
    --task-suite-name libero_spatial --num-trials-per-task 1 --seed 7 \
    --out-root /app/agent-work-space/eval'
```

**回滚**：

```bash
cp third_party/libero/libero/libero/envs/robots/mounted_eco65.py.pre_init_align.bak \
   third_party/libero/libero/libero/envs/robots/mounted_eco65.py
cp robosuite/models/robots/manipulators/eco65_robot.py.pre_init_align.bak \
   robosuite/models/robots/manipulators/eco65_robot.py
```

---

## 6. 边界与后续

**已确认**
- 相机↔夹持点变换本来就一致（实测 0.000 mm / 0.0000°），无需改相机、更不该改 `right_hand`
- IK 可在真实场景中把 grip_site 的 6D 位姿对齐到 0.001 mm / 0.0003°，且关节远离限位
- 对齐后腕部画面差异从 9.5% 降到 1.4%，评测 6/10 → 9/10

**未覆盖**
- 只对齐了**初始**位姿；轨迹执行过程中的动力学差异（6-DOF vs 7-DOF、力矩上限、关节 PD 回拉）未处理
- `agentview`（第三人称）里 ECO65 与 Panda 的**手臂外观**仍然不同，这是本轮没有也无法通过初始位姿解决的
- 仍是每格 10 个 episode；90% vs 100% 需要扩样才有统计意义
- 物体位姿取自 **task0 的 init_idx=0**；其他任务/初始状态的物体分布可能不同，但机器人初始姿态与之无关

**建议下一步**
1. 扩样（`--num-trials-per-task 10~50`）复测 H 配置，给出置信区间
2. 量化 `agentview` 外观差异的残余影响（例如：把 agentview 的手臂区域做遮挡/替换实验）
3. 若仍要追求接近 100%，用 ECO65 + PandaGripper 自己的数据微调（复现方案 M5）
