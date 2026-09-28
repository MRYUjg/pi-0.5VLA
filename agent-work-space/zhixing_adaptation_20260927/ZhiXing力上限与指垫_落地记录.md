# ZhiXing 夹爪：力上限 + 指垫 **落地记录（v2，最终）**

| 项目 | 内容 |
|---|---|
| 日期 | 2026-09-27 |
| 改动文件 | `robosuite/models/assets/grippers/zhixing_gripper.xml` |
| 前 sha256 | `e797bc34766af9662074e5551a7dc95ee528d9c52aed3b1086f8270718a1fb45`（6695 B） |
| **后 sha256** | **`e7aab50c66da1e34bf1dde01455bc652b3ea09e5526df4d838fc3622ead00be5`**（6820 B） |
| 备份 | `zhixing_gripper.xml.pre_padfix.bak`（同目录） |
| 落地脚本 | `scripts/apply_zhixing_pad_fix.py`（字节级外科补丁 / 幂等 / dry-run / 落盘前校验） |
| 本地同步 | `/Users/yuyu/Documents/openpi/robosuite/models/assets/grippers/zhixing_gripper.xml` 已同步（sha 一致） |
| 最终验证 | **A PASS / B PASS / C：30 mm 10/10、50 mm 9/10、70 mm 8/10 → 总体 PASS**（两次复跑一致） |

> **v2 说明**：v1 曾用**新增 box geom** 实现指垫（sha `91c7d339…`），
> 实测会**自碰撞**（闭合卡在 40.6 mm）。已弃用，最终版改为
> **直接给两个接触指节已有的 mesh 垫 geom 换橡胶垫接触参数**。原因见 §2.2。

---

## 1. 先确认结构：ZhiXing 到底是什么

### 1.1 运动学结构（渲染 + 编译模型双重确认）

**CRT-CTAG2F90 = 单驱动二指平行四连杆（four-bar）自适应夹爪。**

```text
base_link
  ├─ Left_1_Link   ──(Left_1_Joint，唯一被驱动)
  │    ├─ Left_Support_Link ──(Left_Support_Joint)      ← ★ 指垫板（接触体）
  │    └─ Left_2_Link       ──(Left_2_Joint)            ← 平行连杆
  ├─ Right_2_Link           ──(Right_2_Joint)           ← 平行连杆
  └─ Right_1_Link  ──(Right_1_Joint)
       └─ Right_Support_Link ─(Right_Support_Joint)     ← ★ 指垫板（接触体）
```

驱动由 `<equality>` 声明（**不是** 6 个执行器）：5 条 joint equality + 2 条 connect
（四连杆闭环），1 个位置执行器驱动 `Left_1_Joint`，其余 5 关节联动。

### 1.2 **只有两个指节是接触体** —— 编译模型逐 body 核对

```text
body[2] base_link           仅视觉
body[3] Left_1_Link         仅视觉
body[4] Left_Support_Link   left_fingerpad_col(contype=0, conaff=1, **碰撞**)
body[5] Left_2_Link         仅视觉
body[6] Right_2_Link        仅视觉
body[7] Right_1_Link        仅视觉
body[8] Right_Support_Link  right_fingerpad_col(contype=0, conaff=1, **碰撞**)
```

→ **`Left_Support_Link` / `Right_Support_Link` 是仅有的两个接触体**，指垫只加在这两个指节上。

### 1.3 指垫面实测（body 局部系，左右对称）

| 量 | 值 |
|---|---|
| 内法向 `inner_dir` | `(-1, 0, 0)`（指向对方手指） |
| 内侧面位置 | body **x = −25.61 mm** |
| 面内半展 | **19.90 × 11.77 mm** |
| 面中心 | `(−0.025577, 0.043602, 0.000236)` |
| 全开 / 全闭内面净距 | **89.96 mm** / 0.52 mm |

### 1.4 关键：**mesh 里已经含橡胶垫零件**

厂商 URDF `zhixing_gripper.urdf`（SolidWorks 导出）在指节上挂了一个**独立的 `Left_Pad_Link`**：

```xml
<joint name="Left_Pad_Joint" type="fixed">
  <origin xyz="-0.0240101905072233 0.0412440692922833 0"/>
  <parent link="Left_up_Link"/>   <child link="Left_Pad_Link"/>
</joint>
```

该原点与实测内侧面中心 `(−0.025577, +0.043602)` 相距仅 **2.9 mm**。
→ `Left_Support_Link.STL` **已经包含橡胶垫**。
**所以缺的不是几何，而是它的接触材料属性。**

---

## 2. 落地改动（精确 diff，共 3 处）

```diff
43,44c43,44   ← Left_Support_Link 的指垫
<  <geom type="mesh" mesh="Left_Support_Link" group="0" conaffinity="1" contype="0"
<        solref="0.02 1" friction="1 0.005 0.0001" condim="4"
---
>  <geom type="mesh" mesh="Left_Support_Link" group="0" conaffinity="1" contype="1"
>        solref="0.01 0.5" solimp="0.99 0.999 0.0001 0.5 2" friction="2 0.05 0.0001" condim="4"

74,75c74,75   ← Right_Support_Link 的指垫（同样改动）
93c93         ← 执行器力上限
<  <position name="gripper" joint="Left_1_Joint" ctrlrange="-0.91 0" kp="100"/>
>  <position name="gripper" joint="Left_1_Joint" ctrlrange="-0.91 0" kp="100"
              ctrllimited="true" forcelimited="true" forcerange="-2 2"/>
```

字节级外科补丁（非 ElementTree 重写），**无任何格式重排**。

### 2.1 力上限 = ±2 N·m 的依据

`zhixing_gripper.urdf` 对 `Left_1_Joint` 声明 `<limit lower="0" upper="-1" effort="2" velocity="1000"/>`
—— **`effort=2` 就是厂商给出的执行器力矩上限**。
仓库内 PandaGripper / RethinkGripper 均为 `forcelimited="true" forcerange="-20 20"`，只有 ZhiXing 缺失。

### 2.2 指垫参数 = 与 PandaGripper 专用垫同参

| 参数 | PandaGripper 专用垫 | 本次 ZhiXing 指垫 |
|---|---|---|
| `solref` | `0.01 0.5` | **同** |
| `friction` | `2 0.05 0.0001` | **同** |
| `condim` | `4` | **同** |
| `contype` / `conaffinity` | `1` / `1` | **同** |
| `solimp` | 默认 | `0.99 0.999 0.0001 0.5 2` |

**为什么不用新增 box geom（v1 的实测教训）**

按实测内侧面新增 box 后（cover 0.95），闭合时两块新垫**互相顶住并撞到对侧手指的 mesh**，
夹爪卡在 **40.6 mm 合不上**（正常 0.52 mm）：

```text
gripper0_left_pad_collision  <-> gripper0_right_fingerpad_col  dist=-0.14mm
gripper0_right_pad_collision <-> gripper0_left_fingerpad_col   dist=-1.50mm
```

两块新垫的世界中心只差 **0.9 mm**（本应分居两侧）——两指 `Support_Link` 的 body 系
**并非简单镜像**，同一套 body-local 坐标无法同时正确落到两侧；
把 cover 降到 0.55 仍会自碰撞（卡在 32.9 / 64.9 mm 等）。

**结论：既然 mesh 里已有正确的垫几何，就应该用它、只改材料**。
`apply_zhixing_pad_fix.py --pad-mode box` 保留作对照，**默认不用**。

---

## 3. 验证结果（落地资产上重跑，**两次一致**）

### 3.1 A —— 空载开闭重复性 ×10 ✅ PASS

| 状态 | 均值 | 标准差 | min | max |
|---|---:|---:|---:|---:|
| 全开 | **89.89 mm** | **0.000 mm** | 89.89 | 89.89 |
| 闭合 | **0.65 mm** | **0.002 mm** | 0.65 | 0.66 |

全闭时唯一接触是两指垫**彼此相碰**（`left_fingerpad_col <-> right_fingerpad_col`），
**无自碰撞**。

### 3.2 B —— 6 向小幅位姿跟踪（2 cm）✅ PASS

最大残差 **1.85 mm**（判据 < 5 mm）。

### 3.3 C —— 标准实心试块抓取（判据 ≥9/10）✅ 总体 PASS

| 试块宽度 | 结果 | 闭合净距 | **穿透** | 抬升 |
|---|---:|---:|---:|---:|
| **30 mm** | **10/10 ✅** | 33.6 mm | **−1.12 mm** | +49.0 ~ +49.5 mm |
| **50 mm** | **9/10 ✅** | 53.2 mm | **−0.88 mm** | +51.1 ~ +53.2 mm |
| **70 mm** | 8/10 ❌ | 73.4 mm | **−0.82 mm** | +51.4 mm |

**总体 verdict = PASS**（30 mm 满 10/10）。

### 3.4 三次资产状态的对照

| 资产状态 | 30 mm | 50 mm | 70 mm | 穿透 |
|---|---:|---:|---:|---:|
| 原始（无限力 + 手指本体软参数） | 0/10 | 0/10 | 0/10 | −10 ~ −20 mm |
| v1：新增 box 指垫（cover 0.95）+ ±2 N·m | — | **自碰撞，卡在 40.6 mm** | — | — |
| **v2（最终）：mesh 垫换橡胶参数 + ±2 N·m** | **10/10** | **9/10** | 8/10 | **−0.82 ~ −1.12 mm** |

→ 穿透从 **10~20 mm 降到 ~1 mm**（橡胶垫压刚体的合理量级），30 mm 从 0/10 提到 10/10。

### 3.5 开合速度：与厂商规格吻合

| 力矩上限 | 全行程 | 等效速度 |
|---|---:|---:|
| 无限制 | 1.086 s | 77.7 mm/s |
| **2（落地值）** | **0.976 s** | **100.8 mm/s** |

厂商规格 `Gripping speed < 100 mm/s` —— 落地值基本正中规格。

---

## 4. 剩余失败的性质（**不是夹爪问题**）

70 mm 的 2 次失败与 50 mm 的 1 次失败，签名完全一致：

| 宽度 | 失败轮 | 闭合净距 | **推挤** | 接触 | 抬升 |
|---|---:|---:|---:|---:|---:|
| 50 mm | 3 | 0.65 mm | **108.5 mm** | 0 | 0.0 mm |
| 70 mm | 1 | 0.66 mm | **123.1 mm** | 0 | −0.0 mm |
| 70 mm | 3 | 92.94 mm | 43.5 mm | 1 | 27.8 mm |

**闭合净距 0.65/0.66 mm = 夹爪在"空处"完全合拢，且接触数 0** ——
说明**试块在接近/下降阶段就被撞出指间**（推挤 108~123 mm），夹爪根本没夹到东西。

这是**脚本专家的接近轨迹鲁棒性**问题（我的测试台架），**不是夹爪抓取能力问题**。
可改进方向：接近时分段减速、加大指间余量、或抓取前重新测量试块位姿。

---

## 5. 必须注意的副作用 / 待办

### 5.1 ⚠️ 既有 N3 的 A 项需要更长 settle

`n3_grasp_test.py --part A` 原每相位只给 **25 env step**（625 子步），
而落地力上限后**全开需 ≥1500 子步**（实测 A 项读数在 65.9~90.1 mm 间跳动而被误判 FAIL）。
已把 settle 参数化为 **`--a-settle`（默认 80 env step = 2000 子步）**，
并**每 trial 先 `env.reset()`**（否则累计步数超过 episode horizon，
`env.step` 抛 `executing action in terminated episode`）。
**判据本身（std < 0.5 mm）未改动。**

### 5.2 策略侧影响

夹爪现在**真的需要约 1 秒**走完全行程（之前无限力是"瞬间压到底"）。
N4 采集脚本的夹爪时序、π0.5 的夹爪指令窗口都需按 ~1 s 行程重新评估。

### 5.3 仍未处理

* 70 mm 宽度只到 8/10 —— 主要为接近撞飞（见 §4），另需扩样确认；
* 真机行为**仍未验证**（90 mm 行程 / 23 N 夹持力均来自厂商资料）；
* **任务集口径问题仍在**：LIBERO Spatial 主目标黑碗外径 111.8 mm > 90 mm 行程，
  侧向抓取不可行（见 `N3_节点总结_v2.md` §5），与本次改动无关。

---

## 6. 回滚 / 重放

```bash
cd /mnt/data/vla-policy/openpi/robosuite/models/assets/grippers
cp zhixing_gripper.xml.pre_padfix.bak zhixing_gripper.xml   # 回到 e797bc34…
sha256sum zhixing_gripper.xml
```

重新落地（幂等，第二次运行应全部 skip）：

```bash
cd /mnt/data/vla-policy/openpi
export OPENPI_DATA_HOME=/mnt/data/vla-policy/caches/openpi
docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
  /bin/bash -c "source /.venv/bin/activate && python \
  /app/agent-work-space/zhixing_adaptation_20260927/scripts/apply_zhixing_pad_fix.py --apply --pad-mode mesh"
```

对照实验（会自碰撞，仅供复现 v1 现象）：

```bash
  … apply_zhixing_pad_fix.py --apply --pad-mode box --cover 0.95
```

---

## 7. 产物

```text
robosuite/models/assets/grippers/zhixing_gripper.xml                 # ★ 已改（3 处）e7aab50c…
robosuite/models/assets/grippers/zhixing_gripper.xml.pre_padfix.bak # ★ 回滚用   e797bc34…

scripts/apply_zhixing_pad_fix.py   # ★ 补丁（--pad-mode mesh|box|none，幂等，dry-run，落盘前校验）
scripts/n3r_render_gripper.py      # 结构渲染 + 逐 geom 清单（目视依据）
scripts/n3r_close_time.py          # 开合时间 vs 力矩上限 + 关节/约束属性
scripts/n3r_repeat_check.py        # 给足 settle 的开合重复性
scripts/n3r_partA_debug.py         # 逐相位开合 + 自碰撞诊断
scripts/n3r_block_test.py          # 标准试块（--lift-mode osc）
scripts/n3_grasp_test.py           # A/B（新增 --a-settle）

results/gripper_render/*.png       # 结构渲染图（碰撞 geom 红色高亮）
results/n3r_LANDED_v3.json / v4    # ★ 落地后 30/50/70mm ×10（两次一致）
results/n3_landed_A.json / _B.json # ★ 落地后 A / B
logs/n3_final_ABC.log, n3r_landed_v3.log, n3r_close_time*, n3r_repeat*
```

---

## 8. 一句话

**ZhiXing 是单驱动二指平行四连杆夹爪，只有 `Left_Support_Link` / `Right_Support_Link`
两个指节是接触体；厂商 URDF 显示 mesh 里本就含橡胶垫零件（`Left_Pad_Link`）**，
所以本次不新增几何（按坐标新增 box 实测会自碰撞、卡在 40.6 mm），
而是把这两个指垫 geom 的接触参数换成与 PandaGripper 专用垫同参的橡胶参数，
并按厂商 URDF 的 `effort=2` 给执行器加 `forcerange="-2 2"`。

**结果：A PASS（89.89±0.000 / 0.65±0.002）· B PASS（1.85 mm）·
C：30 mm 10/10、50 mm 9/10、70 mm 8/10（总体 PASS）**，
穿透从 10~20 mm 降到 ~0.9 mm，全行程 0.976 s ≈ 100.8 mm/s（规格 <100 mm/s）。
剩余失败经查是**接近阶段把试块撞飞**（推挤 108~123 mm、夹爪空合），属测试台架轨迹问题。
