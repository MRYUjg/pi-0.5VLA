#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4：ECO65 + ZhiXing 的**脚本专家 + 数据采集器**（libero_object）。

为什么是脚本专家而不是 π0.5
---------------------------
π0.5 装 ZhiXing 后有**腕部视觉 OOD**（N1/N3 已证）：策略不产生抓取时序。
故示范必须由**状态驱动的脚本专家**产生；专家用真实物体位姿作特权信息，
**仅作教师**，评测（N7）不得读取。

o_t / a_t 配对（本采集器唯一的正确性核心）
-----------------------------------------
逐帧严格写成「**先取观测、再执行动作**」：

    obs_t   <- 上一帧 env.step 的返回（或 reset 的返回）
    frame_t <- 记录 obs_t（含双视角 RGB、qpos/qvel、grip_site）
    a_t     <- 本帧要执行的 7 维动作
    env.step(a_t) -> obs_{t+1}, reward, done
    frame_t.action <- a_t          # 动作**后**才写入标签，但图像是动作**前**的

即 `image[t]` 与 `actions[t]` 是同一时刻的「动作前观测 / 实际执行动作」，
不存在错位；也不使用任何私有接口去"事后补拍"。

采集契约（`任务方案.md` P1，逐条落实）
------------------------------------
20 Hz；双视角 RGB 各 180° 翻转（与 `examples/libero/main.py` 一致，记录
分辨率与翻转约定）；完整 qpos/qvel + 臂/夹爪索引；8 维 state
`[grip_site xyz, 轴角, Left_1_Joint, Right_1_Joint]`；episode 原子保存
（临时文件 + rename）；成功/失败分目录；失败不进训练集；保存初始完整
仿真状态、对象初态指纹、任务文字、种子、教师类型、动作尺度、夹爪配置、
**TCP 定义（eef 位置 + XML 哈希）**与代码指纹。

夹爪时序（N3 落地记录的硬约束）
------------------------------
力上限 ±2 N·m 后全行程需 ~1 s ≈ 20 env step：闭合必须「**闭合到停住**」，
抬升必须用**力驱动 OSC 闭环**（运动学瞬移传不了载荷，是台架假象）。
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

ROOT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927")
RES = 256
TARGET_IMG = 224
# --------------------------------------------------------------------------- #
# [R6] 教师伺服增益 ≠ 底层动作尺度
# --------------------------------------------------------------------------- #
# 这两个常量只是**脚本专家**把 grip_site 误差换算成归一化动作时用的除数，
# 属于教师内部实现，**不是** OSC 控制器的动作尺度。
# 审核记录 R6：旧元数据把二者混为一谈，写成 osc_pos_scale=0.02 /
# osc_rot_scale=0.05，而实际 OSC_POSE 控制器的 output_max 是
# [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]（满量程 0.05 m / 0.5 rad）。
# 现分开保存：教师增益见下方，真实动作尺度由 controller 读取后写入元数据
# （见 `controller_action_contract()`）。
TEACHER_GAIN_POS_M = 0.02
TEACHER_GAIN_ROT_RAD = 0.05
GRIP_ACT_OPEN, GRIP_ACT_CLOSE = -1.0, 1.0
MAX_FRAMES = 900                       # episode 步数上限（防越 horizon）
# 释放后稳定确认（审核 §4 步骤 5 + 复核 §6.2）
POST_RELEASE_DRIFT_MAX_MM = 5.0        # 沉降后窗口内的总位移上限
POST_RELEASE_MAX_STEP_DISP_MM = 5.0    # [§6.2] 窗口内**单步**位移上限
POST_RELEASE_MAX_SPEED_M_S = 0.05      # [§6.2] 窗口内速度上限（低速度要求）
POST_RELEASE_MIN_WINDOW_STEPS = 70     # [§6.2] 必须跑满 40 沉降 + 30 撤离
XML_PATH = pathlib.Path("/app/robosuite/models/assets/grippers/zhixing_gripper.xml")
ROBOT_XML = pathlib.Path("/app/robosuite/models/assets/robots/ECO65-B/robot.xml")
COLLECTOR = pathlib.Path(__file__)


# --------------------------------------------------------------------------- #
def sha16(p):
    try:
        return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
    except Exception:                                    # noqa: BLE001
        return None


def build(suite, task_id, robot="ECO65", gripper="ZhixingGripper", seed=1):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    s = benchmark.get_benchmark_dict()[suite]()
    task = s.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES,
                             camera_widths=RES, robots=[robot],
                             gripper_types=gripper)
    env.seed(seed)
    return env, task


def goal_objects(env):
    """从 BDDL goal_state 取 (目标物体, 目标区域)，如 ('alphabet_soup_1','basket_1_contain_region')。

    注意：`parsed_problem` 在**内层** BDDL domain（`env.env`）上，
    不在 `OffScreenRenderEnv` 包装层。
    """
    domain = getattr(env, "env", env)
    p = domain.parsed_problem
    # goal_state 的谓词是**小写**（实测 `[["in","alphabet_soup_1","basket_1_contain_region"]]`）
    for clause in p.get("goal_state", []):
        if len(clause) >= 3 and str(clause[0]).lower() in ("in", "on"):
            return clause[1], clause[2]
    # 兜底：BDDL 直接给出的感兴趣物体
    oi = p.get("obj_of_interest") or []
    if oi:
        return oi[0], None
    return None, None


def stem_candidates(name):
    """`basket_1_contain_region` → ['basket_1_contain_region','basket_1_contain','basket_1']。"""
    if name is None:
        return []
    parts = name.split("_")
    return ["_".join(parts[:i]) for i in range(len(parts), 1, -1)]


def find_body(sim, name):
    """按候选前缀找**带碰撞 geom** 的 body（goal 名如 `x_1`，body 为 `x_1_main`）。"""
    import mujoco

    m = sim.model._model
    for stem in stem_candidates(name):
        for b in range(m.nbody):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if nm == stem or nm.startswith(stem + "_"):
                if any(int(m.geom_bodyid[g]) == b and m.geom_contype[g] != 0
                       for g in range(m.ngeom)):
                    return b, nm
    return None, None


def bbox(sim, bid):
    pts = N.body_world_pts(sim, bid, collision_only=True)
    if pts is None or pts.shape[0] == 0:
        return None
    return pts.min(axis=0), pts.max(axis=0)


def left1(env):
    import mujoco

    sim, m, d, _ = N.ids(env)
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.endswith("Left_1_Joint"):
            return float(d.qpos[int(m.jnt_qposadr[j])])
    return float("nan")


def quat2axisangle(quat):
    """与 examples/libero/main.py 的 _quat2axisangle 同实现（输入 wxyz）。"""
    q = np.array(quat, dtype=np.float64).copy()
    q[3] = float(np.clip(q[3], -1.0, 1.0))
    den = float(np.sqrt(1.0 - q[3] * q[3]))
    if abs(den) < 1e-8:
        return np.zeros(3)
    return q[:3] / den * (2.0 * np.arccos(q[3]))


def _qmul(a, b):
    """Hamilton 乘积（wxyz 顺序）：R(a⊗b) = R(a)·R(b)。"""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- #
# [R4] state 的夹爪两维：按 **joint name** 定位，禁止硬编码索引
# --------------------------------------------------------------------------- #
# 任务方案 P1 批准的 state 契约是
#     [grip_site xyz(3), grip_site 轴角(3), Left_1_Joint, Right_1_Joint]
# 而 ZhiXing 的 6 个夹爪关节顺序为
#     [0] Left_1_Joint  [1] Left_Support_Joint  [2] Left_2_Joint
#     [3] Right_2_Joint [4] Right_1_Joint       [5] Right_Support_Joint
# 审核记录 R4：旧代码硬取 `gripper_qpos[[1, 5]]`，实际拿到的是两个 **Support**
# 关节，与声明的主动关节最大差 0.749–0.818 rad，且左侧联动符号相反。
STATE_GRIPPER_JOINT_NAMES = ("Left_1_Joint", "Right_1_Joint")


def resolve_state_gripper_indices(env):
    """按名字解析 state 里的两个夹爪关节，返回 (下标, 关节名)。

    要求每个名字**恰好命中一个**关节；命中 0 个或多个都直接报错，
    不允许静默退化（历史教训：静默取错索引会污染整个数据集契约）。
    """
    joints = list(env.env.robots[0].gripper.joints)
    idx, names = [], []
    for want in STATE_GRIPPER_JOINT_NAMES:
        hits = [i for i, j in enumerate(joints) if j == want or j.endswith("_" + want)]
        if len(hits) != 1:
            raise ValueError(
                "[R4] state 夹爪关节 %r 在 %s 中命中 %d 个（要求恰好 1 个）。"
                % (want, joints, len(hits)))
        idx.append(hits[0])
        names.append(joints[hits[0]])
    return idx, names


def controller_action_contract(env):
    """[R6] 从**实际控制器**读取真实动作尺度，与教师增益分开记录。"""
    c = env.env.robots[0].controller
    out = {
        "controller_type": type(c).__name__,
        "controller_output_max": [float(v) for v in np.atleast_1d(c.output_max)],
        "controller_input_min": [float(v) for v in np.atleast_1d(c.input_min)],
        "controller_input_max": [float(v) for v in np.atleast_1d(c.input_max)],
        "control_timestep": float(env.env.control_timestep),
    }
    om = out["controller_output_max"]
    out["action_contract"] = (
        "7 维 OSC_POSE 归一化动作 ∈[-1,1]：前 3 维平移，满量程 = output_max[0:3] = "
        "%s m；后 3 维旋转，满量程 = output_max[3:6] = %s rad；第 7 维夹爪 -1 开 / +1 合。"
        % (om[:3], om[3:6]))
    return out


# --------------------------------------------------------------------------- #
class Episode:
    """一次示范；负责 o_t/a_t 严格配对的逐帧记录。"""

    def __init__(self, env):
        self.env = env
        self.obs = None
        self.frames = []
        self.t = 0.0
        self.dt = float(env.env.control_timestep)
        self.done = False
        self.n = 0
        # [R4] state 的夹爪两维（按名字解析一次，全程复用）
        self.grip_idx, self.grip_names = resolve_state_gripper_indices(env)
        # [R2] 环境步计数；obs_raw_step 记录"当前缓存的观测产生于第几步"
        self.raw_steps = 0
        self.obs_raw_step = 0
        self.obs_origin = "init"

    def start(self):
        self.obs = self.env.reset()
        self.obs_raw_step = self.raw_steps
        self.obs_origin = "reset"
        return self.obs

    def refresh_obs(self, obs, origin):
        """[R2] 用**当前真实状态**的观测替换缓存。

        重摆物体与静置开爪都是**不记录**的准备动作，但它们已经推进了仿真。
        若不在开始记录前刷新缓存，第 0 帧就会把 reset 时刻的旧图像配上静置后的
        qpos（审核记录 R2 实测：旧图像落后 45 个控制步 = 2.25 s，外部视角平均
        像素差 4.84/255、腕部 12.06/255）。
        """
        if obs is None:
            raise ValueError("[R2] refresh_obs 收到 None：准备阶段最后一步没返回观测")
        self.obs = obs
        self.obs_raw_step = self.raw_steps
        self.obs_origin = origin
        return obs

    def act(self, action, phase):
        """记录 o_t(=self.obs) → 执行 a_t → 取 o_{t+1}。返回本帧。"""
        import mujoco

        sim, m, d, gid = N.ids(self.env)
        act = np.asarray(action, dtype=np.float64)
        # [R2] 本帧的两个步号必须一致：观测产生于第几步 vs 本帧在第几步被记录。
        # 不相等即说明记录的是过期观测 —— 直接失败，不允许静默写进数据集。
        step_index = self.raw_steps
        obs_step = self.obs_raw_step
        if obs_step != step_index:
            raise AssertionError(
                "[R2] 第 %d 帧记录的是过期观测：观测产生于第 %d 步，本帧却在第 %d 步。"
                "准备阶段（重摆/静置）结束后必须调用 Episode.refresh_obs()。"
                % (self.n, obs_step, step_index))
        # [R4] 用按名字解析出的主动关节下标，不再硬编码 [1, 5]
        gi = [int(self.env.env.robots[0]._ref_gripper_joint_pos_indexes[i])
              for i in self.grip_idx]
        gpos = np.array([d.qpos[j] for j in gi])
        fr = {
            "agentview": np.ascontiguousarray(
                self.obs["agentview_image"][::-1, ::-1]),
            "wrist": np.ascontiguousarray(
                self.obs["robot0_eye_in_hand_image"][::-1, ::-1]),
            "qpos": np.array(d.qpos, dtype=np.float64),
            "qvel": np.array(d.qvel, dtype=np.float64),
            "gripper_qpos": gpos,
            "eef_pos": np.array(self.obs["robot0_eef_pos"], dtype=np.float64),
            "eef_quat": np.array(self.obs["robot0_eef_quat"], dtype=np.float64),
            "grip_site": np.array([d.site_xpos[gid]], dtype=np.float64)[0],
            "phase": phase, "action": act.copy(),
            "done": bool(self.done), "reward": 0.0, "timestamp": self.t,
            "obs_step": obs_step, "step_index": step_index,
            "obs_origin": self.obs_origin,
        }
        obs, rew, done, info = self.env.step(act)
        self.raw_steps += 1
        self.obs = obs
        self.obs_raw_step = self.raw_steps
        self.obs_origin = "step"
        self.done = bool(done)
        fr["done"], fr["reward"] = self.done, float(rew)
        self.t += self.dt
        self.n += 1
        self.frames.append(fr)
        return fr

    def quiet(self, n, grip_cmd):
        """保持当前 grip_site 位姿静置 n 步，**不记录**。

        只用于 episode 开始前（reset 后、采样初始化之后）的稳定期 ——
        这段不属于示范本体；一旦开始记录，所有状态变化都必须能由
        记录的 7 维动作解释，否则 episode 不可回放。

        [R2] 返回最后一步的真实观测；调用方**必须**用 refresh_obs() 采纳它，
        否则下一帧会配上过期观测（act() 会直接断言失败）。
        """
        obs = quiet_hold(self.env, n, grip_cmd)
        self.raw_steps += int(n)
        self.refresh_obs(obs, "post_quiet")
        return obs

    def servo(self, target, R_hold, grip_cmd, phase, max_steps, tol=0.004,
              gain=1.0):
        """闭环伺服到 target（世界系），每步一帧。返回 (达成, 步数)。

        gain < 1 会**降低每步位移**（动作仍为合法的 7 维 OSC 增量），
        用于搬运/投放段：实测高速横移时夹持会因惯性滑落
        （N3 测得 μ_eff 仅约名义值一半，保持力余量本就有限）。
        """
        for k in range(max_steps):
            if self.done or self.n >= MAX_FRAMES:
                return False, k
            sim, m, d, gid = N.ids(self.env)
            p, R = N.gripsite(sim, m, d, gid)
            ap = np.clip(gain * (np.asarray(target) - p) / TEACHER_GAIN_POS_M,
                         -1.0, 1.0)
            ar = np.clip(N.rot_err_vec(R_hold, R) / TEACHER_GAIN_ROT_RAD, -1.0, 1.0)
            self.act(np.concatenate([ap, ar, [grip_cmd]]), phase)
            if float(np.linalg.norm(np.asarray(target) - p)) < tol:
                return True, k + 1
        return False, max_steps

    def arrays(self):
        def st(k, dt=None):
            a = np.stack([f[k] for f in self.frames])
            return a.astype(dt) if dt else a

        eef, quat, gq = st("eef_pos"), st("eef_quat"), st("gripper_qpos")
        aa = np.array([quat2axisangle(q) for q in quat])
        ph = sorted({f["phase"] for f in self.frames})
        return {
            "image": st("agentview", np.uint8),
            "wrist_image": st("wrist", np.uint8),
            "qpos": st("qpos"), "qvel": st("qvel"),
            "gripper_qpos": gq, "eef_pos": eef, "eef_quat": quat,
            "grip_site": st("grip_site"),
            "state": np.concatenate([eef, aa, gq], axis=1),
            "actions": st("action"),
            "timestamp": np.array([f["timestamp"] for f in self.frames]),
            # [R2] 逐帧保留步号，便于事后核验"图像不早于动作"（两者必须相等）
            "obs_step": np.array([f["obs_step"] for f in self.frames], dtype=np.int32),
            "step_index": np.array([f["step_index"] for f in self.frames], dtype=np.int32),
            "phase_id": np.array([ph.index(f["phase"]) for f in self.frames],
                                 dtype=np.int32),
            "done": np.array([f["done"] for f in self.frames], dtype=np.int8),
            "reward": np.array([f["reward"] for f in self.frames]),
            "_phases": ph,
        }


# --------------------------------------------------------------------------- #

_USED_INITS = []


def sample_object_init(env, bid, obj_name, bname, region_name, rng,
                       margin=0.005, min_sep=0.012, tries=80):
    """把目标物体重摆到其 BDDL 区域内的一个**随机位姿**，作为本 episode 的初始化。

    为什么必须显式做
    ----------------
    实测 `env.seed(s); env.reset()` 对 `libero_object` 的**目标物体**几乎不改变
    摆放（seed=8/9/10 的物体中心都是 (-0.119,-0.2435)，只有篮子 x 范围在变）。
    若直接采 5 条，就是**同一个初始化**的 5 次重复 —— 不满足方案
    「至少 5 条**不同初始化**的成功示范」，也无法覆盖状态分布。

    做法：读该区域的 BDDL `ranges`，在内部留 margin 均匀采样 xy 与偏航，
    直接写 free joint 后静置。LIBERO 自身摆放物体也是这套区域采样，
    故这不引入区域外的非法状态；采样位姿与指纹**逐 episode 记录**。
    """
    import mujoco

    domain = getattr(env, "env", env)
    pp = domain.parsed_problem
    # ★ 必须用**初始**区域，不是 goal 区域：
    #   goal_state 给的是 `basket_1_contain_region`（投放目标），
    #   物体初始摆放写在 `initial_state` 的 ["on", <obj>, <region>] 里。
    #   `initial_state` 用**对象名**（`alphabet_soup_1`），而 body 名带 `_main` 后缀
    #   （`alphabet_soup_1_main`）—— 必须按对象名匹配，否则永远匹配不上。
    init_region = None
    for clause in pp.get("initial_state", []):
        if len(clause) < 3 or str(clause[0]).lower() != "on":
            continue
        nm = str(clause[1])
        if nm == obj_name or nm == bname or bname.startswith(nm + "_"):
            init_region = clause[2]
            break
    region_name = init_region or region_name
    reg = (pp.get("regions", {}) or {}).get(region_name)
    if not reg or not reg.get("ranges"):
        return None
    r0 = reg["ranges"][0]
    x0, x1 = sorted((float(r0[0]), float(r0[2])))
    y0, y1 = sorted((float(r0[1]), float(r0[3])))
    # 拒绝采样：保证与本次已用过的初始化至少相距 min_sep。
    # 区域只有 50x50 mm，纯随机 5 条极易聚在一起（实测最小间距仅 1.0 mm），
    # 那样「不同初始化」名不副实。
    x = y = yaw = None
    for _ in range(tries):
        cx = float(rng.uniform(x0 + margin, x1 - margin))
        cy = float(rng.uniform(y0 + margin, y1 - margin))
        if all(np.hypot(cx - ux, cy - uy) >= min_sep for ux, uy in _USED_INITS):
            x, y, yaw = cx, cy, float(rng.uniform(-np.pi, np.pi))
            break
    if x is None:                       # 退让：放宽到最大间距者
        cx = float(rng.uniform(x0 + margin, x1 - margin))
        cy = float(rng.uniform(y0 + margin, y1 - margin))
        x, y, yaw = cx, cy, float(rng.uniform(-np.pi, np.pi))
    _USED_INITS.append((x, y))

    sim, m, d, _ = N.ids(env)
    if not apply_object_sample(env, bid, (x, y), yaw):
        return None
    return {"region": region_name, "xy": [round(x, 5), round(y, 5)],
            "yaw_rad": round(yaw, 5),
            "region_ranges": [x0, y0, x1, y1]}


def free_joint_adr(env, bid):
    """返回目标物体 free joint 的 qpos/dof 起始地址。"""
    import mujoco

    sim, m, d, _ = N.ids(env)
    for j in range(m.njnt):
        if int(m.jnt_bodyid[j]) == bid and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(m.jnt_qposadr[j]), int(m.jnt_dofadr[j])
    return None, None


def apply_object_sample(env, bid, xy, yaw):
    """把目标物体按**已记录的采样值**重摆（确定性；采集与回放共用同一函数）。

    回放必须走同一条代码路径，否则姿态合成（左乘世界 z 偏航）会与采集时
    不一致，回放就不可能逐位复现。
    """
    import mujoco

    a, adr = free_joint_adr(env, bid)
    if a is None:
        return False
    sim, m, d, _ = N.ids(env)
    z = float(d.qpos[a + 2])
    d.qpos[a], d.qpos[a + 1], d.qpos[a + 2] = float(xy[0]), float(xy[1]), z + 0.002
    q_old = np.array(d.qpos[a + 3:a + 7], dtype=float)
    q_yaw = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
    d.qpos[a + 3:a + 7] = _qmul(q_yaw, q_old)
    d.qvel[adr:adr + 6] = 0.0
    mujoco.mj_forward(m, d)
    return True


def quiet_hold(env, n, grip_cmd):
    """保持当前 grip_site 位姿静置 n 步（不记录）。采集与回放共用。

    [R2] **返回最后一步的真实观测**，供调用方刷新 episode 的观测缓存；
    不刷新就会让第 0 帧配上过期图像（审核记录 R2 的根因）。
    """
    sim, m, d, gid = N.ids(env)
    p0, R0 = N.gripsite(sim, m, d, gid)
    obs = None
    for _ in range(n):
        sim, m, d, gid = N.ids(env)
        p, R = N.gripsite(sim, m, d, gid)
        ap = np.clip((p0 - p) / TEACHER_GAIN_POS_M, -1.0, 1.0)
        ar = np.clip(N.rot_err_vec(R0, R) / TEACHER_GAIN_ROT_RAD, -1.0, 1.0)
        obs, _, _, _ = env.step(np.concatenate([ap, ar, [grip_cmd]]))
    return obs


def run_episode(env, args, seed, log):
    ep = Episode(env)
    env.seed(seed)
    obs_reset = ep.start()
    sim, m, d, gid = N.ids(env)

    obj_name, region_name = goal_objects(env)
    bid, bname = find_body(sim, obj_name)
    rid, rname = find_body(sim, region_name)
    if bid is None:
        raise RuntimeError("找不到抓取目标 %s" % obj_name)
    if rid is None:
        raise RuntimeError("找不到目标区域 %s" % region_name)
    log("  目标=%s  区域=%s" % (bname, rname))

    # ---------- 阶段 0（**不记录**）：重摆初始化 + 静置稳定 ----------
    # ★ 顺序至关重要：初始化重摆是对 free joint 的**直接写入**，无法用 7 维
    #   动作解释。若把它放在记录开始之后，episode 里就存在一处「无法由动作
    #   解释的状态跳变」，回放必然发散（实测 qpos 偏差 0.63~1.36）。
    #   故必须在捕获 init_qpos / 开始记录**之前**完成。
    init_sample = None
    if args.vary_init:
        rng = np.random.default_rng(100000 + seed)
        init_sample = sample_object_init(env, bid, obj_name, bname,
                                         region_name, rng)
        if init_sample is not None:
            log("  [init] 区域采样 xy=%s yaw=%.3f rad  区域=%s"
                % (init_sample["xy"], init_sample["yaw_rad"],
                   init_sample["region"]))
        else:
            log("  [!] 初始化采样失败——本批可能仍是同一初始化")
    PREP_HOLD = 45
    obs_prep = quiet_hold(env, PREP_HOLD, GRIP_ACT_OPEN)
    # [R2] 准备阶段的 45 步同样推进了仿真，必须计入步号；并**采纳最后一步的
    #      真实观测**作为第 0 帧的 o_t。旧实现丢弃了它，导致第 0 帧图像
    #      停留在 reset 时刻（落后 45 步 = 2.25 s，含目标重摆前的状态）。
    ep.raw_steps += PREP_HOLD
    ep.refresh_obs(obs_prep, "post_init_quiet_hold")
    # [R2] 首帧一致性检查：量化旧观测与真实当前观测的差异，留作证据。
    obs_staleness = {
        "prep_hold_steps": PREP_HOLD,
        "agentview_mean_abs_diff_255": float(np.mean(np.abs(
            obs_reset["agentview_image"].astype(np.int16)
            - obs_prep["agentview_image"].astype(np.int16)))),
        "wrist_mean_abs_diff_255": float(np.mean(np.abs(
            obs_reset["robot0_eye_in_hand_image"].astype(np.int16)
            - obs_prep["robot0_eye_in_hand_image"].astype(np.int16)))),
        "reset_obs_gripper_qpos": [round(float(v), 6) for v in np.atleast_1d(
            obs_reset["robot0_gripper_qpos"])],
        "fresh_obs_gripper_qpos": [round(float(v), 6) for v in np.atleast_1d(
            obs_prep["robot0_gripper_qpos"])],
        "adopted_obs_origin": ep.obs_origin,
        "adopted_obs_step": int(ep.obs_raw_step),
    }
    log("  [R2] 首帧观测已刷新：origin=%s step=%d  外部像素差=%.4f/255  腕部像素差=%.4f/255"
        % (ep.obs_origin, ep.obs_raw_step,
           obs_staleness["agentview_mean_abs_diff_255"],
           obs_staleness["wrist_mean_abs_diff_255"]))

    # ---------- episode 第 0 帧：此后所有状态变化都由记录的动作用产生 ----------
    sim, m, d, gid = N.ids(env)
    init_qpos = np.array(d.qpos, dtype=np.float64).copy()
    init_qvel = np.array(d.qvel, dtype=np.float64).copy()
    # ★ `qacc_warmstart` 是 MuJoCo 约束求解器的热启动状态，属**完整仿真状态**。
    #   不保存它，回放会因求解器起点不同而漂移（实测 qpos 偏差 ~0.24 rad，
    #   接触富集场景会被放大）。保存后回放可逐位复现。
    init_warm = np.array(d.qacc_warmstart, dtype=np.float64).copy()
    # [R3] 夹爪的**累积控制状态**。`format_action` 是累加式
    #   （current_action ← clip(current_action − speed·sign(a), −1, 1)），
    #   而 reset 后它是 0；本 episode 开始时已经过 PREP_HOLD 步开爪指令，
    #   实测已饱和到 +1（ZhiXing dof=1）。不保存并恢复它，回放第一步就会
    #   发出错误的夹爪命令（审核记录 R3：首步夹爪角误差 0.3154 rad vs
    #   0.00002 rad）。它属于「控制器状态」，与 qpos/qvel/warmstart 并列。
    init_grip_act = np.array(env.env.robots[0].gripper.current_action,
                             dtype=np.float64).copy()
    # [复核 §2.3.2] 除 qpos/qvel/warmstart 外，再保存**执行器控制量与外力缓存**，
    #   使"完整仿真+控制状态"可被回放逐位重建。缺少这些量时，回放的首个分歧帧
    #   往往出现在接触发生的瞬间，且偏差会被接触求解放大到数百毫米。
    init_ctrl = np.array(d.ctrl, dtype=np.float64).copy()
    init_qfrc_applied = np.array(d.qfrc_applied, dtype=np.float64).copy()
    init_xfrc_applied = np.array(d.xfrc_applied, dtype=np.float64).copy()
    ep.servo(N.gripsite(sim, m, d, gid)[0], N.gripsite(sim, m, d, gid)[1],
             GRIP_ACT_OPEN, "settle", 10, tol=1e-9)     # 记录少量零动作等待

    # [R2] 首帧验收：第 0 帧记录的观测必须与同帧仿真状态同步（act() 已强断言，
    #      这里再核一次并留证据）。obs_step 与 step_index 不等即说明首帧过期。
    _f0 = ep.frames[0]
    first_frame_sync = {
        "obs_step": int(_f0["obs_step"]),
        "step_index": int(_f0["step_index"]),
        "obs_origin": _f0["obs_origin"],
        "in_sync": bool(_f0["obs_step"] == _f0["step_index"]),
        "first_frame_qpos_matches_init": bool(
            np.allclose(np.asarray(_f0["qpos"]), init_qpos, atol=0.0)),
    }
    if not first_frame_sync["in_sync"]:
        raise AssertionError("[R2] 首帧观测与仿真状态不同步：%s" % first_frame_sync)
    log("  [R2] 首帧同步检查通过：obs_step=step_index=%d（origin=%s）"
        % (first_frame_sync["obs_step"], first_frame_sync["obs_origin"]))

    sim, m, d, gid = N.ids(env)
    ob, oh = bbox(sim, bid)
    obj_c0 = 0.5 * (ob + oh)
    log("  [settle 后] 物体中心=%s  z范围=[%.4f, %.4f]"
        % (np.round(obj_c0, 4).tolist(), float(ob[2]), float(oh[2])))
    obj_fp = hashlib.sha256(np.round(np.concatenate([ob, oh]), 6).tobytes()
                            ).hexdigest()[:16]
    rb, rh = bbox(sim, rid)
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()

    h = float(oh[2] - ob[2])
    z_g = float(min(ob[2] + max(0.028, min(0.5 * h, h - 0.006)), oh[2] - 0.004))
    pin = np.array([obj_c0[0], obj_c0[1], z_g])

    # 截面宽度：网格顶点可能很稀疏（如圆柱只有上下两圈顶点，中段没有样本），
    # 故**自适应加宽**取样带；仍无样本则退回整体 bbox（对凸体是保守估计）。
    pts = N.body_world_pts(sim, bid, collision_only=True)
    w_x = w_y = None
    for band in (0.006, 0.012, 0.02, 0.035):
        slab = pts[np.abs(pts[:, 2] - z_g) < band]
        if slab.shape[0] > 1:
            w_y = float(slab[:, 1].max() - slab[:, 1].min())
            w_x = float(slab[:, 0].max() - slab[:, 0].min())
            break
    if w_y is None:
        w_x, w_y = float(oh[0] - ob[0]), float(oh[1] - ob[1])
    rotated = False
    if w_y > 0.084 and w_x < w_y:
        R_ref = R_ref @ rotz(np.pi / 2)
        rotated = True
    log("  抓取高度 z=%.4f  截面宽 x=%.1f y=%.1f mm  转90=%s"
        % (z_g, w_x * 1000, w_y * 1000, rotated))

    def diag(tag):
        """grip_site / 夹爪净距 / 物体当前位置 —— 三者一起看才能判断抓空的原因。"""
        sim_, m_, d_, gid_ = N.ids(env)
        pp = np.array(d_.site_xpos[gid_], dtype=float)
        bb = bbox(sim_, bid)
        if bb is None:
            return "  [diag %s] 物体 bbox 丢失" % tag
        bc = 0.5 * (bb[0] + bb[1])
        import verify_tcp_fix as _V
        pf = _V.pad_face_pts(env)
        pad = "n/a"
        if pf is not None:
            fl, fr, sep = pf
            pad = ("Lz=%.4f Rz=%.4f sep=%.1fmm" % (fl[2], fr[2], sep * 1000))
        return ("  [diag %-8s] grip_site=%s  离抓取目标=%.1f mm  指垫 %s"
                "  夹爪净距=%.1f mm  Left_1=%.4f\n"
                "             物体中心=%s  物体z范围=[%.4f, %.4f]"
                "  |物体-grip_site|=%s"
                % (tag, np.round(pp, 4).tolist(),
                   float(np.linalg.norm(pp - pin) * 1000), pad,
                   N.jaw_face_separation_mm(env) or -1.0, left1(env),
                   np.round(bc, 4).tolist(), float(bb[0][2]), float(bb[1][2]),
                   np.round(bc - pp, 4).tolist()))

    lifted_mm, close_steps = 0.0, 0
    pe = {}
    # 释放后稳定确认的载体（审核 §4 步骤 5）：即使抛异常也要有定义
    pr = {"settle_steps": 0, "retreat_steps": 0,
          "object_center_at_release": None, "predicate_true_at_release": None}
    oe_release = None
    try:
        _, pe["approach"] = ep.servo(pin + np.array([0, 0, 0.08]), R_ref,
                                     GRIP_ACT_OPEN, "approach", 90)
        log(diag("approach"))
        _, pe["descend"] = ep.servo(pin, R_ref, GRIP_ACT_OPEN, "descend", 110,
                                    tol=0.003)
        log(diag("descend"))
        # 闭合到停住
        l1p, stable = None, 0
        for _ in range(args.close_max_steps):
            if ep.done or ep.n >= MAX_FRAMES:
                break
            sim, m, d, gid = N.ids(env)
            p, R = N.gripsite(sim, m, d, gid)
            ap = np.clip((pin - p) / TEACHER_GAIN_POS_M, -1, 1)
            ar = np.clip(N.rot_err_vec(R_ref, R) / TEACHER_GAIN_ROT_RAD, -1, 1)
            ep.act(np.concatenate([ap, ar, [GRIP_ACT_CLOSE]]), "close")
            close_steps += 1
            l1 = left1(env)
            if close_steps % 20 == 0 or close_steps <= 6:
                _s2, _m2, _d2, _g2 = N.ids(env)
                _bb = bbox(_s2, bid)
                _bc = 0.5 * (_bb[0] + _bb[1])
                _pf = __import__("verify_tcp_fix").pad_face_pts(env)
                _pd = ("L=(%.3f,%.3f,%.3f) R=(%.3f,%.3f,%.3f) sep=%.1f"
                       % (tuple(_pf[0]) + tuple(_pf[1]) + (_pf[2] * 1000,))
                       ) if _pf else "n/a"
                log("    [close %3d] Left_1=%.4f  物体中心=(%.3f,%.3f,%.3f)  %s"
                    % (close_steps, l1, _bc[0], _bc[1], _bc[2], _pd))
            if l1p is not None and abs(l1 - l1p) < 1e-4:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
            l1p = l1
        pe["close_steps"] = close_steps
        log("  闭合 %d 步后：" % close_steps + diag("close").strip())

        # 抬升 5 cm（力驱动 OSC）
        pe["lift_converged"], pe["lift_steps"] = ep.servo(
            pin + np.array([0, 0, 0.05]), R_ref, GRIP_ACT_CLOSE, "lift", 90,
            tol=0.005)
        sim2, _, _, _ = N.ids(env)
        oc = 0.5 * (lambda b: b[0] + b[1])(bbox(sim2, bid))
        lifted_mm = float((oc[2] - obj_c0[2]) * 1000)
        pe["lift_mm"] = lifted_mm
        log("  闭合 %d 步；抬升 %.1f mm" % (close_steps, lifted_mm))

        # 搬运 → 入篮 → 松开 → 撤退
        # ★ 篮子中心在 x≈0 附近，而机器人 base 在 x=-0.6 —— 中心点已在
        #   ECO65 可达边缘（实测 IK 残差 ~80 mm）。故投放在**朝机器人一侧的
        #   近边**，把所需伸展量减到最小；物体从近边落入篮内即可满足谓词。
        base = np.array(env.env.sim.data.body_xpos[
            env.env.sim.model.body_name2id("robot0_base")], dtype=float)
        #   侧向留 35% 余量（22% 时物体近边离篮壁仅 ~4 mm，下潜必刮篮口 —— 实测主因）
        frac = 0.35
        near_x = rb[0] + frac * (rh[0] - rb[0]) if base[0] < rb[0] else \
            rh[0] - frac * (rh[0] - rb[0])
        bx = np.array([near_x, 0.5 * (rb[1] + rh[1])])
        above = np.array([bx[0], bx[1], float(rh[2]) + 0.12])
        #   ★ 只在**篮口上方**释放，不下潜进篮：下潜时物体必须穿过篮口，
        #     稍有横向误差就会刮到篮沿被蹭掉（实测：抓取 10/10、投放仅 3/10）。
        #     从上释放让物体自由落入，篮口不再成为障碍。
        into = np.array([bx[0], bx[1], float(rh[2]) + 0.03])
        log("  [place] base=%s  篮 x范围=[%.3f, %.3f]  投放点=%s"
            % (np.round(base, 3).tolist(), rb[0], rh[0],
               np.round(into, 3).tolist()))
        # [复核 §6] 不再丢弃 servo 的收敛标志：超时必须留证并触发失败
        ok_t, k_t = ep.servo(above, R_ref, GRIP_ACT_CLOSE, "transport", 220,
                             tol=0.008, gain=0.55)
        pe["transport"] = [bool(ok_t), int(k_t)]
        pe["transport_converged"] = bool(ok_t)
        log(diag("transport"))
        sep_now = N.jaw_face_separation_mm(env) or 0.0
        pe["sep_at_transport_mm"] = sep_now
        if sep_now < 20.0:                       # 物体已掉：不再空跑投放
            log("  [!] 搬运中物体已脱落（净距 %.1f mm），提前结束" % sep_now)
            pe["lost_in_transport"] = True
        else:
            ok_p, k_p = ep.servo(into, R_ref, GRIP_ACT_CLOSE, "place", 160,
                                 tol=0.008, gain=0.55)
            pe["place"] = [bool(ok_p), int(k_p)]
            pe["place_converged"] = bool(ok_p)
            log(diag("place"))
        for _ in range(90):
            if ep.done or ep.n >= MAX_FRAMES:
                break
            sim, m, d, gid = N.ids(env)
            p, R = N.gripsite(sim, m, d, gid)
            ap = np.clip((into - p) / TEACHER_GAIN_POS_M, -1.0, 1.0)
            ar = np.clip(N.rot_err_vec(R_ref, R) / TEACHER_GAIN_ROT_RAD, -1.0, 1.0)
            ep.act(np.concatenate([ap, ar, [GRIP_ACT_OPEN]]), "release")

        # ---------- 释放后稳定确认（审核记录第 4 节步骤 5）----------
        # 审核指出："原生成功第一次成立即停止，不等于已经验证稳定放置后的持续成功"，
        # 且历史每条数据的最后一帧都是 release、**没有 retreat 帧**。
        # 根因：`servo()` 见到 `self.done` 会立即返回，而原生谓词一旦成立 done 即为真，
        # 所以原第 743 行的 retreat servo 一帧都记不下来（实测 5 条里 retreat 均为 0 帧）。
        # 这里改为**显式记录**的释放后观察期：先原地等待让物体沉降（2 s），
        # 再抬升离开，全程逐帧记录，供审计验证"放置后是否稳定"。
        sim_r, _, _, _ = N.ids(env)
        oe_release = 0.5 * (lambda b: b[0] + b[1])(bbox(sim_r, bid))
        PR_SETTLE_STEPS, PR_RETREAT_STEPS = 40, 30
        pr = {"settle_steps": 0, "retreat_steps": 0}
        # [复核 §6.3] 释放瞬间的状态必须在**释放那一刻**取。
        # 旧实现在整段观察结束后才赋 `predicate_true_at_release`，那不是释放时刻的值。
        pr["object_center_at_release"] = [float(v) for v in oe_release]
        pr["predicate_true_at_release"] = bool(ep.done)

        # [复核 §6.2] 观察窗口的**逐步**轨迹：谓词、是否在区域内、逐步位移与速度。
        # 旧实现只要求"观察计数 > 0"，既不要求跑满 40+30 步，也不要求窗口**连续**成立。
        win = {"center": [], "pred": [], "inside": []}

        def _in_region(c):
            return bool(rb[0] - 0.03 <= c[0] <= rh[0] + 0.03
                        and rb[1] - 0.03 <= c[1] <= rh[1] + 0.03
                        and c[2] > rb[2] - 0.03)

        def _sample(phase, steps):
            for _ in range(steps):
                if ep.n >= MAX_FRAMES:
                    break
                sim, m, d, gid = N.ids(env)
                p, R = N.gripsite(sim, m, d, gid)
                tgt = into if phase == "post_release_settle" else above
                ap = np.clip((tgt - p) / TEACHER_GAIN_POS_M, -1.0, 1.0)
                ar = np.clip(N.rot_err_vec(R_ref, R) / TEACHER_GAIN_ROT_RAD, -1.0, 1.0)
                ep.act(np.concatenate([ap, ar, [GRIP_ACT_OPEN]]), phase)
                pr["settle_steps" if phase == "post_release_settle"
                   else "retreat_steps"] += 1
                sim2, _, _, _ = N.ids(env)
                c = 0.5 * (lambda b: b[0] + b[1])(bbox(sim2, bid))
                win["center"].append([float(v) for v in c])
                win["pred"].append(bool(ep.done))
                win["inside"].append(_in_region(c))

        _sample("post_release_settle", PR_SETTLE_STEPS)
        # ★ 稳定判据的基准点取**沉降之后**：释放时物体仍在篮沿上方（z≈0.10–0.13 m），
        #   随后落入篮底（z≈0.0488 m），这 54–84 mm 是**预期的自由落体**，不是不稳定。
        sim_s, _, _, _ = N.ids(env)
        oe_settle = 0.5 * (lambda b: b[0] + b[1])(bbox(sim_s, bid))
        n_settle = len(win["center"])
        pr["object_center_at_settle_end"] = [float(v) for v in oe_settle]
        pr["fall_distance_mm"] = float((oe_release[2] - oe_settle[2]) * 1000.0)
        _sample("retreat", PR_RETREAT_STEPS)

        centers = np.array(win["center"], float) if win["center"] else np.zeros((0, 3))
        need = PR_SETTLE_STEPS + PR_RETREAT_STEPS
        pr["window_steps"] = int(len(centers))
        pr["window_required_steps"] = int(need)
        pr["window_complete"] = bool(len(centers) >= need)
        pr["window_predicate_all_true"] = bool(win["pred"] and all(win["pred"]))
        pr["window_inside_all"] = bool(win["inside"] and all(win["inside"]))
        tail = centers[n_settle:] if 0 < n_settle < len(centers) else centers[:0]
        if len(tail) >= 2:
            d_tail = np.linalg.norm(np.diff(tail, axis=0), axis=1)
            pr["drift_mm"] = float(np.linalg.norm(tail[-1] - tail[0]) * 1000.0)
            pr["tail_max_step_disp_mm"] = float(d_tail.max() * 1000.0)
            pr["tail_max_speed_m_s"] = float(d_tail.max() / ep.dt)
        else:
            pr["drift_mm"] = None
            pr["tail_max_step_disp_mm"] = None
            pr["tail_max_speed_m_s"] = None
    except Exception as e:                                   # noqa: BLE001
        log("  [!] 例外：%s" % e)

    sim3, _, _, _ = N.ids(env)
    oe = 0.5 * (lambda b: b[0] + b[1])(bbox(sim3, bid))
    in_basket = bool(rb[0] - 0.03 <= oe[0] <= rh[0] + 0.03
                     and rb[1] - 0.03 <= oe[1] <= rh[1] + 0.03
                     and oe[2] > rb[2] - 0.03)

    # ---------- 释放后稳定确认（审核 §4 步骤 5 + 复核 §6）----------
    # [§6.2] 稳定窗口必须：跑满 40+30 步、窗口内**每一步**谓词为真且物体在区域内、
    #         沉降后总漂移与**单步位移/速度**都受限。旧实现只要求"计数 > 0"。
    pr["object_center_at_end"] = [float(v) for v in oe]
    pr["predicate_true_at_end"] = bool(ep.done)
    pr["in_basket_at_end"] = bool(in_basket)
    if oe_release is not None:
        pr["drift_from_release_mm"] = float(np.linalg.norm(oe - oe_release) * 1000.0)
    else:
        pr["drift_from_release_mm"] = None
    pr["drift_basis"] = ("沉降结束后 → 观察期结束（不含落入篮底的自由落体段）；"
                         "单步位移/速度在同窗口内逐帧检查")
    pr["drift_max_mm"] = POST_RELEASE_DRIFT_MAX_MM
    pr["max_step_disp_mm_limit"] = POST_RELEASE_MAX_STEP_DISP_MM
    pr["max_speed_m_s_limit"] = POST_RELEASE_MAX_SPEED_M_S
    pr["stable"] = bool(
        pr.get("window_complete", False)
        and pr.get("window_steps", 0) >= POST_RELEASE_MIN_WINDOW_STEPS
        and pr.get("window_predicate_all_true", False)
        and pr.get("window_inside_all", False)
        and pr["predicate_true_at_end"] and in_basket
        and pr.get("drift_mm") is not None
        and pr["drift_mm"] < POST_RELEASE_DRIFT_MAX_MM
        and pr.get("tail_max_step_disp_mm") is not None
        and pr["tail_max_step_disp_mm"] < POST_RELEASE_MAX_STEP_DISP_MM
        and pr.get("tail_max_speed_m_s") is not None
        and pr["tail_max_speed_m_s"] < POST_RELEASE_MAX_SPEED_M_S)

    # [复核 §6] servo 超时清单：超时必须触发失败，不能静默放行。
    # 复核实测 transport 恒耗满 220 步、place 恒耗满 160 步（占约 73% 帧数），
    # 说明这两段**从未满足 8 mm 容差**。现在把它显式记为失败。
    TIMEOUT_PHASES = ("approach", "descend", "lift", "transport", "place")
    to_list = []
    for ph in TIMEOUT_PHASES:
        v = pe.get(ph)
        if isinstance(v, (list, tuple)) and len(v) == 2 and not bool(v[0]):
            to_list.append({"phase": ph, "steps": int(v[1])})
    pe["servo_timeouts"] = to_list
    pe["servo_no_timeout"] = bool(not to_list)

    # [复核 §6.1] 稳定性必须进入**正式成功门槛**
    success = bool(ep.done) and in_basket and pr["stable"] and pe["servo_no_timeout"]

    meta = {
        "node": "N4", "suite": args.suite, "task_id": args.task_id,
        "seed": seed, "teacher": "scripted_state_driven",
        "object": bname, "region": rname,
        "object_init_fingerprint": obj_fp,
        "object_init_sample": init_sample,
        "object_bbox_lo": [float(v) for v in ob],
        "object_bbox_hi": [float(v) for v in oh],
        "grasp_pinch_target": [float(v) for v in pin], "grasp_z": z_g,
        "section_width_x_m": w_x, "section_width_y_m": w_y,
        "gripper_rotated_90": rotated,
        "success_predicate_libero": bool(ep.done),
        "geometric_in_basket": in_basket, "success": success,
        "lifted_mm": lifted_mm, "phase_end": pe, "n_frames": ep.n,
        "control_hz": 1.0 / ep.dt, "control_timestep": ep.dt,
        "image_flip": "180 ([::-1,::-1])，与 examples/libero/main.py 一致",
        "image_res": RES, "eval_resize": TARGET_IMG,
        "action_space": "OSC_POSE 7-dim：世界系 dx,dy,dz + 世界系 drx,dry,drz + grip(-1开/+1合)",
        # ---- [R6] 教师增益与真实动作尺度分开保存 ----
        "teacher_gain_pos_m": TEACHER_GAIN_POS_M,
        "teacher_gain_rot_rad": TEACHER_GAIN_ROT_RAD,
        "teacher_gain_note": "脚本专家把 grip_site 误差换算成归一化动作用的除数，"
                             "属教师内部实现，**不是**控制器的动作尺度。",
        # ---- [R4] state 契约：按 joint name 落地 ----
        "state_contract": "8 维 = [grip_site xyz(3, m), grip_site 轴角(3, rad), "
                          "Left_1_Joint, Right_1_Joint]（主动驱动关节，按 joint name 定位）",
        "state_gripper_joint_names": list(ep.grip_names),
        "state_gripper_joint_slots": [int(i) for i in ep.grip_idx],
        "state_gripper_qpos_indices": [
            int(env.env.robots[0]._ref_gripper_joint_pos_indexes[i]) for i in ep.grip_idx],
        "gripper_joint_order": list(env.env.robots[0].gripper.joints),
        # ---- [R2] 首帧观测一致性证据 ----
        "first_frame_obs_staleness": obs_staleness,
        "first_frame_sync": first_frame_sync,
        # ---- 释放后稳定确认（审核 §4 步骤 5）----
        # [复核 §6] 完成任务的时间与"成功后的诊断观察"分开记录：
        # 不能用 70 步观察掩盖前面 380 步未收敛。
        "task_frames": int(len(ep.frames) - pr.get("window_steps", 0)),
        "observation_frames": int(pr.get("window_steps", 0)),
        "servo_timeouts": pe.get("servo_timeouts", []),
        "servo_no_timeout": bool(pe.get("servo_no_timeout", False)),
        "post_release": pr,
        "post_release_stable": bool(pr.get("stable")),
        "gripper_speed": float(env.env.robots[0].gripper.speed),
        "final_object_center": [float(v) for v in oe],
        "basket_bbox_lo": [float(v) for v in rb],
        "basket_bbox_hi": [float(v) for v in rh],
        "robot": "OnTheGroundECO65", "gripper": "ZhixingGripper",
        "tcp_eef_pos": EEF_POS,
        "zhixing_gripper_xml_sha16": sha16(XML_PATH),
        "eco65_robot_xml_sha16": sha16(ROBOT_XML),
        "collector_sha16": sha16(COLLECTOR),
        "init_qpos_sha16": hashlib.sha256(init_qpos.tobytes()).hexdigest()[:16],
    }
    meta.update(controller_action_contract(env))
    return (ep, meta, init_qpos, init_qvel, init_warm, init_grip_act,
            init_ctrl, init_qfrc_applied, init_xfrc_applied, success)


EEF_POS = None


def eef_pos_of(env):
    import mujoco

    _, m, _, _ = N.ids(env)
    for b in range(m.nbody):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith("eef"):
            return [round(float(v), 6) for v in m.body_pos[b]]
    return None


def save_episode(path, arr, meta, iq, iv, iw, iga=None, ictrl=None,
                 iqfrc=None, ixfrc=None):
    """原子保存一个 episode（写 .tmp 再 rename），**拒绝覆盖**已有成果。

    格式说明（与方案的唯一有意偏离）
    --------------------------------
    `任务方案.md` P1 **建议**原始记录用 HDF5。但实测 `libero:latest` 运行容器
    的 venv 里**没有 h5py**（也没有 pip / uv，tables / zarr / pyarrow 均无），
    且 venv 位于镜像内、`--rm` 每次重建，装不进去。若为此重建镜像，会动摇
    已验证的 9/10 基线与 N0~N3 全部结论，代价与风险都不划算。
    故原始 episode 采用 **numpy `.npz`（压缩）**：零依赖、数组语义与 HDF5
    等价、同样满足「per-episode 原子保存 + 不覆盖」；N5 转 LeRobot 时再统一
    成发布格式（届时容器里有没有 h5py 已不影响数据内容）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError("拒绝覆盖已存在的 episode：%s" % path)
    tmp = path.with_name(path.name + ".tmp.npz")
    payload = {k: v for k, v in arr.items() if not k.startswith("_")}
    payload["init_qpos"] = iq
    payload["init_qvel"] = iv
    payload["init_qacc_warmstart"] = iw
    # [R3] 夹爪累积控制状态（缺它就复现不了回放第一步的夹爪命令）
    if iga is not None:
        payload["init_gripper_current_action"] = np.asarray(iga, dtype=np.float64)
    # [复核 §2.3.2] 执行器控制量与外力缓存（缺失时回放会出现首个分歧帧）
    for key, val in (("init_ctrl", ictrl), ("init_qfrc_applied", iqfrc),
                     ("init_xfrc_applied", ixfrc)):
        if val is not None:
            payload[key] = np.asarray(val, dtype=np.float64)
    payload["_meta_json"] = np.array(json.dumps(meta, ensure_ascii=False))
    payload["_phases_json"] = np.array(json.dumps(arr["_phases"], ensure_ascii=False))
    np.savez_compressed(tmp, **payload)
    tmp.replace(path)                          # 原子落盘
    return path


def main():
    global EEF_POS
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seeds", default="")
    ap.add_argument("--want", type=int, default=5)
    ap.add_argument("--max-attempts", type=int, default=24)
    ap.add_argument("--close-max-steps", type=int, default=300)
    ap.add_argument("--vary-init", dest="vary_init", action="store_true",
                    default=True,
                    help="在 BDDL 区域内重摆目标物体，保证每条示范是**不同初始化**")
    ap.add_argument("--no-vary-init", dest="vary_init", action="store_false")
    ap.add_argument("--gripper-speed", type=float, default=0.01,
                    help="ZhixingGripper.speed（经环境变量注入）。默认 0.01："
                         "资产原值 0.2 会让夹爪在 ~2 个 env step 内闭死"
                         "（≈900 mm/s，远超声厂商规格 <100 mm/s），脚本专家"
                         "闭合时会把物体**拨飞**；0.01 与 PandaGripper 同值，"
                         "闭合渐进可控。该值写入 episode 元数据。")
    ap.add_argument("--tag", default="n4")
    args = ap.parse_args()

    out_s = ROOT / "raw" / "success"
    out_f = ROOT / "raw" / "failure"
    out_s.mkdir(parents=True, exist_ok=True)
    out_f.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    log("=" * 78)
    log("[N4] 脚本专家采集  suite=%s task=%d  目标 %d 条成功"
        % (args.suite, args.task_id, args.want))

    os.environ["ZHIXING_GRIPPER_SPEED"] = str(args.gripper_speed)
    env, task = build(args.suite, args.task_id)
    EEF_POS = eef_pos_of(env)
    _spd = float(env.env.robots[0].gripper.speed)
    log("[grip] ZhixingGripper.speed = %.3f（资产原值 0.2；本次采集值已写入元数据）"
        % _spd)
    if abs(_spd - args.gripper_speed) > 1e-9:
        log("[FATAL] speed 注入失败：期望 %.4f，实得 %.4f"
            % (args.gripper_speed, _spd))
        return 3
    log("[env] task: %s" % task.language)
    log("[env] 机器人=%s  夹爪=%s"
        % (type(env.env.robots[0].robot_model).__name__,
           type(env.env.robots[0].gripper).__name__))
    log("[tcp] eef body pos = %s  zhixing_gripper.xml sha16 = %s"
        % (EEF_POS, sha16(XML_PATH)))

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] or \
        list(range(1, args.max_attempts + 1))
    summary = {"node": "N4", "tag": args.tag, "suite": args.suite,
               "task_id": args.task_id, "task_language": task.language,
               "want": args.want, "attempts": [], "n_success": 0}
    n_ok, t0 = 0, time.time()
    for i, sd in enumerate(seeds):
        if n_ok >= args.want:
            break
        log("\n[ep %d] seed=%d" % (i + 1, sd))
        try:
            (ep, meta, iq, iv, iw, iga, ictrl, iqfrc, ixfrc, ok) = \
                run_episode(env, args, sd, log)
        except Exception as e:                            # noqa: BLE001
            log("  [!] episode 异常：%s" % e)
            summary["attempts"].append({"seed": sd, "success": False,
                                        "error": str(e)})
            continue
        meta["task_language"] = task.language
        dest = (out_s if ok else out_f) / (
            "%s_%s_t%d_s%d_%04d.npz" % (args.tag, args.suite, args.task_id, sd, i + 1))
        save_episode(dest, ep.arrays(), meta, iq, iv, iw, iga,
                     ictrl, iqfrc, ixfrc)
        n_ok += int(ok)
        summary["attempts"].append({
            "seed": sd, "success": bool(ok), "frames": ep.n,
            "path": str(dest.relative_to(ROOT)),
            "lifted_mm": meta["lifted_mm"],
            "success_predicate": meta["success_predicate_libero"],
            "geometric_in_basket": meta["geometric_in_basket"],
            "close_steps": meta["phase_end"].get("close_steps")})
        log("  → %s  帧=%d  抬升=%.1f mm  谓词=%s  几何入篮=%s"
            % ("✅成功" if ok else "❌失败", ep.n, meta["lifted_mm"],
               meta["success_predicate_libero"], meta["geometric_in_basket"]))
        log("     %s" % dest.name)

    summary["n_success"] = n_ok
    summary["success_rate"] = n_ok / max(1, len(summary["attempts"]))
    summary["wall_seconds"] = round(time.time() - t0, 1)
    summary["verdict"] = "PASS" if n_ok >= args.want else "NOT_ENOUGH"
    p = ROOT / "results" / ("%s_collect_summary.json" % args.tag)
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log("\n" + "=" * 78)
    log("[N4] 成功 %d/%d（尝试 %d 次）  用时 %.0f s  → %s"
        % (n_ok, args.want, len(summary["attempts"]), summary["wall_seconds"],
           summary["verdict"]))
    log("[out] %s" % p)
    env.close()
    return 0 if n_ok >= args.want else 1


if __name__ == "__main__":
    sys.exit(main())
