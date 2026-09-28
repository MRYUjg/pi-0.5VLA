#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4 审计：对 `raw/success`（及可选 `raw/failure`）的 episode 做契约与可回放检查。

对应 `任务方案.md` P2 的数据质量检查清单，逐条落地为可判定项：

  1. 结构：形状/类型/帧数一致；`image`/`wrist_image` 是**两路不同相机**
  2. 数值：无 NaN/Inf；动作落在合法区间；夹爪指令只取 {-1, +1}
  3. 时间：`timestamp` 严格等间隔且 = `control_timestep`（20 Hz）
  4. o_t/a_t 配对：**回放验证** + **观测/状态一致性** + **滞后判别**（R3）
  5. 初始化多样性：`init_qpos` 哈希与物体初态指纹**两两不同**
  6. 夹爪语义：`state` 两维必须是**主动关节** Left_1/Right_1（R4）
  7. 动作尺度元数据：教师增益与控制器 output_max 分离（R6）
  8. 首帧观测新鲜度：`obs_step == step_index`（R2）

R3 修复要点（审核记录 R3）
--------------------------
* **严格重建环境配置**：采集用 `ZHIXING_GRIPPER_SPEED=<meta>`，回放必须显式恢复，
  否则干净进程按 0.2 跑（实测首步夹爪角误差 0.3154 rad）。
* **恢复夹爪累积控制状态**：`format_action` 是累加式，采集前 45 步开爪已把它
  饱和；只恢复 qpos/qvel/warmstart 不够（恢复后首步误差 0.00002 rad）。
* **回放异常与超阈值必须是硬失败**，不再只写进结果。
* **分类比较**：臂关节角、夹爪关节角、物体位置、物体姿态**分开**统计，
  不再把混合 qpos 最大差统称为 rad。
* **图像/状态一致性**：回放渲染出的画面必须与存档图像一致；
  另用刻意可辨别的短动作序列做**检测器自检**（`--selftest`）。
"""
import argparse
import datetime
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402
import n4_collect as C     # noqa: E402

ROOT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927")

# ---- 预定义回放阈值 ----
EARLY_FROM, EARLY_FRAMES = 20, 44   # 自由空间段（接触前）
ARM_TOL_RAD = 0.01                  # 自由空间段 臂关节角 允许最大偏差
GRIP_TOL_RAD = 0.01                 # 自由空间段 夹爪关节角 允许最大偏差
RENDER_TOL_255 = 3.0                # 回放渲染 vs 存档图像 允许平均像素差
LAG_MIN_CHANGE = 0.15               # 认定为"动作突变帧"的最小指令变化量
# [复核 §2.3.5] 自检要求 lag0 比**两个**错位方向都低至少该倍数
SELFTEST_MARGIN = 1.25
OBS_STATE_TOL_M = 1e-4              # 观测/状态一致性（中位数）容差
# 期望的 OSC_POSE 动作尺度（robosuite/controllers/config/osc_pose.json）
EXPECTED_OUTPUT_MAX = [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]

# --------------------------------------------------------------------------- #
# [复核 §2.3] 全程判据与容差标定
# --------------------------------------------------------------------------- #
# 复核指出：原实现只把第 20–44 帧接入硬判据，后段物体位置偏差到 575 mm 仍判 PASS。
# 现要求**全程所有阶段**都进判据，且容差必须"基于同配置重复运行标定，并在最终
# 放行前固定"。因此：
#   * 阈值不再硬编码在代码里充当"事实标准"，而是从 TOL_FILE 读取；
#   * 该文件不存在（未标定）时，相关闸门一律记为**未标定 → 不得放行**，
#     而不是用一个宽松默认值把数据放过去。
TOL_FILE = ROOT / "configs" / "n4_audit_tolerances.json"
# 标定脚本用这组量：对同一 episode 重复回放 N 次，量出"同配置重复运行"的离散度
CALIB_REPEATS_DEFAULT = 3
# 判据键 → 该量在标定文件中的字段名
TOL_KEYS = {
    "phase_arm_rad": "phase_max_arm_rad",
    "phase_grip_rad": "phase_max_grip_rad",
    "phase_obj_pos_m": "phase_max_obj_pos_m",
    "phase_obj_rot_rad": "phase_max_obj_rot_rad",
    "final_obj_pos_m": "final_obj_pos_m",
    "render_agentview_255": "render_agentview_masked_255",
    "render_wrist_255": "render_wrist_masked_255",
}

# 双相机核对：首帧、每个阶段边界前后、末帧，以及运动窗口
RENDER_KEY_FRAMES_MAX = 14          # 每条 episode 最多核对的帧数（含首/末帧）
MOTION_PIXEL_THRESH = 6             # 判定"运动像素"的相邻帧灰度差阈值

# [复核 §2.3.5] 负对照：对**同一个正式验收入口**注入，逐一验证检测能力
NEG_MODES = ("action_shift", "tail_replace", "wrist_shift", "object_state")


def load_tolerances():
    """读取标定后的容差；文件缺失或字段不全时返回 None（→ 判为未标定）。"""
    if not TOL_FILE.exists():
        return None
    try:
        d = json.loads(TOL_FILE.read_text(encoding="utf-8"))
    except Exception:                                     # noqa: BLE001
        return None
    if not all(k in d for k in TOL_KEYS.values()):
        return None
    return d


def load(p):
    return np.load(p, allow_pickle=True)


def quat_angle(q1, q2):
    """两个 wxyz 四元数之间的夹角（rad）。"""
    a = np.asarray(q1, float)
    b = np.asarray(q2, float)
    d = abs(float(np.dot(a, b)))
    return 2.0 * np.arccos(min(1.0, d))


def object_qpos_addr(m, obj_name):
    """按 body 名找该物体的 free joint 的 qpos 起始地址；找不到返回 None。"""
    import mujoco

    if not obj_name:
        return None
    cands = []
    for b in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if nm == obj_name:
            cands.insert(0, b)
        elif obj_name and (nm.startswith(obj_name) or obj_name.startswith(nm)):
            cands.append(b)
    for b in cands:
        ja = int(m.body_jntadr[b])
        if ja >= 0 and int(m.jnt_type[ja]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return int(m.jnt_qposadr[ja])
    return None


# --------------------------------------------------------------------------- #
# [复核 §2.3.1] 按 episode 元数据重建环境，取消固定 libero_object/task0 假设
# --------------------------------------------------------------------------- #
# LIBERO 会按 problem 类给机器人名加前缀（floor/coffee/living-room → `OnTheGround`，
# tabletop → `Mounted`），而 `build()` 接收的是**基础名**。元数据里记录的是加前缀
# 之后的类名（如 `OnTheGroundECO65`），故重建时必须剥掉前缀，并在建好后
# **断言实际模型类名与元数据一致** —— 对不上就报错，不静默用错环境。
ROBOT_NAME_PREFIXES = ("OnTheGround", "Mounted")


def env_spec_from_meta(meta):
    """从 episode 元数据推出 (suite, task_id, 基础机器人名, 夹爪名)。

    优先使用采集时**显式记录**的构建参数；旧数据没有该字段时按前缀反推。
    """
    suite = meta.get("suite")
    task_id = int(meta.get("task_id", 0))
    gripper = meta.get("gripper") or "ZhixingGripper"
    if meta.get("env_robot_spec"):
        return suite, task_id, meta["env_robot_spec"], meta.get("env_gripper_spec", gripper)
    cls = str(meta.get("robot") or "")
    base = cls
    for pfx in ROBOT_NAME_PREFIXES:
        if cls.startswith(pfx):
            base = cls[len(pfx):]
            break
    return suite, task_id, base or "ECO65", gripper


def build_env_cached(cache, spec):
    """按 spec 建环境并缓存；同一 spec 只建一次。"""
    if spec in cache:
        return cache[spec]
    suite, task_id, robot, gripper = spec
    env, _task = C.build(suite, task_id, robot=robot, gripper=gripper, seed=1)
    cache[spec] = env
    return env


def assert_env_matches_meta(env, meta, r):
    """[§2.3.1] 启动时核验资产/控制器/速度/相机配置指纹；不一致就记 FAIL。"""
    rb = env.env.robots[0]
    checks = {}
    info = {"robot_model_class": type(rb.robot_model).__name__,
            "gripper_class": type(rb.gripper).__name__}
    # 1) 机器人/夹爪类名必须与元数据一致
    checks["env_robot_class_matches_meta"] = bool(
        not meta.get("robot") or info["robot_model_class"] == meta["robot"])
    checks["env_gripper_class_matches_meta"] = bool(
        not meta.get("gripper") or info["gripper_class"] == meta["gripper"])
    # 2) 资产哈希（夹爪 XML / 机器人 XML）
    import hashlib
    for key, path, meta_key in (
            ("zhixing_gripper_xml", "/app/robosuite/models/assets/grippers/zhixing_gripper.xml",
             "zhixing_gripper_xml_sha16"),
            ("eco65_robot_xml", "/app/robosuite/models/assets/robots/ECO65-B/robot.xml",
             "eco65_robot_xml_sha16")):
        want = meta.get(meta_key)
        try:
            got = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16]
        except Exception:                                 # noqa: BLE001
            got = None
        info[key + "_sha16"] = got
        info[key + "_recorded"] = want
        checks["env_" + key + "_matches_meta"] = bool(want is None or got == want)
    # 3) 控制器与动作尺度
    ctrl = rb.controller
    info["controller_type"] = type(ctrl).__name__
    info["controller_output_max"] = [float(v) for v in np.atleast_1d(ctrl.output_max)]
    com = meta.get("controller_output_max")
    checks["env_controller_output_max_matches_meta"] = bool(
        com is None or np.allclose(info["controller_output_max"], com, atol=1e-12))
    # 4) 夹爪速度（采集时逐条写入元数据）
    want_spd = meta.get("gripper_speed")
    info["gripper_speed"] = float(rb.gripper.speed)
    checks["env_gripper_speed_matches_meta"] = bool(
        want_spd is None or abs(info["gripper_speed"] - float(want_spd)) < 1e-12)
    # 5) 相机配置
    info["camera_names"] = sorted(list(env.env.camera_names)) if hasattr(env.env, "camera_names") else None
    # 相机分辨率：LIBERO/robosuite 可能把它存成 list（每个相机一项），统一成 list[int]
    def _as_int_list(v):
        if v is None:
            return None
        if isinstance(v, (list, tuple, np.ndarray)):
            return [int(x) for x in np.atleast_1d(v)]
        return [int(v)]
    ch = _as_int_list(getattr(env.env, "camera_heights", None))
    cw = _as_int_list(getattr(env.env, "camera_widths", None))
    info["camera_heights"] = ch
    info["camera_widths"] = cw
    info["camera_res"] = [ch, cw]
    want_res = meta.get("image_res")
    checks["env_camera_res_matches_meta"] = bool(
        want_res is None or (ch and cw
                             and all(v == int(want_res) for v in ch)
                             and all(v == int(want_res) for v in cw)))
    checks["env_has_agentview_and_wrist"] = bool(
        info["camera_names"] is not None
        and any("agentview" in c for c in info["camera_names"])
        and any("eye_in_hand" in c for c in info["camera_names"]))
    r["env_fingerprint"] = info
    r["checks"].update(checks)
    return checks


# --------------------------------------------------------------------------- #
# [复核 §2.3.4] 双相机关键帧核对 + 运动窗口掩膜
# --------------------------------------------------------------------------- #
def key_check_frames(pid, phases, n):
    """返回要核对的帧下标：首帧、末帧、每个阶段边界、以及按比例补充的帧。

    纯背景帧会把错位信号稀释掉（复核 §2.3.4），所以这里既选**相位边界**
    （动作切换处最容易暴露错位），也选**运动幅度最大**的若干帧。
    """
    sel = {0, n - 1}
    if len(pid) != n:
        return sorted(sel)
    # 阶段边界（每次 phase_id 变化）
    change = np.nonzero(np.diff(pid))[0]
    for c in change:
        sel.add(int(c))
        sel.add(int(c) + 1)
    # 均匀补足
    if len(sel) < RENDER_KEY_FRAMES_MAX:
        step = max(1, n // RENDER_KEY_FRAMES_MAX)
        sel.update(range(0, n, step))
    return sorted(f for f in sel if 0 <= f < n)[:max(RENDER_KEY_FRAMES_MAX, len(change) * 2 + 2)]


def masked_render_diff(rendered, stored, prev_stored):
    """返回 (掩膜内平均差, 全图平均差, 掩膜占比)。

    掩膜 = 存档相邻帧之间发生变化的像素（运动窗口）∪ 明显偏暗/偏亮的机器人区域近似。
    仅用全图平均会把错位信号用背景稀释掉。
    """
    d = np.abs(rendered.astype(np.int16) - stored.astype(np.int16))
    plain = float(d.mean())
    if prev_stored is None:
        return plain, plain, 1.0
    motion = (np.abs(stored.astype(np.int16)
                     - prev_stored.astype(np.int16)).max(axis=2) > MOTION_PIXEL_THRESH)
    frac = float(motion.mean())
    if motion.sum() < 50:            # 几乎静止：退回全图
        return plain, plain, frac
    return float(d.max(axis=2)[motion].mean()), plain, frac


# --------------------------------------------------------------------------- #
# [复核 §2.3.5] 负对照注入（作用于内存数组，不改原始文件）
# --------------------------------------------------------------------------- #
def apply_negative_control(act, wri, qpos, mode, start=100):
    """就地破坏一份**内存中的** episode 数组，用于验证验收器能否拒绝坏样本。

    返回 (act, wri, qpos)。全部先 `np.array(..., copy=True)`，
    确保绝不触碰磁盘上的原始 NPZ。
    """
    if mode is None:
        return act, wri, qpos
    act = np.array(act, dtype=np.float64, copy=True)
    wri = np.array(wri, copy=True)
    qpos = np.array(qpos, dtype=np.float64, copy=True)
    if mode == "action_shift":
        # 动作整体错位一帧 —— 检验配对判据
        act = np.roll(act, 1, axis=0)
    elif mode == "tail_replace":
        # 尾段动作替换（复核的原始负对照做法）—— 检验全程判据
        act[start:, :6] = 0.0
        act[start:, 6] = -1.0
    elif mode == "wrist_shift":
        # 腕部图像错移一帧 —— 检验腕部相机核对
        wri = np.roll(wri, 1, axis=0)
    elif mode == "object_state":
        # 物体状态破坏 —— 检验物体位置/姿态判据
        qpos[start:, :3] += 0.05
    else:
        raise ValueError("未知负对照模式 %r（可选 %s）" % (mode, NEG_MODES))
    return act, wri, qpos


def lag_residuals(act, disp, free, min_change=LAG_MIN_CHANGE):
    """用**最小二乘拟合残差**判别 o_t/a_t 是否错位一帧。

    审核记录 R3 指出：旧的"动作与位移方向点积为正 ≥95%"在平滑轨迹上
    lag0 与 lag±1 同为 0.975/1.0，**没有判别力**。这里改为
      * 只取**相邻动作变化明显**的帧（突变帧），错位一帧在这些帧上必然对不齐；
      * 对每个 lag 用最小二乘拟合 `disp ≈ k·a[:, :3]`，取归一化残差；
      * 正确配对应给出最小且足够小的残差。
    """
    n1 = len(disp)
    ch = np.zeros(n1, dtype=bool)
    if n1 >= 2:
        ch[:-1] = np.abs(act[1:n1][:, :3] - act[:n1 - 1][:, :3]).max(axis=1) > min_change
    base = np.arange(n1)
    out = {}
    for lag, off in (("0", 0), ("+1", 1), ("-1", -1)):
        ai = base + off
        ok = (ai >= 0) & (ai < len(act))
        a = act[ai[ok]]
        dv = disp[base[ok]]
        ff = free[base[ok]]
        cc = ch[base[ok]]
        sel = ((np.abs(a[:, :3]).max(axis=1) > 0.05)
               & (np.linalg.norm(dv, axis=1) > 1e-5) & ff & cc)
        if int(sel.sum()) < 5:
            out[lag] = None
            continue
        A, D = a[sel][:, :3], dv[sel]
        k = float((A * D).sum() / max(float((A * A).sum()), 1e-12))
        resid = D - k * A
        out[lag] = round(float(np.linalg.norm(resid, axis=1).mean()
                               / max(float(np.linalg.norm(D, axis=1).mean()), 1e-9)), 4)
    return out


def run_pairing_selftest(env):
    """验证配对检测器**真的有判别力**：用刻意可辨别的短动作序列做 ±1 帧对照。

    做法：
      1. 在环境里跑一段方向频繁突变（6 个方向各 3 步）的动作序列，
         逐帧存档 动作 + 渲染画面 + 状态；
      2. 从**同一初始状态**出发，分别按 lag 0 / +1 / −1 重放同一段动作，
         比较每一步的渲染画面与存档画面；
      3. 正确配对（lag0）应当几乎逐像素一致；错位一帧则轨迹偏离、画面明显不同。

    这正是审核记录 R3 要求的"可辨别的短动作序列"检验：若 lag0 不能显著
    优于 lag±1，说明该检测器无法排除一帧错位，后续配对结论不可信。
    """
    import mujoco

    env.seed(20260928)
    env.reset()
    sim, m, d, gid = N.ids(env)
    rbt = env.env.robots[0]
    q0 = np.array(d.qpos, dtype=float).copy()
    v0 = np.array(d.qvel, dtype=float).copy()
    w0 = np.array(d.qacc_warmstart, dtype=float).copy()
    ca0 = np.array(rbt.gripper.current_action, dtype=float).copy()
    p0, _ = N.gripsite(sim, m, d, gid)

    def frame(o):
        return np.ascontiguousarray(o["agentview_image"][::-1, ::-1]).astype(np.int16)

    dirs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                     [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    acts, imgs = [], []
    for dv in dirs:
        for _ in range(3):
            sim, m, d, gid = N.ids(env)
            p, _ = N.gripsite(sim, m, d, gid)
            ap = np.clip(0.5 * dv - 0.25 * (p - p0), -1.0, 1.0)
            a = np.concatenate([ap, np.zeros(3), [-1.0]])
            acts.append(a.copy())
            obs, _, _, _ = env.step(a)
            imgs.append(frame(obs))
    acts = np.array(acts)

    def restore():
        sim_, m_, d_, _ = N.ids(env)
        d_.qpos[:] = q0
        d_.qvel[:] = v0
        d_.qacc_warmstart[:] = w0
        mujoco.mj_forward(m_, d_)
        rb = env.env.robots[0]
        for meth in ("reset_goal", "update"):
            fn = getattr(rb.controller, meth, None)
            if fn is None:
                continue
            try:
                fn(force=True)
            except TypeError:
                fn()
        rb.gripper.current_action = ca0.copy()

    hold = np.concatenate([np.zeros(6), [-1.0]])
    errs = {}
    for lag in (0, 1, -1):
        restore()
        e = []
        for t in range(len(acts)):
            ai = t + lag
            a = acts[ai] if 0 <= ai < len(acts) else hold
            obs, _, _, _ = env.step(a)
            e.append(float(np.mean(np.abs(frame(obs) - imgs[t]))))
        # 丢掉前 2 步的控制器暂态，取中位数
        errs[str(lag)] = round(float(np.median(e[2:])), 3)

    # [复核 §2.3.5] 原实现用 max(err(+1), err(-1)) 比较：只要**一个**错位方向较差
    # 就能通过。改为两个方向都必须被区分。
    # 判据形式：lag0 必须比**两个**错位方向都低至少 SELFTEST_MARGIN 倍。
    # 倍数取 1.25（而非 2.0）是因为该裕度会随渲染后端变化：GPU/EGL 下实测
    # 0.128 vs 0.258/0.309（2.0×/2.4×），CPU/OSMesa 下 0.198 vs 0.329/0.387
    # （1.66×/1.95×）。判据必须能在两种后端下都成立，同时仍要求"两个方向都被区分"。
    r1 = errs["1"] / max(errs["0"], 1e-9)
    rm1 = errs["-1"] / max(errs["0"], 1e-9)
    ok = bool(errs["0"] < RENDER_TOL_255
              and r1 >= SELFTEST_MARGIN and rm1 >= SELFTEST_MARGIN)
    return {"render_err_by_lag": errs, "lag0_is_best": ok,
            "both_shifts_distinguished": ok,
            "improvement_ratio_vs_plus1": round(r1, 3),
            "improvement_ratio_vs_minus1": round(rm1, 3),
            "margin_required": SELFTEST_MARGIN,
            "criterion": ("lag0 必须比 lag+1 与 lag−1 **都**低至少 %.2f 倍，"
                          "且 < 渲染阈值 %.1f" % (SELFTEST_MARGIN, RENDER_TOL_255)),
            "render_tol_255": RENDER_TOL_255,
            "note": "可辨别短动作序列（6 方向突变 ×3 步）上的 ±1 帧回放对照"}


def check_episode(path, replay_env=None, replay=True, selftest=None,
                  neg=None, tolerances=None, env_cache=None):
    """单条 episode 的完整验收。

    [复核 §2.3.5] 负对照通过 `neg` 注入到**这个正式入口**，不改文件；
    `neg is None` 即正常样本。任何一条负对照若仍判 PASS，说明验收器有漏检。
    """
    f = load(path)
    meta = json.loads(str(f["_meta_json"]))
    r = {"file": path.name, "frames": int(f["image"].shape[0]), "checks": {}}
    if neg is not None:
        r["negative_control"] = neg

    img, wri = f["image"], f["wrist_image"]
    act, qpos, qvel = f["actions"], f["qpos"], f["qvel"]
    st, ts = f["state"], f["timestamp"]
    n = img.shape[0]
    c = r["checks"]

    # [复核 §2.3.5] 负对照注入必须在任何检查之前完成（只改内存副本）
    if neg is not None:
        act, wri, qpos = apply_negative_control(act, wri, qpos, neg)

    c["frames_match"] = bool(img.shape[0] == wri.shape[0] == act.shape[0]
                             == qpos.shape[0] == qvel.shape[0] == st.shape[0]
                             == ts.shape[0])
    c["shapes_ok"] = bool(img.shape[1:] == (C.RES, C.RES, 3)
                          and wri.shape[1:] == (C.RES, C.RES, 3)
                          and st.shape[1] == 8 and act.shape[1] == 7)
    c["dtypes_ok"] = bool(img.dtype == np.uint8 and wri.dtype == np.uint8)
    c["no_nan_inf"] = bool(np.all(np.isfinite(qpos)) and np.all(np.isfinite(qvel))
                           and np.all(np.isfinite(act)) and np.all(np.isfinite(st)))

    dts = np.diff(ts) if n > 1 else np.array([0.0])
    c["dt_uniform_20hz"] = bool(n > 1 and np.allclose(dts, dts[0])
                                and abs(dts[0] - 0.05) < 1e-9)
    c["timestamp_monotonic"] = bool(np.all(dts > 0))

    c["action_in_range"] = bool(act.min() >= -1.0 - 1e-9 and act.max() <= 1.0 + 1e-9)
    gc = np.unique(np.round(act[:, 6], 6))
    c["grip_cmd_is_plusminus1"] = bool(set(gc.tolist()) <= {-1.0, 1.0})
    c["grip_sign_convention"] = bool(-1.0 in gc and 1.0 in gc)

    diff = float(np.mean(np.abs(img.astype(np.int16) - wri.astype(np.int16))))
    r["camera_mean_abs_diff"] = round(diff, 3)
    c["cameras_differ"] = bool(diff > 5.0)
    r["image_mean"] = round(float(img.mean()), 2)
    r["wrist_mean"] = round(float(wri.mean()), 2)
    c["images_not_blank"] = bool(3.0 < float(img.mean()) < 252.0
                                 and 3.0 < float(wri.mean()) < 252.0)

    phases = json.loads(str(f["_phases_json"]))
    r["phases"] = phases
    need = {"settle", "approach", "descend", "close", "lift", "transport",
            "place", "release", "post_release_settle", "retreat"}
    c["phases_cover_pickplace"] = bool(need <= set(phases))

    # ---- [审核 §4 步骤 5] 释放后稳定确认 ----
    # 要求：观察期内原生谓词仍为真、物体仍在目标区域、且漂移 < 阈值。
    # 旧数据没有这一段（`servo()` 见 done 即返回，retreat 一帧都记不下），故如实判否。
    pres = meta.get("post_release")
    if isinstance(pres, dict):
        r["post_release"] = pres
        c["post_release_recorded"] = bool(int(pres.get("settle_steps") or 0) > 0
                                          and int(pres.get("retreat_steps") or 0) > 0)
        # [复核 §6] **不直接信任**元数据里的 `stable`，按同一判据从原始字段独立重算；
        # 并要求窗口跑满 40+30 步（缺步骤/中途异常不得以"最后一个成功点"放行）。
        c["post_release_no_missing_steps"] = bool(
            int(pres.get("settle_steps") or 0) >= 40
            and int(pres.get("retreat_steps") or 0) >= 30
            and int(pres.get("window_steps") or 0) >= 70)
        recomputed = bool(
            c["post_release_no_missing_steps"]
            and pres.get("window_predicate_all_true")
            and pres.get("window_inside_all")
            and pres.get("predicate_true_at_end")
            and pres.get("in_basket_at_end")
            and pres.get("drift_mm") is not None
            and pres["drift_mm"] < 5.0
            and pres.get("tail_max_step_disp_mm") is not None
            and pres["tail_max_step_disp_mm"] < 5.0
            and pres.get("tail_max_speed_m_s") is not None
            and pres["tail_max_speed_m_s"] < 0.05)
        r["post_release_recomputed_stable"] = recomputed
        c["post_release_stable"] = recomputed
        # 元数据自报与独立重算必须一致（防"元数据说稳定、原始数据说不稳定"）
        c["post_release_metadata_agrees"] = bool(bool(pres.get("stable")) == recomputed)
    else:
        r["post_release"] = None
        r["post_release_recomputed_stable"] = False
        c["post_release_recorded"] = False
        c["post_release_no_missing_steps"] = False
        c["post_release_stable"] = False
        c["post_release_metadata_agrees"] = False

    c["success_flag_consistent"] = bool(
        bool(meta.get("success")) == (path.parent.name == "success"))
    r["meta"] = {k: meta.get(k) for k in
                 ("success", "success_predicate_libero", "geometric_in_basket",
                  "lifted_mm", "object_init_sample", "gripper_speed",
                  "tcp_eef_pos", "zhixing_gripper_xml_sha16",
                  "object_init_fingerprint", "init_qpos_sha16",
                  "state_gripper_joint_names", "controller_output_max",
                  "teacher_gain_pos_m", "teacher_gain_rot_rad")}

    # ------------------------------------------------------------------ #
    # [R4] state 契约：最后两维必须是**主动关节** Left_1 / Right_1
    # ------------------------------------------------------------------ #
    gq = f["gripper_qpos"]
    names = meta.get("state_gripper_joint_names")
    idxs = meta.get("state_gripper_qpos_indices")
    if names is not None and idxs is not None:
        r["state_contract"] = {"names": names, "qpos_indices": idxs}
        c["state_tail_matches_declared_gripper_qpos"] = bool(
            np.allclose(st[:, -2:], gq, atol=1e-12))
        c["state_tail_matches_declared_qpos_indices"] = bool(
            np.allclose(st[:, -2:], qpos[:, idxs], atol=1e-12))
        c["state_joint_names_are_active"] = bool(
            len(names) == len(C.STATE_GRIPPER_JOINT_NAMES)
            and all(str(nm).endswith(w) for nm, w in
                    zip(names, C.STATE_GRIPPER_JOINT_NAMES)))
    else:
        # 旧数据没有契约字段：只能按"主动关节"独立判定
        r["state_contract"] = "MISSING（旧数据，无 state_gripper_joint_names）"
        c["state_tail_matches_declared_gripper_qpos"] = bool(
            np.allclose(st[:, -2:], gq, atol=1e-12))
        c["state_joint_names_are_active"] = False

    # ------------------------------------------------------------------ #
    # [R6] 动作尺度元数据：必须与真实控制器一致，且与教师增益分离
    # ------------------------------------------------------------------ #
    com = meta.get("controller_output_max")
    c["controller_output_max_recorded"] = bool(com is not None)
    c["controller_output_max_matches_osc"] = bool(
        com is not None and np.allclose(com, EXPECTED_OUTPUT_MAX, atol=1e-12))
    c["teacher_gain_separated"] = bool(
        meta.get("teacher_gain_pos_m") is not None
        and meta.get("teacher_gain_rot_rad") is not None)

    # ------------------------------------------------------------------ #
    # [R2] 首帧观测新鲜度：obs_step 必须等于 step_index
    # ------------------------------------------------------------------ #
    if "obs_step" in f.files and "step_index" in f.files:
        os_, si_ = f["obs_step"], f["step_index"]
        r["obs_step_vs_step_index_max_diff"] = int(np.abs(os_ - si_).max())
        c["frames_obs_in_sync"] = bool(np.array_equal(os_, si_))
        c["first_frame_obs_fresh"] = bool(int(os_[0]) == int(si_[0]))
    else:
        r["obs_step_vs_step_index_max_diff"] = None
        c["frames_obs_in_sync"] = False
        c["first_frame_obs_fresh"] = False

    # ------------------------------------------------------------------ #
    # 回放验证
    # ------------------------------------------------------------------ #
    if replay:
        import mujoco

        # --- [§2.3.1] 环境按**本条 episode 的元数据**重建，不再共用固定
        #     libero_object/task0 环境；同时核验资产/控制器/速度/相机指纹 ---
        spec = env_spec_from_meta(meta)
        r["env_spec"] = {"suite": spec[0], "task_id": spec[1],
                         "robot": spec[2], "gripper": spec[3]}
        # 速度必须在**建环境之前**注入，gripper.speed 是读环境变量的属性
        spd = meta.get("gripper_speed")
        if spd is not None:
            os.environ["ZHIXING_GRIPPER_SPEED"] = str(float(spd))
        if env_cache is not None:
            env = build_env_cached(env_cache, spec)
        elif replay_env is not None:
            env = replay_env
        else:
            raise RuntimeError("replay=True 但没有可用的环境（env_cache/replay_env 均为空）")

        # --- [R3-1] 严格恢复**环境配置**：夹爪速度必须来自元数据 ---
        rb0 = env.env.robots[0]
        r["replay_gripper_speed_applied"] = float(rb0.gripper.speed)
        c["replay_gripper_speed_matches_meta"] = bool(
            spd is None or abs(float(rb0.gripper.speed) - float(spd)) < 1e-12)

        env.seed(int(meta["seed"]))
        env.reset()
        # [§2.3.1] 指纹核对必须放在 reset **之后**：`robots[0].controller` 是
        # reset 时创建的，reset 之前读会拿到 None（或上一集的陈旧对象）。
        assert_env_matches_meta(env, meta, r)
        sim, m, d, gid = N.ids(env)
        iq = np.array(f["init_qpos"], dtype=float)
        iv = np.array(f["init_qvel"], dtype=float)

        rb = env.env.robots[0]
        arm_idx = np.array(rb._ref_joint_pos_indexes, dtype=int)
        grip_idx = np.array(rb._ref_gripper_joint_pos_indexes, dtype=int)
        oa = object_qpos_addr(m, meta.get("object"))
        r["replay_object_qpos_addr"] = oa
        eef_site = int(getattr(rb, "eef_site_id", -1))

        def restore():
            """把仿真恢复到 episode 第 0 帧的**完整物理+控制状态**。

            [复核 §2.3.2] 除 qpos/qvel/warmstart/夹爪累积命令外，还恢复
            执行器控制量与外力缓存；并在 `r` 里逐项记录"恢复了哪些"，
            便于判断首个分歧帧到底缺哪一项状态。
            """
            sim_, m_, d_, gid_ = N.ids(env)
            rbt = env.env.robots[0]
            d_.qpos[:] = iq
            d_.qvel[:] = iv
            restored = ["qpos", "qvel"]
            if "init_qacc_warmstart" in f.files:
                d_.qacc_warmstart[:] = f["init_qacc_warmstart"]
                restored.append("qacc_warmstart")
            if "init_ctrl" in f.files:
                d_.ctrl[:] = f["init_ctrl"]
                restored.append("ctrl")
            if "init_qfrc_applied" in f.files:
                d_.qfrc_applied[:] = f["init_qfrc_applied"]
                restored.append("qfrc_applied")
            if "init_xfrc_applied" in f.files:
                d_.xfrc_applied[:] = f["init_xfrc_applied"]
                restored.append("xfrc_applied")
            mujoco.mj_forward(m_, d_)
            ctrl = rbt.controller
            for meth in ("reset_goal", "update"):
                fn = getattr(ctrl, meth, None)
                if fn is None:
                    continue
                try:
                    fn(force=True)
                except TypeError:
                    fn()
            restored.append("controller_reset_goal/update")
            # [R3-2] 恢复夹爪**累积控制状态**
            if "init_gripper_current_action" in f.files:
                rbt.gripper.current_action = np.array(
                    f["init_gripper_current_action"], dtype=float).copy()
                r["gripper_current_action_source"] = "npz"
            else:
                nrep = int((meta.get("first_frame_obs_staleness") or {})
                           .get("prep_hold_steps", 45))
                ca = np.zeros(rbt.gripper.dof)
                for _ in range(nrep):   # 累加式：45 步开爪(-1) → 饱和
                    ca = np.clip(ca - rbt.gripper.speed * np.sign(-1.0), -1.0, 1.0)
                rbt.gripper.current_action = ca
                r["gripper_current_action_source"] = \
                    "reconstructed_from_prep_hold(%d)" % nrep
            restored.append("gripper.current_action")
            r["restored_state_items"] = restored
            r["gripper_current_action_used"] = [
                round(float(v), 6) for v in np.atleast_1d(rbt.gripper.current_action)]

        restore()

        # --- (0) 观测/状态一致性 + **滞后判别** ---
        # 说明：MuJoCo 的 `site_xpos` 是 `mj_forward` 的产物，而 `mj_step` 之后
        # `qpos` 已积分、`site_xpos` 仍对应本步 forward 的状态；因此同一帧里
        # `qpos` 与 `site_xpos` 会有**亚子步**级偏差（实测中位数 1e-6 m，
        # 运动最快处最大 7.7e-4 m）。这不是配对错误 —— 用滞后判别证明：
        # 若把 qpos 平移一帧再比，偏差会大一个数量级以上。
        def fk_dev(t, qq):
            d.qpos[:] = qq
            mujoco.mj_forward(m, d)
            return float(np.linalg.norm(d.site_xpos[eef_site] - f["eef_pos"][t]))

        dev0 = np.array([fk_dev(t, qpos[t]) for t in range(n)])
        devp = np.array([fk_dev(t, qpos[t + 1]) for t in range(n - 1)]) if n > 1 else None
        devm = np.array([fk_dev(t + 1, qpos[t]) for t in range(n - 1)]) if n > 1 else None
        r["obs_state_consistency"] = {
            "lag0_median_m": float(np.median(dev0)),
            "lag0_max_m": float(dev0.max()),
            "lag0_argmax_frame": int(dev0.argmax()),
            "lag_plus1_median_m": None if devp is None else float(np.median(devp)),
            "lag_minus1_median_m": None if devm is None else float(np.median(devm)),
        }
        r["obs_state_consistency_max_m"] = float(dev0.max())
        r["eef_pos_vs_grip_site_max_m"] = float(
            np.max(np.linalg.norm(f["eef_pos"] - f["grip_site"], axis=1)))
        c["eef_pos_equals_grip_site"] = bool(r["eef_pos_vs_grip_site_max_m"] < 1e-9)
        # 绝对一致性用**中位数**判定（避开亚子步尖峰）
        c["obs_state_consistent"] = bool(float(np.median(dev0)) < OBS_STATE_TOL_M)
        # **配对判别力**：lag0 必须显著优于平移一帧
        if devp is not None and devm is not None:
            base = max(float(np.median(dev0)), 1e-12)
            r["obs_state_lag_ratio"] = {
                "plus1_over_lag0": float(np.median(devp) / base),
                "minus1_over_lag0": float(np.median(devm) / base)}
            c["obs_state_lag0_better_than_shift"] = bool(
                np.median(devp) > 10 * base and np.median(devm) > 10 * base)

        restore()
        sim, m, d, gid = N.ids(env)
        r["setup_reproduce_max_qpos_err"] = float(np.max(np.abs(np.array(d.qpos) - iq)))
        c["setup_restores_init_state"] = bool(r["setup_reproduce_max_qpos_err"] < 1e-12)

        # --- (1) **全程**分类回放偏差 + 双相机关键帧核对 ---
        # [复核 §2.3.3] 原实现只把第 20–44 帧接入硬判据，后段偏差到数百毫米仍 PASS。
        # 现按**阶段**累计全程偏差，并把两个相机都在关键帧上核对。
        ph = json.loads(str(f["_phases_json"]))
        pid = f["phase_id"]
        phase_of = [ph[i] for i in pid] if len(pid) == n else ["?"] * n
        keyf = set(key_check_frames(pid, ph, n))
        r["render_key_frames"] = sorted(keyf)
        probe = [1, 2, 5, 10, 20, 40, 60, 100, 150, 200, 300, 400]
        prof = {}
        kinds = ("arm", "grip", "obj_pos", "obj_rot")
        mx = {k: 0.0 for k in kinds}
        early = {k: 0.0 for k in kinds}
        per_phase = {}
        render = {"agentview": {}, "wrist": {}}
        ok, nonfinite = True, 0
        first_div = None
        ARM_DIVERGE_RAD = 1e-3      # 判定"开始分歧"的臂关节角阈值
        for t in range(n):
            try:
                obs_t, _, _, _ = env.step(act[t])
            except Exception as e:                        # noqa: BLE001
                r["replay_error"] = str(e)
                ok = False
                break
            sim, m, d, gid = N.ids(env)
            if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
                nonfinite += 1
            if t + 1 >= n:
                continue
            e_arm = float(np.abs(d.qpos[arm_idx] - qpos[t + 1][arm_idx]).max())
            e_grip = float(np.abs(d.qpos[grip_idx] - qpos[t + 1][grip_idx]).max())
            e_op = e_or = 0.0
            if oa is not None:
                e_op = float(np.linalg.norm(d.qpos[oa:oa + 3] - qpos[t + 1][oa:oa + 3]))
                e_or = float(quat_angle(d.qpos[oa + 3:oa + 7], qpos[t + 1][oa + 3:oa + 7]))
            vals = {"arm": e_arm, "grip": e_grip, "obj_pos": e_op, "obj_rot": e_or}
            pb = per_phase.setdefault(phase_of[t + 1], {k: 0.0 for k in kinds})
            for k in kinds:
                v = vals[k]
                mx[k] = max(mx[k], v)
                pb[k] = max(pb[k], v)
                if EARLY_FROM <= t + 1 <= EARLY_FRAMES:
                    early[k] = max(early[k], v)
            # [§2.3.2] 首个分歧帧：臂关节角首次超过紧阈值的位置
            if first_div is None and e_arm > ARM_DIVERGE_RAD:
                first_div = {"frame": t + 1, "phase": phase_of[t + 1],
                             "arm_rad": round(e_arm, 9),
                             "grip_rad": round(e_grip, 9),
                             "obj_pos_m": round(e_op, 9),
                             "ncon": int(d.ncon)}
            if (t + 1) in probe:
                prof[str(t + 1)] = {k: round(vals[k], 6) for k in kinds}
            # [§2.3.4] 双相机：关键帧的渲染 vs 存档（掩膜内 + 全图）
            if (t + 1) in keyf:
                for cam, obskey, stored in (("agentview", "agentview_image", img),
                                            ("wrist", "robot0_eye_in_hand_image", wri)):
                    if obskey not in obs_t:
                        continue
                    ri = np.ascontiguousarray(obs_t[obskey][::-1, ::-1])
                    msk, plain, frac = masked_render_diff(ri, stored[t + 1], stored[t])
                    render[cam][str(t + 1)] = {"masked": round(msk, 3),
                                               "plain": round(plain, 3),
                                               "mask_frac": round(frac, 3)}

        # 回放末端原生成功谓词（复核 §2.3.3：回放必须要求最终原生成功）
        # API 说明：`ControlEnv.check_success()`（env_wrapper.py）才是对外入口，
        # 它内部调用 `self.env._check_success()`；内层 task env 上**没有**
        # `check_success`。旧实现只找 `env.env.check_success`，因此恒为 None。
        final_success, final_err, final_fn = None, None, None
        for label, obj, name in (("env.check_success", env, "check_success"),
                                 ("env.env.check_success", env.env, "check_success"),
                                 ("env.env._check_success", env.env, "_check_success")):
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    final_success = bool(fn())
                    final_fn = label
                    break
                except Exception as e:                    # noqa: BLE001
                    final_err = "%s: %s" % (label, e)
        r["replay_final_native_success_fn"] = final_fn

        r["replay_max_err_by_kind"] = {k: round(v, 9) for k, v in mx.items()}
        r["replay_early_max_err_by_kind"] = {k: round(v, 9) for k, v in early.items()}
        r["replay_phase_max_err"] = {
            p: {k: round(v, 9) for k, v in d_.items()} for p, d_ in per_phase.items()}
        r["replay_divergence_profile"] = prof
        r["replay_first_divergence"] = first_div
        r["replay_nonfinite_steps"] = int(nonfinite)
        r["replay_final_native_success"] = final_success
        if final_err:
            r["replay_final_native_success_error"] = final_err
        r["replay_render_vs_stored_image_255"] = render
        r["replay_render_max_masked"] = {
            cam: (max((v["masked"] for v in d_.values()), default=None))
            for cam, d_ in render.items()}
        r["replay_tolerances"] = {"source": ("configs/n4_audit_tolerances.json"
                                             if tolerances else "UNRECORDED"),
                                  "applied": tolerances}

        # --- (2) 配对滞后诊断（拟合残差）---
        # 注意：**该方法没有判别力**，只作诊断记录。原因是 OSC 底层控制器存在
        # 约一个控制步的响应滞后，位移与"上一步"指令的拟合优度天然接近甚至更好。
        # 自检（--selftest）用刻意可辨别的序列证实了这一点，因此**配对判据**
        # 交给上方的"回放渲染 vs 存档图像"：动作若错位一帧，重放轨迹就会偏离，
        # 渲染画面必然对不上。
        ph = json.loads(str(f["_phases_json"]))
        pid = f["phase_id"]
        free_ph = [ph.index(x) for x in ("settle", "approach", "descend") if x in ph]
        free = np.isin(pid, free_ph) if free_ph else np.ones(n, dtype=bool)
        disp = np.diff(f["grip_site"], axis=0)
        res = lag_residuals(act, disp, free[:-1] if len(free) == n else free)
        r["pairing_lag_residuals_diagnostic"] = res
        r["pairing_note"] = ("配对判据 = 回放渲染 vs 存档图像（见 "
                             "replay_render_matches_stored_image）；"
                             "残差法因控制器响应滞后无判别力，仅诊断。")

        # --- [§2.3.3] 硬判据：异常/非有限值/**全程**阶段误差/回放末端原生成功 ---
        c["replay_no_exception"] = bool(ok)
        c["replay_no_nonfinite"] = bool(nonfinite == 0)
        c["replay_free_space_arm_within_tol"] = bool(ok and early["arm"] < ARM_TOL_RAD)
        c["replay_free_space_gripper_within_tol"] = bool(ok and early["grip"] < GRIP_TOL_RAD)
        c["replay_final_native_success"] = bool(final_success is True)

        # 全程分阶段容差：**必须来自标定文件**。未标定一律标记为未标定并判 FAIL，
        # 不允许用代码里的宽松默认值把数据放过去（复核 §2.3.3 的核心诉求）。
        tol = tolerances or {}
        ra_max = r["replay_render_max_masked"].get("agentview")
        rw_max = r["replay_render_max_masked"].get("wrist")
        if tol:
            def _t(key):
                return float(tol[TOL_KEYS[key]])
            c["tolerances_calibrated"] = True
            c["phase_arm_within_tol"] = bool(ok and mx["arm"] <= _t("phase_arm_rad"))
            c["phase_gripper_within_tol"] = bool(ok and mx["grip"] <= _t("phase_grip_rad"))
            c["phase_obj_pos_within_tol"] = bool(ok and mx["obj_pos"] <= _t("phase_obj_pos_m"))
            c["phase_obj_rot_within_tol"] = bool(ok and mx["obj_rot"] <= _t("phase_obj_rot_rad"))
            c["render_agentview_within_tol"] = bool(
                ra_max is not None and ra_max <= _t("render_agentview_255"))
            c["render_wrist_within_tol"] = bool(
                rw_max is not None and rw_max <= _t("render_wrist_255"))
        else:
            r["tolerances_uncalibrated_reason"] = (
                "configs/n4_audit_tolerances.json 缺失或字段不全：全程容差未经"
                "同配置重复运行标定，按复核 §2.3.3 不得放行。"
                "先跑 --calibrate 生成，再重跑审计。")
            for k in ("tolerances_calibrated", "phase_arm_within_tol",
                      "phase_gripper_within_tol", "phase_obj_pos_within_tol",
                      "phase_obj_rot_within_tol", "render_agentview_within_tol",
                      "render_wrist_within_tol"):
                c[k] = False

    if selftest is not None:
        r["pairing_selftest"] = selftest
        c["pairing_detector_has_discriminating_power"] = bool(
            selftest.get("lag0_is_best"))

    r["pass"] = bool(all(c.values()))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(ROOT / "raw" / "success"))
    ap.add_argument("--also-failure", action="store_true")
    ap.add_argument("--replay", dest="replay", action="store_true", default=True)
    ap.add_argument("--no-replay", dest="replay", action="store_false")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--tag", default="n4_audit")
    # [§2.3.5] 对同一个正式入口注入 4 类负对照，逐一验证检测能力
    ap.add_argument("--negative-controls", action="store_true",
                    help="对同一条 episode 注入 4 类负对照并断言每种都被拒绝")
    ap.add_argument("--neg-file", default="",
                    help="负对照所用的 episode 文件名（默认取目录里第一条）")
    # [§2.3.3] 容差标定：同配置重复运行量离散度
    ap.add_argument("--calibrate", action="store_true",
                    help="重复回放同一条 episode，标定离散度并写 configs/n4_audit_tolerances.json")
    ap.add_argument("--repeats", type=int, default=CALIB_REPEATS_DEFAULT)
    a = ap.parse_args()

    dirs = [pathlib.Path(a.dir)]
    if a.also_failure:
        dirs.append(ROOT / "raw" / "failure")
    files = []
    for d in dirs:
        files += sorted(d.glob("*.npz"))
    if not files:
        print("[FATAL] 没有找到 episode")
        return 2

    env_cache = {}
    tolerances = load_tolerances()

    # ---- 负对照模式：只针对**同一条** episode，走同一个 check_episode ----
    if a.negative_controls:
        if not a.replay:
            print("[FATAL] 负对照必须开启回放")
            return 2
        target = next((p for p in files if a.neg_file and a.neg_file in p.name), files[0])
        print("=" * 78)
        print("[负对照] 目标 episode：%s" % target.name)
        results = {}
        clean = check_episode(target, replay=True, selftest=None, neg=None,
                              tolerances=tolerances, env_cache=env_cache)
        results["clean"] = {"pass": clean["pass"],
                            "failed": [k for k, v in clean["checks"].items() if not v]}
        print("  clean            pass=%-5s  未通过项=%s"
              % (clean["pass"], results["clean"]["failed"] or "无"))
        for mode in NEG_MODES:
            rr = check_episode(target, replay=True, selftest=None, neg=mode,
                               tolerances=tolerances, env_cache=env_cache)
            failed = [k for k, v in rr["checks"].items() if not v]
            results[mode] = {"pass": rr["pass"], "failed_checks": failed}
            print("  %-16s pass=%-5s  被拒绝=%s  触发判据=%s"
                  % (mode, rr["pass"], (not rr["pass"]), failed[:6]))
        all_rejected = all(not results[m]["pass"] for m in NEG_MODES)
        clean_ok = bool(results["clean"]["pass"])
        verdict = "PASS" if (all_rejected and clean_ok) else "FAIL"
        print("-" * 78)
        print("  全部负对照被拒绝=%s   正常样本通过=%s" % (all_rejected, clean_ok))
        print("NEGATIVE_CONTROL_VERDICT = %s" % verdict)
        if not clean_ok:
            print("  [注意] 正常样本未通过，说明还有未标定/未修复项，不能据此判定检测能力")
        out = {"node": "N4", "check": "audit_negative_controls",
               "target": target.name, "results": results,
               "all_negatives_rejected": bool(all_rejected),
               "clean_passed": clean_ok, "verdict": verdict}
        pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
        p = pathlib.Path(a.out) / ("%s_negative_controls.json" % a.tag)
        p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("[out] %s" % p)
        for e in env_cache.values():
            e.close()
        return 0 if verdict == "PASS" else 1

    # ---- 标定模式：重复回放同一条，量离散度 ----
    if a.calibrate:
        if not a.replay:
            print("[FATAL] 标定必须开启回放")
            return 2
        target = files[0]
        print("=" * 78)
        print("[标定] %s  重复 %d 次" % (target.name, a.repeats))
        runs = []
        for i in range(a.repeats):
            rr = check_episode(target, replay=True, tolerances=None,
                               env_cache=env_cache)
            runs.append(rr)
            print("  run %d: 全程 %s" % (i + 1, rr.get("replay_max_err_by_kind")))
        keys = ("arm", "grip", "obj_pos", "obj_rot")
        spread = {k: float(max(r_["replay_max_err_by_kind"][k] for r_ in runs)
                           - min(r_["replay_max_err_by_kind"][k] for r_ in runs))
                  for k in keys}
        rend = {cam: [r_["replay_render_max_masked"][cam] for r_ in runs
                      if r_["replay_render_max_masked"].get(cam) is not None]
                for cam in ("agentview", "wrist")}
        rend_spread = {cam: (float(max(v) - min(v)) if len(v) > 1 else 0.0)
                       for cam, v in rend.items()}
        # 容差 = 重复运行离散度 (+ 1e-9 数值下限)。若离散度为 0，说明回放是确定性的，
        # 那么任何偏差都来自**未恢复的状态**，不应被容差吸收 —— 这正是要找的东西。
        tol = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "calibrated_on": target.name, "repeats": a.repeats,
            "method": ("同一条 episode 同配置重复回放的离散度；离散度为 0 表示回放确定性，"
                       "此时偏差只应来自未恢复状态，不应放宽"),
            "repeat_spread": spread,
            "render_spread": rend_spread,
            "observed_max": {k: float(max(r_["replay_max_err_by_kind"][k] for r_ in runs))
                             for k in keys},
            TOL_KEYS["phase_arm_rad"]: spread["arm"] + 1e-9,
            TOL_KEYS["phase_grip_rad"]: spread["grip"] + 1e-9,
            TOL_KEYS["phase_obj_pos_m"]: spread["obj_pos"] + 1e-9,
            TOL_KEYS["phase_obj_rot_rad"]: spread["obj_rot"] + 1e-9,
            TOL_KEYS["final_obj_pos_m"]: spread["obj_pos"] + 1e-9,
            TOL_KEYS["render_agentview_255"]: rend_spread["agentview"] + 1e-9,
            TOL_KEYS["render_wrist_255"]: rend_spread["wrist"] + 1e-9,
        }
        TOL_FILE.parent.mkdir(parents=True, exist_ok=True)
        TOL_FILE.write_text(json.dumps(tol, indent=2, ensure_ascii=False), encoding="utf-8")
        print("-" * 78)
        print("[标定] 重复离散度 = %s" % spread)
        print("[标定] 渲染离散度 = %s" % rend_spread)
        print("[标定] 实测最大值 = %s" % tol["observed_max"])
        print("[标定] 写入 %s" % TOL_FILE)
        print("[标定] 注意：离散度为 0 时容差为 1e-9，未恢复状态造成的偏差**不会**被吸收")
        for e in env_cache.values():
            e.close()
        return 0

    # ---- 常规审计 ----
    reps = []
    selftest = None
    if a.replay:
        _suite, _tid, _rob, _grip = env_spec_from_meta(
            json.loads(str(load(files[0])["_meta_json"])))
        se_env = build_env_cached(env_cache, (_suite, _tid, _rob, _grip))
        print("[selftest] 用可辨别短动作序列校验配对检测器…", flush=True)
        selftest = run_pairing_selftest(se_env)
        print("[selftest] ±1 帧回放渲染误差 = %s  → 两个错位方向都被区分 = %s"
              % (selftest["render_err_by_lag"], selftest.get("both_shifts_distinguished")),
              flush=True)

    print("=" * 78)
    print("[N4-audit] 共 %d 个 episode  回放=%s  容差标定=%s"
          % (len(files), a.replay, "有" if tolerances else "无（未标定→不放行）"))
    for p in files:
        r = check_episode(p, replay=True, selftest=selftest,
                          neg=None, tolerances=tolerances, env_cache=env_cache)
        reps.append(r)
        bad = [k for k, v in r["checks"].items() if not v]
        print("\n%s  frames=%d  camΔ=%.2f  %s"
              % (r["file"], r["frames"], r.get("camera_mean_abs_diff", -1),
                 "✅全部通过" if not bad else "❌ 未通过：\n      - " + "\n      - ".join(bad)))
        if "replay_max_err_by_kind" in r:
            print("    全程最大偏差 = %s" % r["replay_max_err_by_kind"])
            print("    首个分歧帧 = %s" % r.get("replay_first_divergence"))
            print("    回放末端原生成功 = %s" % r.get("replay_final_native_success"))
            print("    渲染最大(掩膜) = %s" % r.get("replay_render_max_masked"))

    init_hashes = [r["meta"]["init_qpos_sha16"] for r in reps
                   if r["meta"].get("init_qpos_sha16")]
    obj_fps = [r["meta"]["object_init_fingerprint"] for r in reps
               if r["meta"].get("object_init_fingerprint")]
    samples = [tuple(r["meta"]["object_init_sample"]["xy"])
               for r in reps if r["meta"].get("object_init_sample")]
    div = {
        "n": len(reps),
        "unique_init_qpos": len(set(init_hashes)),
        "unique_object_fingerprints": len(set(obj_fps)),
        "unique_sampled_xy": len(set(samples)),
    }
    div["all_initializations_distinct"] = bool(
        len(reps) > 1 and len(set(init_hashes)) == len(init_hashes))
    if len(samples) > 1:
        arr = np.array(samples)
        dmin = float(np.min([np.linalg.norm(arr[i] - arr[j])
                             for i in range(len(arr)) for j in range(i + 1, len(arr))]))
        div["min_pairwise_init_distance_mm"] = round(dmin * 1000, 2)
        div["inits_spread_gt_8mm"] = bool(dmin > 0.008)

    allpass = all(r["pass"] for r in reps) and div["all_initializations_distinct"] \
        and bool(selftest is None or selftest.get("lag0_is_best"))
    # [复核 §2.3.6] 只做静态检查时**不得**输出完整验收 PASS
    verdict = "PASS" if (allpass and a.replay) else (
        "STATIC_PASS_ONLY" if (a.replay is False and all(r["pass"] for r in reps))
        else "FAIL")
    out = {"node": "N4", "check": "data_contract_audit", "replay": a.replay,
           "full_acceptance": bool(a.replay),
           "tolerances_calibrated": bool(tolerances),
           "selftest": selftest, "episodes": reps, "diversity": div,
           "n_pass": sum(1 for r in reps if r["pass"]), "n_total": len(reps),
           "verdict": verdict}
    if not a.replay:
        out["verdict_note"] = ("--no-replay 只做了静态契约检查，"
                               "**不构成完整数据验收**（复核 §2.3.6）")

    print("\n" + "=" * 78)
    print("[diversity] %s" % json.dumps(div, ensure_ascii=False))
    print("[verdict] %s   （%d/%d episode 通过契约检查）"
          % (out["verdict"], out["n_pass"], out["n_total"]))
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.out) / ("%s.json" % a.tag)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[out] %s" % p)
    for e in env_cache.values():
        e.close()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
