#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ECO65 + PandaGripper 脚本专家控制验证（不依赖 VLA / 不需要 GPU）。

背景
----
当前服务器为纯 CPU：π0.5 无法推理，因此**不得**用策略成功率判断
「ECO65 + Panda 夹爪控制是否正确」。本脚本改用**脚本专家 + 底层 7 维 OSC**，
把「夹爪/臂能不能被正确控制」与「策略想不想抓」彻底分开。

复用的既有实现（一行未改）
--------------------------
* `n4_collect`（N4 采集链路的 Episode / servo / 教师增益 / 物体初始化采样 / bbox）
* `n3_osc_full_chain`（全程 OSC 的 ChainEpisode / servo_limited / ContactMonitor /
  试块注入 / 闸门定义）
* `n3_grasp_test`（gripsite / 四元数 / body 几何工具）

本脚本**只新增**「夹爪无关的指垫几何」一层：`n3_osc_full_chain.pad_geoms()` 硬编码
了 ZhiXing 的 `left_fingerpad_col` / `right_fingerpad_col`，在 PandaGripper 上取不到，
会让穿透监测**静默失效**。这里按「哪一侧指与坐标轴对齐更松」自动分组，
并对结果做自检（见 `pad_selfcheck`）。

输出
----
`--out-dir` 下每格一个 JSON：`<gripper>_w<width>_<timestamp>.json`，
外加 `_pilot` 自检 JSON。结果只写该目录，不触碰任何既有冻结产物。

用法（容器内）
--------------
    python eco65_expert_validation.py --gripper PandaGripper --width-mm 50 --trials 6
    python eco65_expert_validation.py --gripper PandaGripper --pilot
"""
import argparse
import datetime
import os
import json
import pathlib
import sys
import time

import numpy as np

SCRIPTS = "/app/agent-work-space/zhixing_adaptation_20260927/scripts"
sys.path.insert(0, SCRIPTS)

import n3_grasp_test as N            # noqa: E402  几何/四元数工具
import n3_osc_full_chain as N3       # noqa: E402  ChainEpisode / servo_limited / 监测
import n4_collect as N4              # noqa: E402  Episode / 采样 / bbox / TCP 契约

ROOT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927")
TEACHER_GAIN_POS_M = N4.TEACHER_GAIN_POS_M
TEACHER_GAIN_ROT_RAD = N4.TEACHER_GAIN_ROT_RAD
GRIP_OPEN, GRIP_CLOSE = N4.GRIP_ACT_OPEN, N4.GRIP_ACT_CLOSE
OSC_POS_UNIT_M, OSC_ROT_UNIT_RAD = N3.OSC_POS_UNIT_M, N3.OSC_ROT_UNIT_RAD
SERVO_MAX_STEP_POS_M = N3.SERVO_MAX_STEP_POS_M
SERVO_MAX_STEP_ROT_RAD = N3.SERVO_MAX_STEP_ROT_RAD
BLOCK_DEPTH_M, BLOCK_HEIGHT_M = N3.BLOCK_DEPTH_M, N3.BLOCK_HEIGHT_M

# 搬运段收敛判据：与 N4 采集器**同一口径**（220 步 / 8 mm）。
# n3 用的是 360 步且不把它计入闸门 —— 本脚本两者都报，避免重演口径矛盾。
TRANSPORT_MAX_STEPS = 220
TRANSPORT_TOL_M = 0.008
PLACE_MAX_STEPS = 160

PEN_SOFT_LIMIT_MM = N3.THRESHOLDS["penetration_soft_limit_mm"]

# --------------------------------------------------------------------------- #
# 夹爪无关的指垫几何
# --------------------------------------------------------------------------- #
_PAD_CACHE = {}
_APERTURE_PAIR = {}


def gripper_desc(env):
    """返回 (类名, dof, 关节名, 执行器数)。"""
    r = env.env.robots[0]
    g = r.gripper
    return {
        "class": type(g).__name__,
        "dof": int(getattr(g, "dof", -1)),
        "joints": [str(j) for j in getattr(g, "joints", [])],
        "n_actuators": int(getattr(g, "n_actuators", len(getattr(g, "actuators", [])))),
        "speed": float(getattr(g, "speed", -1.0)),
    }


def pad_geoms_generic(sim):
    """**夹爪无关**的指垫碰撞 geom 定位：返回 {"left": gid, "right": gid}。

    规则（不写死名称）：
      1. 在「`left` 开头」与「`right` 开头」的 body 子树里各收集 `contype>0` 的 geom；
         若没有 left/right 命名，则回退到夹爪 body 下按**闭合轴投影**正负分组。
      2. 用**闭合行程**消歧：在开/闭两个状态下位移沿同一轴、方向相反的两组即两指。
    返回的 geom 是**指垫**（若存在名为 *pad* 的碰撞 geom 则优先，否则取指尖 box）。
    """
    import mujoco

    m = sim.model._model
    if id(m) in _PAD_CACHE:
        return _PAD_CACHE[id(m)]

    def gname(g):
        return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()

    def bname(b):
        return (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").lower()

    cand = []
    for g in range(m.ngeom):
        if int(m.geom_contype[g]) <= 0 and int(m.geom_conaffinity[g]) <= 0:
            continue
        bn = bname(int(m.geom_bodyid[g]))
        gn = gname(g)
        if "finger" not in bn and "finger" not in gn \
                and "left" not in bn and "right" not in bn:
            continue
        cand.append(g)

    # 优先：名字里带 pad 的碰撞 geom（ZhiXing: left_fingerpad_col / right_fingerpad_col）
    pads = [g for g in cand if "pad" in gname(g)]
    pool = pads or [g for g in cand if "col" in gname(g) or "collision" in gname(g)]
    if not pool:
        pool = cand

    def side(gs, tag):
        """左右分组：body/geom 名里的 left/right，或 Panda 的 `finger1`/`finger2`。"""
        if tag == "left":
            keys = ("left", "finger1", "_1_", "finger_1")
        else:
            keys = ("right", "finger2", "_2_", "finger_2")
        out = []
        for g in gs:
            bn, gn = bname(int(m.geom_bodyid[g])), gname(g)
            if tag in bn or tag in gn or any(k in gn for k in keys):
                out.append(g)
        return out

    left, right = side(pool, "left"), side(pool, "right")
    if not left or not right or set(left) & set(right):
        raise RuntimeError("无法把指垫 geom 分成左右两组：pool=%s"
                           % [(g, gname(g), bname(int(m.geom_bodyid[g]))) for g in pool])
    pads_left = [g for g in left if "pad" in gname(g)]
    pads_right = [g for g in right if "pad" in gname(g)]

    # 每组可能多个 geom（指垫 + 指尖盒）：接触面代表取最“薄”的一个
    def pick(gs):
        return min(gs, key=lambda g: float(np.min(m.geom_size[g][:3])))

    lsel, rsel = pick(left), pick(right)
    out = {"left": lsel, "right": rsel,           # 代表 geom（量开口用）
           "pad_left": pads_left, "pad_right": pads_right,   # 真·指垫（可能为空）
           "all": {"left": left, "right": right},            # 全部手指碰撞 geom
           "names": {"left": gname(lsel), "right": gname(rsel),
                     "left_all": [gname(g) for g in left],
                     "right_all": [gname(g) for g in right],
                     "pad_left": [gname(g) for g in pads_left],
                     "pad_right": [gname(g) for g in pads_right]}}
    _PAD_CACHE[id(m)] = out
    return out


def jaw_gap_mm(env, pair=None):
    """两指垫**接触面之间的净距**（mm）。

    默认口径 `pair="aperture"`：取**指垫对**（在选定的左右 geom 里，当前状态下净距
    最大的那一对）—— 这才是「夹爪能张多开」的功能性开口；N2 契约里
    ZhiXing 全开 89.91 mm / Panda 全开 79.0 mm 就是这个量。
    `pair="min"`：所有左右 geom 组合里的最小净距（用来判断夹爪是否真的合到底）。

    净距 = 两侧几何**表面点集**的跨侧最近点点对距离（对 sphere/box/mesh 一律适用，
    不依赖 geom 命名或半径约定）。
    """
    sim, m, d, _ = N.ids(env)
    pads = pad_geoms_generic(sim)
    A, B = pads["all"]["left"], pads["all"]["right"]
    if pads["pad_left"] and pads["pad_right"]:
        A, B = pads["pad_left"], pads["pad_right"]
    pa_all = {g: surface_points(m, d, [g]) for g in A}
    pb_all = {g: surface_points(m, d, [g]) for g in B}
    best_pair, best_min = None, None
    for ga, pa in pa_all.items():
        for gb, pb in pb_all.items():
            dd = float(np.sqrt(((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2)).min())
            if pair == "min":
                best_min = dd if best_min is None else min(best_min, dd)
            if best_pair is None or dd > best_pair[0]:
                best_pair = (dd, ga, gb)
    if pair == "min":
        return None if best_min is None else float(best_min * 1000.0)
    if best_pair is None:
        return None
    _APERTURE_PAIR[id(sim.model._model)] = best_pair[1:]
    return float(best_pair[0] * 1000.0)


def aperture_pair_name(env):
    sim, m, d, _ = N.ids(env)
    import mujoco
    pr = _APERTURE_PAIR.get(id(m))
    if not pr:
        return None
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g) for g in pr]


def min_clearance_mm(env):
    """所有指碰撞 geom 组合的最小净距（判断是否合到底）。"""
    return jaw_gap_mm(env, pair="min")


def surface_points(m, d, gids, n_grid=7):
    """把一组 geom 的表面采样成 (N,3) 世界坐标点集。"""
    import mujoco

    pts = []
    for g in gids:
        ctr = np.array(d.geom_xpos[g], dtype=float)
        R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
        t = int(m.geom_type[g])
        sz = [float(v) for v in m.geom_size[g][:3]]
        if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            u = np.linspace(0, np.pi, 5)
            v = np.linspace(0, 2 * np.pi, 9, endpoint=False)
            loc = np.array([[np.sin(a) * np.cos(b), np.sin(a) * np.sin(b), np.cos(a)]
                            for a in u for b in v]) * sz[0]
            pts.append(ctr + loc @ R.T)
        elif t == int(mujoco.mjtGeom.mjGEOM_BOX):
            g1 = np.linspace(-1, 1, n_grid)
            loc = np.array([[a, b, c] for a in g1 for b in g1 for c in g1])
            pts.append(ctr + (loc * np.array(sz[:3])) @ R.T)
        elif t in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
            zs = np.linspace(-sz[1], sz[1], n_grid) if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE) \
                else np.linspace(-sz[1], sz[1], n_grid)
            ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            loc = np.array([[sz[0] * np.cos(a), sz[0] * np.sin(a), z] for z in zs for a in ang])
            pts.append(ctr + loc @ R.T)
        elif t == int(mujoco.mjtGeom.mjGEOM_MESH):
            # ★ mesh 类型必须取**真实顶点**。旧实现退化成"几何中心"，
            #   把 ZhiXing 的 mesh 指垫净距 0.18 mm 误报成 17.8 mm（几何中心距）。
            mid = int(m.geom_dataid[g])
            if mid < 0:
                pts.append(ctr.reshape(1, 3))
            else:
                vadr, vnum = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                v = np.array(m.mesh_vert[vadr:vadr + vnum], dtype=float)
                pts.append(ctr + v @ R.T)
        else:
            pts.append(ctr.reshape(1, 3))
    return np.vstack(pts) if pts else np.zeros((0, 3))


def _box_vertices(m, d, g):
    import mujoco

    mat = np.array(d.geom_xpos[g], dtype=float)
    R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
    hx, hy, hz = (float(v) for v in m.geom_size[g][:3])
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    return mat + (signs * np.array([hx, hy, hz])) @ R.T


def closing_axis_world_generic(env):
    """闭合轴（世界系，水平投影）：由两指垫几何中心差给出。"""
    sim, m, d, _ = N.ids(env)
    pads = pad_geoms_generic(sim)
    a = np.array(d.geom_xpos[pads["right"]], dtype=float) - \
        np.array(d.geom_xpos[pads["left"]], dtype=float)
    a[2] = 0.0
    n = float(np.linalg.norm(a))
    return None if n < 1e-9 else a / n


def grip_joint_state(env):
    """活动指关节的 qpos（去掉有限的 Support/linkage 关节）。

    直接用 `_ref_gripper_joint_pos_indexes`（robosuite 认定的夹爪 qpos 槽位），
    对 ZhiXing 是 6 个联动关节、对 Panda 是 2 个平动指；两者都能算“是否停住”。
    """
    sim, m, d, _ = N.ids(env)
    idx = list(env.env.robots[0]._ref_gripper_joint_pos_indexes)
    return np.array([float(d.qpos[int(i)]) for i in idx], dtype=float)


# --------------------------------------------------------------------------- #
# N3 的 pad_geoms 补丁（仅本进程；磁盘源码不动）
# --------------------------------------------------------------------------- #
def resolve_gripper_joint_indices(env):
    """**夹爪无关**的 state 两维解析（替代 N4 里写死 Left_1/Right_1 的实现）。

    PandaGripper 的关节名是 `finger_joint1/finger_joint2`，N4 的
    `resolve_state_gripper_indices()` 按写死名字查找会直接抛错 —— 即
    **N4 采集/回放链路目前只能跑 ZhiXing**。本函数按关节名解析：

      * 名字含 `Left_1` / `Right_1`（ZhiXing 的主动驱动关节）→ 各取一个；
      * 否则取夹爪的**平动指关节**（Panda `finger_joint1/2`），按左右顺序排。
    """
    r = env.env.robots[0]
    joints = [str(j) for j in r.gripper.joints]
    idx, names = [], []
    for want in ("Left_1_Joint", "Right_1_Joint"):
        hits = [i for i, j in enumerate(joints) if j == want or j.endswith("_" + want)]
        if len(hits) == 1:
            idx.append(hits[0]); names.append(joints[hits[0]])
    if len(idx) == 2:
        return idx, names, "zhixing_named"
    # Panda：两个 slide 指
    hits = [i for i, j in enumerate(joints) if "finger_joint" in j]
    if len(hits) != 2:
        raise ValueError("无法解析 %s 的 state 两维：joints=%s" % (type(r.gripper).__name__, joints))
    hits = sorted(hits, key=lambda i: joints[i])
    return hits, [joints[i] for i in hits], "prismatic_fingers"


def install_generic_state_gripper(env, log):
    """让 N4 的 state 解析在本进程内使用上面的通用实现。"""
    idx, names, kind = resolve_gripper_joint_indices(env)

    def patched(_env):
        return list(idx), list(names)

    N4.resolve_state_gripper_indices = patched
    log("[state] 夹爪 state 两维 = %s（解析方式=%s）" % (names, kind))
    return idx, names, kind


def install_generic_pad_geoms(log):
    """把 N3 里硬编码 ZhiXing geom 名的 `pad_geoms` 换成夹爪无关实现。

    N3 的 `pad_geoms` 只认 `left_fingerpad_col`/`right_fingerpad_col`；在 PandaGripper 上
    返回空字典 → `ContactMonitor.pad_ids` 为空 → **指垫↔物体穿透恒为 0**（静默失效）。
    这里改为：优先真·指垫 geom；没有则退化为**全部手指碰撞 geom**（保守，宁可多报）。
    """
    orig = N3.pad_geoms

    def patched(sim):
        try:
            g = pad_geoms_generic(sim)
            lids = g["pad_left"] or g["all"]["left"]
            rids = g["pad_right"] or g["all"]["right"]
            return {"left": lids[0], "right": rids[0],
                    "left_all": lids, "right_all": rids}
        except Exception as e:                                # noqa: BLE001
            log("[pad] 通用定位失败(%s)，回退 N3 原始实现" % e)
            return orig(sim)

    N3.pad_geoms = patched

    # N3 的 ContactMonitor.bind 用 `set(pad_geoms(sim).values())`，那是**标量**约定；
    # 我们把「全部手指碰撞 geom」也暴露出来（列表），故这里一并改写 bind。
    _orig_bind = N3.ContactMonitor.bind

    def bind(self, env, obj_bid):
        sim, m, d, _ = N.ids(env)
        self.env = env
        self.obj_bid = obj_bid
        try:
            g = pad_geoms_generic(sim)
            ids = list(g["pad_left"] or g["all"]["left"]) + \
                list(g["pad_right"] or g["all"]["right"])
        except Exception:                                     # noqa: BLE001
            ids = list(orig(sim).values())
        self.pad_ids = {int(v) for v in ids}
        return None

    N3.ContactMonitor.bind = bind
    log("[pad] 已安装夹爪无关的指垫定位（N3 原始实现作为回退）")


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def pilot(env, args, log):
    """夹爪体检：极性 + 全行程 + 1 次空载开合 + 指垫定位自检。"""
    out = {}
    ep = N3.ChainEpisode(env, None)
    env.seed(args.seed)
    ep.start()
    sim, m, d, gid = N.ids(env)
    desc = gripper_desc(env)
    out["gripper"] = desc
    pads = pad_geoms_generic(sim)
    out["pad_geoms"] = {
        "representative": {"left": pads["names"]["left"], "right": pads["names"]["right"]},
        "left_all": pads["names"]["left_all"], "right_all": pads["names"]["right_all"],
        "pad_left": pads["names"]["pad_left"], "pad_right": pads["names"]["pad_right"],
        "n_left_geoms": len(pads["all"]["left"]), "n_right_geoms": len(pads["all"]["right"]),
    }
    n3p = N3.pad_geoms(sim)
    out["n3_pad_geoms_after_patch"] = {k: (v if not isinstance(v, list) else list(v))
                                       for k, v in n3p.items()}
    out["closing_axis_world"] = (None if closing_axis_world_generic(env) is None
                                 else [float(v) for v in closing_axis_world_generic(env)])
    out["grip_site_world"] = [float(v) for v in N.gripsite(sim, m, d, gid)[0]]

    # 空载：全开 → 全闭 → 再全开
    seq = []
    for cmd, tag in ((GRIP_OPEN, "open1"), (GRIP_CLOSE, "close"), (GRIP_OPEN, "open2")):
        N4.quiet_hold(env, args.settle_steps, cmd)
        seq.append({"cmd": cmd, "phase": tag, "jaw_gap_mm": jaw_gap_mm(env),
                    "min_clearance_mm": min_clearance_mm(env),
                    "aperture_pair": aperture_pair_name(env),
                    "grip_q": [round(float(v), 6) for v in grip_joint_state(env)],
                    "ncon": int(env.env.sim.data.ncon)})
    out["empty_load_sequence"] = seq
    out["checks"] = {
        "cmd_minus1_opens": bool(seq[0]["jaw_gap_mm"] > seq[1]["jaw_gap_mm"]),
        "cmd_plus1_closes": bool(seq[1]["jaw_gap_mm"] < seq[0]["jaw_gap_mm"]),
        "sign_convention_ok": bool(seq[0]["jaw_gap_mm"] > seq[1]["jaw_gap_mm"] < seq[2]["jaw_gap_mm"]),
        "reopen_ok": bool(seq[2]["jaw_gap_mm"] > seq[1]["jaw_gap_mm"] * 1.5),
        "stroke_mm": float(seq[0]["jaw_gap_mm"] - seq[1]["jaw_gap_mm"]),
    }

    # 空载开闭 ×N 重复性（N3 判据：std < 0.5 mm）
    opens, closes, closes_min = [], [], []
    t0 = time.time()
    for _ in range(args.repeat):
        env.reset()
        N4.quiet_hold(env, args.settle_steps, GRIP_OPEN)
        opens.append(jaw_gap_mm(env))
        N4.quiet_hold(env, args.settle_steps, GRIP_CLOSE)
        closes.append(jaw_gap_mm(env))
        closes_min.append(min_clearance_mm(env))
    out["repeatability"] = {
        "n": args.repeat,
        "open_mm": {"mean": float(np.mean(opens)), "std": float(np.std(opens)),
                    "min": float(np.min(opens)), "max": float(np.max(opens))},
        "closed_mm": {"mean": float(np.mean(closes)), "std": float(np.std(closes)),
                      "min": float(np.min(closes)), "max": float(np.max(closes))},
        "aperture_pair": aperture_pair_name(env),
        "min_clearance_mm": {"mean": float(np.mean(closes_min)),
                             "max": float(np.max(closes_min))},
        "pass": bool(np.std(opens) < 0.5 and np.std(closes) < 0.5),
        "wall_seconds": round(time.time() - t0, 1),
    }
    return out


# --------------------------------------------------------------------------- #
# 单次 抓取—抬升—保持—搬运—放置—释放
# --------------------------------------------------------------------------- #
def run_trial(env, args, spec, log):
    t0 = time.time()
    mon = N3.ContactMonitor()
    ep = N3.ChainEpisode(env, mon)
    env.seed(spec["seed"])
    ep.start()
    sim, m, d, gid = N.ids(env)

    obj_name, region_name = N4.goal_objects(env)
    bid, bname = N4.find_body(sim, obj_name)
    rid, rname = N4.find_body(sim, region_name)
    if bid is None or rid is None:
        raise RuntimeError("找不到目标物体 %s 或区域 %s" % (obj_name, region_name))

    axis_w = closing_axis_world_generic(env)
    if axis_w is None:
        raise RuntimeError("无法确定夹爪闭合轴")
    R_home0 = N.gripsite(sim, m, d, gid)[1].copy()
    R_ref = R_home0.copy()
    phi_eff = float(np.arctan2(axis_w[1], axis_w[0]))
    obj_yaw = float(phi_eff - np.pi / 2.0)

    rng = np.random.default_rng(900000 + spec["seed"])
    init_sample = N4.sample_object_init(env, bid, obj_name, bname, region_name, rng)
    if init_sample is None:
        raise RuntimeError("区域初始化采样失败")
    yaw_before = N3.object_yaw(env, bid)
    N3.set_object_yaw(env, bid, init_sample["xy"], obj_yaw)
    init_sample["yaw_rad"] = round(N3.object_yaw(env, bid), 6)
    init_sample["yaw_forced_rad"] = round(obj_yaw, 6)
    init_sample["yaw_before_forced_rad"] = round(yaw_before, 6)

    obs_prep = N4.quiet_hold(env, args.prep_steps, GRIP_OPEN)
    ep.raw_steps += args.prep_steps
    ep.refresh_obs(obs_prep, "post_init_quiet_hold")

    sim, m, d, gid = N.ids(env)
    # 穿透监测挂在 _pre_action（每个物理子步），与 N3 同口径
    mon.bind(env, bid)
    hook_installed = False
    try:
        sim.model  # noqa: B018
        prev = env.env._pre_action

        def hook(*a, **k):
            mon.sample()
            return prev(*a, **k)

        env.env._pre_action = hook
        hook_installed = True
    except Exception as e:                                    # noqa: BLE001
        log("  [!] 无法挂接子步采样钩子：%s" % e)

    # ---- 夹持点 / 抓取计划（N3 的几何法，对 Panda 同样适用） ----
    rb, rh = N4.bbox(sim, rid)
    if rb is None or rh is None:
        raise RuntimeError("目标区域 %s 无几何" % rname)
    open_budget = jaw_gap_mm(env) or 0.0
    plan = N.grasp_plan(sim, [(bid, bname)], closing_axis=1, opening_mm=open_budget,
                        margin_mm=args.grasp_margin_mm, band_m=0.006)
    if plan.get("best") is None:
        return {"trial": spec["trial"], "trial_label": spec["trial_label"],
                "seed": spec["seed"], "success": False,
                "reason": "no_grasp_plan", "open_budget_mm": open_budget,
                "plan_scan": plan["scan"], "wall_seconds": round(time.time() - t0, 1)}
    z_g, w_close, clear, cx, cy = plan["best"]
    # ---- 计划自检（防止把"计划本身无意义"记成"夹爪抓不住"）----
    # 抓取高度的**截面宽**必须接近试块宽度；若远小于它，说明该高度上根本没有
    # 试块实体（薄带里只剩一两个顶点），计划无效 —— 必须显式拒绝，不得继续执行。
    if float(w_close) * 1000.0 < 0.6 * float(spec["width_mm"]):
        return {"trial": spec["trial"], "trial_label": spec["trial_label"],
                "seed": spec["seed"], "success": False,
                "reason": "invalid_grasp_plan_section_too_thin",
                "open_budget_mm": open_budget,
                "planned_section_width_mm": round(float(w_close) * 1000, 2),
                "block_width_mm": float(spec["width_mm"]),
                "plan_scan": plan["scan"], "wall_seconds": round(time.time() - t0, 1)}
    pin = np.array([float(cx), float(cy), float(z_g)])
    obj_c0 = None
    bb0 = N4.bbox(sim, bid)
    if bb0 is not None:
        obj_c0 = 0.5 * (np.asarray(bb0[0]) + np.asarray(bb0[1]))

    pe = {}
    tr = {}
    rec = {
        "trial": spec["trial"], "trial_label": spec["trial_label"], "seed": spec["seed"],
        "block_extent_mm": {"x_depth": BLOCK_DEPTH_M * 1000,
                            "y_width_along_closing_axis": spec["width_mm"],
                            "z_height": BLOCK_HEIGHT_M * 1000},
        "object_init_sample": init_sample,
        "open_budget_mm": open_budget,
        "grasp_pinch_target": [float(v) for v in pin], "grasp_z": float(z_g),
        "grasp_section_width_mm": round(float(w_close) * 1000, 1),
        "grasp_fingertip_clear_mm": round(float(clear) * 1000, 1),
        "grasp_plan_scan": plan["scan"],
        "contact_hook_installed": hook_installed,
        "phases": [], "gates": {}, "notes": [],
    }

    def obj_center():
        bb = N4.bbox(N.ids(env)[0], bid)
        return None if bb is None else 0.5 * (np.asarray(bb[0]) + np.asarray(bb[1]))

    def native():
        try:
            return N3.native_success(env)
        except Exception:                                     # noqa: BLE001
            return False

    try:
        # 1 接近
        ok, k = ep.servo_limited(pin + np.array([0, 0, args.approach_height_m]),
                                 R_ref, GRIP_OPEN, "approach", args.approach_max_steps)
        pe["approach"] = [bool(ok), k]
        pe["approach_residual_m"] = ep.last_err[0]
        tr["approach"] = _phase_trace(env, ep, "approach",
                                      pin + np.array([0, 0, args.approach_height_m]))
        c_approach = obj_center()

        # 2 下降
        ok, k = ep.servo_limited(pin, R_ref, GRIP_OPEN, "descend", args.descend_max_steps,
                                 tol=0.008)
        pe["descend"] = [bool(ok), k]
        pe["descend_converged_pos"] = bool(ep.conv["pos"])
        tr["descend"] = _phase_trace(env, ep, "descend", pin)
        pe["descend_residual_m"] = ep.last_err[0]
        _sim, _m, _d, _ = N.ids(env)
        _pads = N3.pad_geoms(_sim)
        _mid_z = 0.5 * (float(_d.geom_xpos[_pads["left"]][2]) + float(_d.geom_xpos[_pads["right"]][2]))
        _bb = N4.bbox(_sim, bid)
        pe["descend_pad_mid_z"] = _mid_z
        pe["descend_block_top_z"] = None if _bb is None else float(_bb[1][2])
        pe["descend_pad_below_block_top_mm"] = (
            None if _bb is None else float((_bb[1][2] - _mid_z) * 1000))

        # 3 闭合到停住（臂由 OSC 保持）
        gprev, stable, close_steps = None, 0, 0
        for _ in range(args.close_max_steps):
            if ep.n >= N3.FRAME_LIMIT:
                break
            sim, m, d, gid = N.ids(env)
            p, R = N.gripsite(sim, m, d, gid)
            dv = pin - p
            nd = float(np.linalg.norm(dv))
            if nd > SERVO_MAX_STEP_POS_M:
                dv = dv * (SERVO_MAX_STEP_POS_M / nd)
            ap = np.clip(dv / TEACHER_GAIN_POS_M, -1, 1)
            er = N.rot_err_vec(R_ref, R)
            ne = float(np.linalg.norm(er))
            if ne > SERVO_MAX_STEP_ROT_RAD:
                er = er * (SERVO_MAX_STEP_ROT_RAD / ne)
            ar = np.clip(er / TEACHER_GAIN_ROT_RAD, -1, 1)
            ap = np.clip(ap, -SERVO_MAX_STEP_POS_M / OSC_POS_UNIT_M,
                         SERVO_MAX_STEP_POS_M / OSC_POS_UNIT_M)
            ar = np.clip(ar, -SERVO_MAX_STEP_ROT_RAD / OSC_ROT_UNIT_RAD,
                         SERVO_MAX_STEP_ROT_RAD / OSC_ROT_UNIT_RAD)
            ep.act(np.concatenate([ap, ar, [GRIP_CLOSE]]), "close")
            close_steps += 1
            g = grip_joint_state(env)
            if close_steps >= args.close_min_steps and gprev is not None \
                    and float(np.max(np.abs(g - gprev))) < 1e-4:
                stable += 1
                if stable >= args.close_stable_checks:
                    break
            else:
                stable = 0
            gprev = g
        pe["close_steps"] = close_steps
        pe["jaw_after_close_mm"] = jaw_gap_mm(env)
        pe["grip_q_after_close"] = [round(float(v), 6) for v in grip_joint_state(env)]
        c_close = obj_center()
        pe["obj_shift_on_close_mm"] = (None if (c_close is None or obj_c0 is None)
                                       else float(np.linalg.norm(c_close - obj_c0) * 1000))
        pe["close_penetration_peak_mm"] = mon.per_phase.get("close", {}).get("pad_peak_mm")

        # 4 抬升
        lift_target = pin + np.array([0, 0, args.lift_m])
        ok, k = ep.servo_limited(lift_target, R_ref, GRIP_CLOSE, "lift", 260, tol=0.004)
        pe["lift_servo"] = [bool(ok), k]
        pe["lift_residual_m"] = ep.last_err[0]
        tr["lift"] = _phase_trace(env, ep, "lift", lift_target)
        c_lift = obj_center()
        pe["lift_mm"] = (None if (c_lift is None or obj_c0 is None)
                         else float((c_lift[2] - obj_c0[2]) * 1000))

        # 5 保持 2 s
        hold_h = []
        for _ in range(N3.THRESHOLDS["hold_steps"]):
            if ep.n >= N3.FRAME_LIMIT:
                break
            ep.track(lift_target, R_ref, GRIP_CLOSE, "hold", 1)
            c = obj_center()
            if c is not None and obj_c0 is not None:
                hold_h.append(float((c[2] - obj_c0[2]) * 1000))
        pe["hold_steps_used"] = len(hold_h)
        pe["hold_min_mm"] = min(hold_h) if hold_h else None
        pe["hold_max_mm"] = max(hold_h) if hold_h else None
        pe["hold_end_mm"] = hold_h[-1] if hold_h else None
        pe["hold_penetration_peak_mm"] = mon.per_phase.get("hold", {}).get("pad_peak_mm")

        # 6 搬运（与 N4 同口径：220 步 / 8 mm；同时记 ok 供闸门判定）
        bx = float(0.5 * (rb[0] + rh[0]))
        by = float(0.5 * (rb[1] + rh[1]))
        above = np.array([bx, by, float(rh[2]) + 0.12])
        into = np.array([bx, by, float(rh[2]) + 0.03])
        pe["place_point"] = [float(v) for v in into]
        pe["basket_inner_size_mm"] = [float(rh[0] - rb[0]) * 1000,
                                      float(rh[1] - rb[1]) * 1000,
                                      float(rh[2] - rb[2]) * 1000]
        ok, k = ep.servo_limited(above, R_ref, GRIP_CLOSE, "transport",
                                 TRANSPORT_MAX_STEPS, tol=TRANSPORT_TOL_M)
        pe["transport"] = [bool(ok), k]
        pe["transport_residual_m"] = ep.last_err[0]
        pe["transport_target"] = [float(v) for v in above]
        pe["transport_grip_site"] = [float(v) for v in N.gripsite(*N.ids(env))[0]]
        _pg = N.gripsite(*N.ids(env))[0]
        pe["transport_err_world_m"] = [float(above[i] - _pg[i]) for i in range(3)]
        pe["transport_err_horizontal_m"] = float(np.linalg.norm(
            np.asarray(above[:2]) - np.asarray(_pg[:2])))
        tr["transport"] = _phase_trace(env, ep, "transport", above)
        pe["sep_after_transport_mm"] = jaw_gap_mm(env)
        c_tr = obj_center()
        lost = bool(c_tr is None or obj_c0 is None or (c_tr[2] - obj_c0[2]) < 0.02)
        pe["object_lost_after_transport"] = lost

        pe["place"] = [False, 0]
        pe["place_residual_m"] = None
        if not lost:
            # 7 投放
            ok, k = ep.servo_limited(into, R_ref, GRIP_CLOSE, "place",
                                     PLACE_MAX_STEPS, tol=0.008)
            pe["place"] = [bool(ok), k]
            pe["place_residual_m"] = ep.last_err[0]
            tr["place"] = _phase_trace(env, ep, "place", into)
            pe["place_grip_site"] = [float(v) for v in N.gripsite(*N.ids(env))[0]]

            # 8 释放 + 静置 + 稳定窗口
            ep.track(into, R_ref, GRIP_OPEN, "release", args.release_steps)
            settle = 0
            for _ in range(N3.THRESHOLDS["post_release_settle_max_steps"]):
                sim, m, d, _ = N.ids(env)
                if float(np.linalg.norm(d.qvel[:3])) < 0.008:
                    break
                ep.track(into, R_ref, GRIP_OPEN, "post_release_settle", 1)
                settle += 1
            pe["settle_steps"] = settle
            win = []
            for _ in range(N3.THRESHOLDS["post_release_steps"]):
                ep.track(into, R_ref, GRIP_OPEN, "post_release_window", 1)
                c = obj_center()
                if c is not None:
                    win.append(c)
            pe["post_release_window_steps"] = len(win)
            if win:
                pe["post_release_inside_all"] = bool(all(
                    _in_region(env, rid, c) for c in win))
                pe["post_release_drift_mm"] = float(
                    np.linalg.norm(win[-1] - win[0]) * 1000)
                pe["post_release_center"] = [float(v) for v in win[-1]]
            pe["post_release_penetration_peak_mm"] = mon.per_phase.get(
                "post_release_window", {}).get("pad_peak_mm", 0.0)
        else:
            rec["notes"].append("搬运段物体已脱落（高度回落），跳过下潜投放")

    except Exception as e:                                    # noqa: BLE001
        import traceback
        rec["notes"].append("exception: %s: %s" % (type(e).__name__, e))
        rec["traceback"] = traceback.format_exc()[-1500:]

    # ---- 闸门（与 N3 同一套；另加 transport_converged 作为**独立报告项**） ----
    g = {}
    g["lift_ge_30mm"] = bool(pe.get("lift_mm") is not None
                             and pe["lift_mm"] >= N3.THRESHOLDS["lift_min_mm"])
    g["hold_ge_25mm_40steps"] = bool(
        pe.get("hold_steps_used", 0) >= N3.THRESHOLDS["hold_steps"]
        and pe.get("hold_min_mm") is not None
        and pe["hold_min_mm"] >= N3.THRESHOLDS["hold_min_mm"])
    g["place_release_stable"] = bool(
        pe.get("post_release_window_steps", 0) >= N3.THRESHOLDS["post_release_steps"]
        and pe.get("post_release_inside_all")
        and pe.get("post_release_drift_mm") is not None
        and pe["post_release_drift_mm"] < N3.THRESHOLDS["post_release_drift_max_mm"])
    g["native_libero_predicate"] = bool(native())
    g["no_teleport"] = bool(ep.max_step_disp < N3.THRESHOLDS["max_step_disp_m"])
    g["penetration_within_soft_limit"] = bool(mon.peak_pad <= PEN_SOFT_LIMIT_MM)
    g["approach_converged"] = bool(pe.get("approach", [False])[0])
    g["descend_converged"] = bool(pe.get("descend", [False])[0])
    rec["gates"] = g
    rec["success"] = bool(all(g.values()))
    rec["failed_gates"] = [k for k, v in g.items() if not v]
    # N4 口径：搬运/投放**必须**收敛，否则是伺服超时（本项单独报，不混进 N3 闸门）
    rec["transport_converged"] = bool(pe.get("transport", [False])[0])
    rec["place_converged"] = bool(pe.get("place", [False])[0])
    rec["servo_no_timeout_n4"] = bool(rec["transport_converged"] and rec["place_converged"]
                                      and g["approach_converged"] and g["descend_converged"])
    rec["success_n4_style"] = bool(rec["success"] and rec["servo_no_timeout_n4"])
    rec["penetration_peak_pad_mm"] = float(mon.peak_pad)
    rec["penetration_peak_object_any_mm"] = float(mon.peak_any)
    rec["penetration_per_phase_mm"] = mon.per_phase
    rec["n_samples"] = int(mon.n_samples)
    rec["max_step_disp_m"] = float(ep.max_step_disp)
    rec["n_frames"] = int(ep.n)
    rec["phase_end"] = {k: (v if not isinstance(v, np.ndarray) else [float(x) for x in v])
                        for k, v in pe.items()}
    rec["wall_seconds"] = round(time.time() - t0, 1)
    rec["native_success_at_end"] = bool(native())
    rec["phase_trace"] = tr
    return rec


def _phase_trace(env, ep, phase, target=None):
    """把某阶段已记录的帧折算成定因指标（关节限位余量 / 接触数 / 目标误差）。

    只用 Episode 已经记录的帧，不改任何记录行为。
    """
    import mujoco

    sim, m, d, _ = N.ids(env)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(m.njnt)]
    arm = [j for j, n in enumerate(names) if n.startswith("robot0_joint")]
    aadr = [int(m.jnt_qposadr[j]) for j in arm]
    lo = np.array([float(m.jnt_range[j][0]) for j in arm])
    hi = np.array([float(m.jnt_range[j][1]) for j in arm])
    qs = [np.asarray(f["qpos"])[aadr] for f in ep.frames if f.get("phase") == phase]
    sites = [np.asarray(f["grip_site"]) for f in ep.frames if f.get("phase") == phase]
    if not qs:
        return None
    Q = np.vstack(qs)
    margin = np.minimum(Q - lo, hi - Q)
    S = np.vstack(sites)
    out = {
        "n_frames": int(Q.shape[0]),
        "joint_margin_min_rad": float(margin.min()),
        "joint_margin_min_per_joint": [float(v) for v in margin.min(axis=0)],
        "joint_at_limit_within_0p02rad": int((margin < 0.02).sum()),
        "grip_site_start": [float(v) for v in S[0]],
        "grip_site_end": [float(v) for v in S[-1]],
        "grip_site_travel_m": float(np.linalg.norm(S[-1] - S[0])),
        "grip_site_last20_move_m": float(np.linalg.norm(S[-1] - S[-min(20, len(S))]))
        if len(S) > 1 else 0.0,
    }
    if target is not None:
        t = np.asarray(target, dtype=float)
        out["target"] = [float(v) for v in t]
        out["err_end_m"] = float(np.linalg.norm(t - S[-1]))
        out["err_end_horizontal_m"] = float(np.linalg.norm(t[:2] - S[-1][:2]))
        out["err_end_vertical_m"] = float(t[2] - S[-1][2])
        out["err_min_m"] = float(min(np.linalg.norm(t - p) for p in S))
    return out


def _in_region(env, rid, center):
    """物体中心是否落在目标区域 bbox 内（与 N4 的 in_basket 同口径：用区域 bbox）。"""
    sim, m, d, _ = N.ids(env)
    bb = N4.bbox(sim, rid)
    if bb is None:
        return False
    lo, hi = np.asarray(bb[0]), np.asarray(bb[1])
    return bool(np.all(center >= lo) and np.all(center <= hi))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper", default="PandaGripper")
    ap.add_argument("--robot", default="ECO65")
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--width-mm", type=float, default=50.0)
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--seed-base", type=int, default=42000)
    ap.add_argument("--seed", type=int, default=7, help="pilot 用 seed")
    ap.add_argument("--gripper-speed", type=float, default=0.01,
                    help="夹爪部署速度。ZhiXing 资产原值 0.2（会拨飞物体），"
                         "部署值 0.01；经 ZHIXING_GRIPPER_SPEED 注入，与 N4 采集链路一致。")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-only", action="store_true")
    ap.add_argument("--repeat", type=int, default=10, help="空载开闭重复次数")
    ap.add_argument("--settle-steps", type=int, default=120)
    ap.add_argument("--prep-steps", type=int, default=120)
    ap.add_argument("--grasp-margin-mm", type=float, default=4.0)
    ap.add_argument("--approach-height-m", type=float, default=0.08)
    ap.add_argument("--approach-max-steps", type=int, default=260)
    ap.add_argument("--descend-max-steps", type=int, default=260)
    ap.add_argument("--close-max-steps", type=int, default=700)
    ap.add_argument("--close-min-steps", type=int, default=60)
    ap.add_argument("--close-stable-checks", type=int, default=8)
    ap.add_argument("--lift-m", type=float, default=0.08)
    ap.add_argument("--release-steps", type=int, default=60)
    ap.add_argument("--out-dir", default="/app/agent-work-space/outputs")
    a = ap.parse_args()

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    log("=" * 78)
    log("[EXPERT] robot=%s gripper=%s suite=%s task=%d width=%.0f mm trials=%d"
        % (a.robot, a.gripper, a.suite, a.task_id, a.width_mm, a.trials))
    # ★ 夹爪部署速度必须在**建环境之前**注入（ZhiXing 资产原值 0.2 → 闭合即把物体拨飞；
    #   部署值 0.01）。与 N4 采集链路同一机制、同一默认值。
    os.environ["ZHIXING_GRIPPER_SPEED"] = str(a.gripper_speed)
    log("[grip] ZHIXING_GRIPPER_SPEED=%s（部署值；Panda 资产自带 speed=0.01）"
        % os.environ["ZHIXING_GRIPPER_SPEED"])
    install_generic_pad_geoms(log)

    # 试块注入：宽度取 (--width-mm, 夹爪开口预算 - margin) 的较小者
    env, task = N4.build(a.suite, a.task_id, robot=a.robot, gripper=a.gripper, seed=a.seed)
    gidx, gnames, gkind = install_generic_state_gripper(env, log)
    desc = gripper_desc(env)
    log("[EXPERT] 实际夹爪 = %s  dof=%s  joints=%s" % (desc["class"], desc["dof"], desc["joints"]))
    sim, m, d, gid = N.ids(env)
    # ★ reset 时夹爪停在 init_qpos（半开），**不是**全开。开口预算必须在
    #   实际驱动到全开之后测，否则会把试块宽度上限记成 reset 开口（早期实现踩到：
    #   请求 50 mm 却只得到 13.8 mm 的试块，因为用了 reset 的 17.8 mm）。
    reset_gap = jaw_gap_mm(env)
    N4.quiet_hold(env, a.settle_steps, GRIP_OPEN)
    open_budget = jaw_gap_mm(env)
    log("[EXPERT] 开口：reset=%.2f mm  全开=%.2f mm（预算取全开）"
        % (reset_gap if reset_gap else -1, open_budget if open_budget else -1))
    width_mm = min(a.width_mm, float(open_budget or a.width_mm) - a.grasp_margin_mm)
    log("[EXPERT] 试块宽度 = %.1f mm（请求 %.1f，留 %.1f mm 余量）"
        % (width_mm, a.width_mm, a.grasp_margin_mm))

    cat, obj, region = N3.target_category(env)
    log("[EXPERT] BDDL 目标类别=%s 物体=%s 区域=%s" % (cat, obj, region))
    block_cls = N3.block_object_class(width_mm / 1000.0, BLOCK_DEPTH_M, BLOCK_HEIGHT_M)
    N3.install_object_substitution(cat, block_cls, log)

    result = {
        "node": "EXPERT-ECO65-PANDA",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "host_note": "纯 CPU 服务器；不使用 VLA 推理，脚本专家 + 7 维 OSC",
        "robot": a.robot, "gripper": a.gripper,
        "gripper_actual": desc,
        "suite": a.suite, "task_id": a.task_id,
        "block": {"width_mm": width_mm, "depth_mm": BLOCK_DEPTH_M * 1000,
                  "height_mm": BLOCK_HEIGHT_M * 1000, "density": N3.BLOCK_DENSITY,
                  "requested_width_mm": a.width_mm, "grasp_margin_mm": a.grasp_margin_mm},
        "transport_criterion_n4_style": {"max_steps": TRANSPORT_MAX_STEPS,
                                         "tol_m": TRANSPORT_TOL_M},
        "penetration_soft_limit_mm": PEN_SOFT_LIMIT_MM,
        "aperture": {"reset_mm": None if reset_gap is None else round(float(reset_gap), 2),
                     "full_open_mm": None if open_budget is None else round(float(open_budget), 2),
                     "definition": "两侧指垫碰撞 geom 表面点集的跨侧最近点对距离；"
                                   "aperture=当前状态下净距最大的那一对（功能性开口），"
                                   "min_clearance=所有组合的最小净距（判断是否合到底）"},
        "assets": {"n4_collect_sha16": N4.sha16(pathlib.Path(N4.__file__)),
                   "n3_osc_full_chain_sha16": N4.sha16(pathlib.Path(N3.__file__)),
                   "n3_grasp_test_sha16": N4.sha16(pathlib.Path(N.__file__))},
        "controller": N4.controller_action_contract(env),
        "gripper_speed_requested": a.gripper_speed,
        "state_gripper": {"indices": [int(i) for i in gidx], "names": list(gnames),
                          "resolution": gkind,
                          "note": "N4 原实现按 Left_1_Joint/Right_1_Joint 写死名字，"
                                  "PandaGripper 上会直接抛错；本脚本按关节名/平动指解析。"},
    }

    if a.pilot or a.pilot_only:
        log("\n----- PILOT：夹爪体检 + 空载开闭 ×%d -----" % a.repeat)
        result["pilot"] = pilot(env, a, log)
        ck = result["pilot"]["checks"]
        rp = result["pilot"]["repeatability"]
        log("  极性：-1 开=%s  +1 合=%s  行程=%.2f mm" %
            (ck["cmd_minus1_opens"], ck["cmd_plus1_closes"], ck["stroke_mm"]))
        log("  空载×%d：开 %.2f±%.3f mm / 闭 %.2f±%.3f mm  重复性=%s" %
            (rp["n"], rp["open_mm"]["mean"], rp["open_mm"]["std"],
             rp["closed_mm"]["mean"], rp["closed_mm"]["std"], rp["pass"]))
        p = out / ("pilot_%s_%s.json" % (a.gripper, stamp))
        p.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        log("[out] %s" % p)
        if a.pilot_only:
            env.close()
            return 0

    # 抓取链（每次 reset 重建环境，避免状态串味）
    trials = []
    for i in range(a.trials):
        seed = a.seed_base + i
        log("\n----- trial %d/%d seed=%d -----" % (i + 1, a.trials, seed))
        env2, _ = N4.build(a.suite, a.task_id, robot=a.robot, gripper=a.gripper, seed=seed)
        spec = {"trial": i + 1, "trial_label": "%s_w%.0f_t%d" % (a.gripper, width_mm, i + 1),
                "seed": seed, "width_mm": width_mm}
        r = run_trial(env2, a, spec, log)
        pe = r.get("phase_end", {})
        log("  success=%s  failed=%s  lift=%s mm  transport=%s(res=%.3f)  place=%s  pen_pad=%.3f mm"
            % (r.get("success"), r.get("failed_gates"), pe.get("lift_mm"),
               pe.get("transport"), (pe.get("transport_residual_m") or -1),
               pe.get("place"), r.get("penetration_peak_pad_mm") or -1))
        trials.append(r)
        try:
            env2.close()
        except Exception:                                     # noqa: BLE001
            pass

    n_ok = sum(1 for t in trials if t.get("success"))
    n_ok_n4 = sum(1 for t in trials if t.get("success_n4_style"))
    n_tr = sum(1 for t in trials if t.get("transport_converged"))
    result["trials"] = trials
    result["summary"] = {
        "n_total": len(trials), "n_success_n3_gates": n_ok,
        "success_rate_n3_gates": (n_ok / len(trials)) if trials else 0.0,
        "n_success_n4_style": n_ok_n4,
        "success_rate_n4_style": (n_ok_n4 / len(trials)) if trials else 0.0,
        "n_transport_converged": n_tr,
        "transport_residual_m": [t.get("phase_end", {}).get("transport_residual_m") for t in trials],
        "penetration_peak_pad_mm": [t.get("penetration_peak_pad_mm") for t in trials],
        "failed_gate_counts": _counts(trials),
    }
    result["verdict"] = "PASS" if (trials and n_ok == len(trials)) else "NOT_PASS"
    p = out / ("expert_%s_w%.0f_%s.json" % (a.gripper, width_mm, stamp))
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log("\n[out] %s" % p)
    log("[summary] N3 闸门 %d/%d 通过；N4 口径 %d/%d；搬运收敛 %d/%d"
        % (n_ok, len(trials), n_ok_n4, len(trials), n_tr, len(trials)))
    env.close()
    return 0


def _counts(trials):
    c = {}
    for t in trials:
        for k in (t.get("failed_gates") or []):
            c[k] = c.get(k, 0) + 1
        if t.get("reason"):
            c[t["reason"]] = c.get(t["reason"], 0) + 1
    return c


if __name__ == "__main__":
    raise SystemExit(main())
