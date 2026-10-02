#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZhiXingGripper 控制实现**问题诊断**（只读，不做任何纠错）。

目的：用脚本专家 + 底层 7 维 OSC，把 ZhiXing 夹爪控制实现当前存在的问题**找出来**，
不改代码、不改参数、不写 zhixing_adaptation_20260927/ 下任何受控文件。

四项测量
--------
A. 映射链体检：命令 u ∈ {-1,-0.5,0,+0.5,+1} 与 ctrl / Left_1 / 开口 的对应；
   给出**关节与开口的映射曲线**，看执行器行程是否被用满。
B. 动态闭合：持续 +1 到停住，与 A 的同构型对比（静态 vs 动态差异）；
   用命令 0 与命令 +1 的差值量化**闭合路径的可达开口**。
C. 重复性 ×10（两种 reset 状态各一轮）：开/闭净距的均值、极差、标准差。
D. 带载保持：把试块夹在两指之间闭合，再逐步加水平外力，测
   **最大可承受外力（把试块拽出所需的力）**、闭合时指垫净距、指垫位移、
   以及作动器力是否饱和。

输出：仅写 --out 指定的 JSON。
"""
import argparse
import datetime
import json
import os
import pathlib
import sys

import numpy as np

SCRIPTS = "/app/agent-work-space/zhixing_adaptation_20260927/scripts"
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, "/app/agent-work-space/outputs/scripts")
import eco65_expert_validation as EX   # noqa: E402  夹爪无关几何/开口度量
import n3_grasp_test as N              # noqa: E402
import n3_osc_full_chain as N3         # noqa: E402
import n4_collect as N4                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--gripper", default="ZhixingGripper")
ap.add_argument("--robot", default="ECO65")
ap.add_argument("--suite", default="libero_object")
ap.add_argument("--task-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--gripper-speed", type=float, default=0.01)
ap.add_argument("--repeat", type=int, default=10)
ap.add_argument("--width-mm", type=float, default=30.0)
ap.add_argument("--out", default="/app/agent-work-space/outputs/diag/zhixing_control.json")
a = ap.parse_args()

os.environ["ZHIXING_GRIPPER_SPEED"] = str(a.gripper_speed)
RESULT = {
    "node": "ZHIXING-CONTROL-DIAG",
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "note": "只读诊断：未修改任何代码/参数，未做纠错",
    "gripper": a.gripper, "robot": a.robot, "suite": a.suite, "task_id": a.task_id,
    "gripper_speed_env": os.environ["ZHIXING_GRIPPER_SPEED"],
}


def log(m):
    print(m, flush=True)


def state(env):
    """一次取齐：命令累积量 / Left_1 / 全 6 关节 / 开口 / 最小净距 / ctrl。"""
    import mujoco
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    g = r.gripper
    idx = [int(i) for i in r._ref_gripper_joint_pos_indexes]
    q = [float(d.qpos[i]) for i in idx]
    joints = [str(j) for j in g.joints]
    l1 = None
    for j, nm in enumerate(joints):
        if nm.endswith("Left_1_Joint"):
            l1 = float(q[j])
    ctrl = [float(v) for v in np.atleast_1d(d.ctrl[6])]
    return {
        "current_action": [float(v) for v in np.atleast_1d(g.current_action)],
        "Left_1": l1,
        "gripper_qpos": [round(v, 6) for v in q],
        "joint_names": joints,
        "aperture_mm": None if EX.jaw_gap_mm(env) is None else round(EX.jaw_gap_mm(env), 3),
        "min_clearance_mm": (None if EX.min_clearance_mm(env) is None
                             else round(EX.min_clearance_mm(env), 3)),
        "ctrl": ctrl,
        "ncon": int(d.ncon),
        "actuator_force": [float(v) for v in np.atleast_1d(d.actuator_force)],
        "qfrc_actuator_gripper": [float(d.qfrc_actuator[i]) for i in idx],
    }


def step_raw(env, cmd):
    """走真实 env.step；只清 done 标志（与 N3 ChainEpisode 同一手法），
    不改任何物理量，以便在一次 episode 内继续做纯夹爪测量。"""
    try:
        env.env.done = False
    except Exception:                                          # noqa: BLE001
        pass
    return env.step(np.concatenate([np.zeros(6), [cmd]]))


def hold(env, cmd, steps):
    for _ in range(steps):
        step_raw(env, cmd)
    return state(env)


def build_env():
    env, task = N4.build(a.suite, a.task_id, robot=a.robot, gripper=a.gripper, seed=a.seed)
    EX.install_generic_pad_geoms(lambda m: None)
    EX.install_generic_state_gripper(env, lambda m: None)
    return env


def main():
    env = build_env()
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    g = r.gripper
    RESULT["gripper_info"] = {
        "class": type(g).__name__, "dof": int(g.dof), "speed": float(g.speed),
        "joints": [str(j) for j in g.joints],
        "n_actuators": int(getattr(g, "n_actuators", -1)),
    }
    # 作动器与 XML 限位（直接从模型读，不引用文档）
    import mujoco
    ainfo = []
    for i in range(m.nu):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        jid = int(m.actuator_trnid[i][0])
        jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        ainfo.append({"i": i, "name": nm, "joint": jn,
                      "ctrlrange": [float(v) for v in m.actuator_ctrlrange[i]],
                      "forcelimited": bool(m.actuator_forcelimited[i]),
                      "forcerange": [float(v) for v in m.actuator_forcerange[i]],
                      "gainprm": [float(v) for v in m.actuator_gainprm[i][:3]],
                      "biasprm": [float(v) for v in m.actuator_biasprm[i][:3]]})
    RESULT["actuators"] = ainfo
    # 关节限位
    jl = {}
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm:
            jl[nm] = {"range": [float(v) for v in m.jnt_range[j]],
                      "limited": bool(m.jnt_limited[j])}
    RESULT["joint_limits"] = jl

    # ---------------- A. 命令 → ctrl → Left_1 → 开口 ----------------
    log("----- A. 静态映射链（每个命令收敛到停住） -----")
    env.reset()
    N4.quiet_hold(env, 400, -1.0)          # 先到全开
    env.reset()
    hold(env, -1.0, 400)
    A = []
    seq = [(-1.0, "open_sat"), (-0.5, "half_open"), (0.0, "zero_cmd"),
           (0.5, "half_close"), (1.0, "close_sat")]
    for cmd, tag in seq:
        st = hold(env, cmd, 400)
        st["cmd"] = cmd
        st["tag"] = tag
        A.append(st)
        log("  cmd=%+.1f %-11s ctrl=%-9s Left_1=%+.4f  aperture=%7.3f mm  min_clear=%7.3f mm  |q|=%s"
            % (cmd, tag, st["ctrl"], st["Left_1"] or 0, st["aperture_mm"] or -1,
               st["min_clearance_mm"] or -1,
               [round(v, 3) for v in st["gripper_qpos"]]))
    RESULT["A_command_map"] = A

    # 逐 ctrl 扫描：把 ctrl 直接钉在一串值上，得到 关节角 → 开口 的映射曲线
    # （打桩的是**实例方法**，源码文件零改动；测完恢复原方法）
    log("----- A2. ctrl → Left_1 → 开口 曲线 -----")
    g_real = r.gripper
    orig_fmt = g_real.format_action

    def _make_stub(value):
        def _stub(action):
            return np.array([value], dtype=float)
        return _stub

    curve = []
    for l1 in [round(v, 4) for v in np.linspace(0.0, -0.91, 14)]:
        env.reset()
        hold(env, -1.0, 300)
        g_real.format_action = _make_stub(l1)
        for _ in range(300):
            step_raw(env, 1.0)
        st = state(env)
        st["ctrl_cmd"] = l1
        curve.append(st)
    g_real.format_action = orig_fmt
    RESULT["A2_ctrl_to_aperture"] = curve
    for row in curve:
        log("  ctrl=%+.3f  Left_1=%+.4f  aperture=%8.3f mm  min_clear=%8.3f mm"
            % (row["ctrl_cmd"], row["Left_1"] or 0, row["aperture_mm"] or -1,
               row["min_clearance_mm"] or -1))

    # ---------------- B. 动态闭合 / 打开 轨迹 ----------------
    log("----- B. 动态：reset → +1 到停住 → −1 到停住 -----")
    env.reset()
    traj = []
    for i in range(600):
        step_raw(env, 1.0)
        if i % 25 == 0 or i < 5:
            st = state(env)
            traj.append({"i": i, "Left_1": st["Left_1"], "aperture_mm": st["aperture_mm"],
                         "min_clearance_mm": st["min_clearance_mm"],
                         "current_action": st["current_action"][0], "ctrl": st["ctrl"][0]})
    close_end = state(env)
    traj2 = []
    for i in range(600):
        step_raw(env, -1.0)
        if i % 25 == 0 or i < 5:
            st = state(env)
            traj2.append({"i": i, "Left_1": st["Left_1"], "aperture_mm": st["aperture_mm"],
                          "current_action": st["current_action"][0], "ctrl": st["ctrl"][0]})
    open_end = state(env)
    RESULT["B_dynamic"] = {
        "close_traj": traj, "open_traj": traj2,
        "close_end": close_end, "open_end": open_end,
        "closed_aperture_mm": close_end["aperture_mm"],
        "open_aperture_mm": open_end["aperture_mm"],
        "stroke_mm": (None if None in (close_end["aperture_mm"], open_end["aperture_mm"])
                      else round(open_end["aperture_mm"] - close_end["aperture_mm"], 3)),
        "closed_Left_1": close_end["Left_1"], "open_Left_1": open_end["Left_1"],
    }
    log("  合到底：Left_1=%+.4f aperture=%s mm" % (close_end["Left_1"] or 0, close_end["aperture_mm"]))
    log("  全开：  Left_1=%+.4f aperture=%s mm" % (open_end["Left_1"] or 0, open_end["aperture_mm"]))
    log("  行程 = %s mm" % RESULT["B_dynamic"]["stroke_mm"])

    # ---------------- C. 重复性 ×N（两种起始状态） ----------------
    log("----- C. 空载开闭 ×%d（两种起始状态） -----" % a.repeat)
    reps = {}
    for tag, prep in (("from_reset", None), ("from_full_open", -1.0)):
        opens, closes, mins = [], [], []
        for _ in range(a.repeat):
            env.reset()
            if prep is not None:
                hold(env, prep, 400)
            hold(env, -1.0, 400)
            s_open = state(env)
            hold(env, 1.0, 400)
            s_close = state(env)
            opens.append(s_open["aperture_mm"])
            closes.append(s_close["aperture_mm"])
            mins.append(s_close["min_clearance_mm"])
        reps[tag] = {
            "n": a.repeat,
            "open_mm": {"mean": float(np.mean(opens)), "std": float(np.std(opens)),
                        "min": float(np.min(opens)), "max": float(np.max(opens)),
                        "range": float(np.max(opens) - np.min(opens))},
            "closed_mm": {"mean": float(np.mean(closes)), "std": float(np.std(closes)),
                          "min": float(np.min(closes)), "max": float(np.max(closes)),
                          "range": float(np.max(closes) - np.min(closes))},
            "closed_min_clearance_mm": float(np.mean(mins)),
            "pass_n3_criterion": bool(np.std(opens) < 0.5 and np.std(closes) < 0.5),
        }
        log("  [%s] 开 %.3f±%.3f (极差 %.3f) / 闭 %.3f±%.3f (极差 %.3f)  判据(<0.5mm)=%s"
            % (tag, reps[tag]["open_mm"]["mean"], reps[tag]["open_mm"]["std"],
               reps[tag]["open_mm"]["range"], reps[tag]["closed_mm"]["mean"],
               reps[tag]["closed_mm"]["std"], reps[tag]["closed_mm"]["range"],
               reps[tag]["pass_n3_criterion"]))
    RESULT["C_repeatability"] = reps

    # ---------------- D. 带载保持：最大可承受外力 ----------------
    log("----- D. 带载：把试块夹住后逐步加水平外力，测拽出所需的力 -----")
    width_mm = min(a.width_mm, float(EX.jaw_gap_mm(env) or a.width_mm) - 4.0)
    # 重开一个环境做带载（避免前面的姿态残留）
    try:
        env.close()
    except Exception:                                          # noqa: BLE001
        pass
    env = build_env()
    sim, m, d, gid = N.ids(env)
    cat, obj, region = N3.target_category(env)
    block_cls = N3.block_object_class(width_mm / 1000.0, N3.BLOCK_DEPTH_M, N3.BLOCK_HEIGHT_M)
    env.reset()
    N3.install_object_substitution(cat, block_cls, log)
    env.close()
    env = build_env()
    sim, m, d, gid = N.ids(env)
    env.reset()
    N4.quiet_hold(env, 200, -1.0)
    sim, m, d, gid = N.ids(env)
    bid, bname = N4.find_body(sim, obj)
    axis = EX.closing_axis_world_generic(env)
    p_grip = np.array(N.gripsite(sim, m, d, gid)[0], dtype=float)
    # 把试块中心摆到夹持点上（宽 60 mm 沿闭合轴，指垫纵深够）
    c = None
    if bid is not None:
        import mujoco
        qadr, _ = N4.free_joint_adr(env, bid)
        d.qpos[qadr:qadr + 3] = np.array([p_grip[0], p_grip[1], p_grip[2]])
        d.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
        mujoco.mj_forward(m, d)
        bb = N4.bbox(sim, bid)
        c = None if bb is None else 0.5 * (np.asarray(bb[0]) + np.asarray(bb[1]))
    RESULT["D_load_setup"] = {
        "object": bname, "block_width_mm": width_mm,
        "grip_site": [float(v) for v in p_grip],
        "closing_axis": None if axis is None else [float(v) for v in axis],
        "object_center_after_place": None if c is None else [float(v) for v in c],
        "object_mass_kg": None,
    }
    # 闭合
    st = hold(env, 1.0, 400)
    RESULT["D_load_setup"]["after_close"] = st
    mass = None
    if bid is not None:
        mass = float(sum(m.body_mass[b] for b in range(m.nbody) if int(m.body_rootid[b]) == int(m.body_rootid[bid])))
    RESULT["D_load_setup"]["object_mass_kg"] = mass
    log("  闭合后：Left_1=%+.4f aperture=%s mm min_clear=%s mm 物体质量=%.4f kg"
        % (st["Left_1"] or 0, st["aperture_mm"], st["min_clearance_mm"], mass or -1))

    # 逐步加水平外力（沿闭合轴方向），找拽出阈值
    pulls = []
    pulled_out = None
    c0 = None if bid is None else 0.5 * (np.asarray(N4.bbox(sim, bid)[0]) +
                                         np.asarray(N4.bbox(sim, bid)[1]))
    for f in [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0]:
        if bid is None:
            break
        d.xfrc_applied[bid] = 0.0
        if axis is not None:
            d.xfrc_applied[bid][:3] = np.asarray(axis, dtype=float) * f
        for _ in range(200):
            step_raw(env, 1.0)
            d.xfrc_applied[bid][:3] = np.asarray(axis, dtype=float) * f
        bb = N4.bbox(sim, bid)
        c = None if bb is None else 0.5 * (np.asarray(bb[0]) + np.asarray(bb[1]))
        slip = None if (c is None or c0 is None) else float(np.linalg.norm(c - c0) * 1000)
        stp = state(env)
        row = {"force_N": f, "slip_mm": slip, "aperture_mm": stp["aperture_mm"],
               "Left_1": stp["Left_1"], "actuator_force": stp["actuator_force"]}
        pulls.append(row)
        log("  外力 %5.1f N → 滑移 %s mm  aperture=%s mm  actuator_force=%s"
            % (f, None if slip is None else round(slip, 3), stp["aperture_mm"],
               [round(v, 3) for v in stp["actuator_force"]]))
        if slip is not None and slip > 20.0 and pulled_out is None:
            pulled_out = f
            break
    if bid is not None:
        d.xfrc_applied[bid] = 0.0
    RESULT["D_load"] = {"pulls": pulls, "pull_out_force_N": pulled_out,
                        "note": "外力沿闭合轴施加在物体 body 上；滑移 >20 mm 视为拽出"}

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RESULT, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log("[out] %s" % out)
    try:
        env.close()
    except Exception:                                          # noqa: BLE001
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
