#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZhiXing 夹爪：握持力 / 闭合刚度 快速探针（只读，不做纠错）。

D1 空载闭合刚度：把 ctrl 逐步压向闭合端，记录 Left_1、指垫净距、
   作动器力矩 qfrc_actuator，找**卡住点**（净距不再变小的那个 ctrl）。
D2 握持力：把试块摆到两指之间**空载闭合**，再**直接**在物体 body 上施加
   沿闭合轴的水平外力（不碰 free joint，避免弄坏仿真），找把试块拽出所需的力。
D3 对照：PandaGripper 同口径跑一遍（若 --compare-panda）。

输出：--out 指定 JSON。
"""
import argparse
import datetime
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
sys.path.insert(0, "/app/agent-work-space/outputs/scripts")
import eco65_expert_validation as EX   # noqa: E402
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
ap.add_argument("--width-mm", type=float, default=30.0)
ap.add_argument("--pull-max-n", type=float, default=60.0)
ap.add_argument("--skip-d1", action="store_true")
ap.add_argument("--open-steps", type=int, default=150)
ap.add_argument("--close-steps", type=int, default=200)
ap.add_argument("--pull-steps", type=int, default=60)
ap.add_argument("--out", default="/app/agent-work-space/outputs/diag/zhixing_grip_force.json")
a = ap.parse_args()

os.environ["ZHIXING_GRIPPER_SPEED"] = str(a.gripper_speed)
RES = {"node": "ZHIXING-GRIP-FORCE", "gripper": a.gripper, "robot": a.robot,
       "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
       "note": "只读：未修改任何代码/参数"}


def log(m):
    print(m, flush=True)


def step_raw(env, cmd):
    try:
        env.env.done = False
    except Exception:                                          # noqa: BLE001
        pass
    return env.step(np.concatenate([np.zeros(6), [cmd]]))


def snap(env):
    import mujoco
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    g = r.gripper
    idx = [int(i) for i in r._ref_gripper_joint_pos_indexes]
    joints = [str(j) for j in g.joints]
    q = [float(d.qpos[i]) for i in idx]
    l1 = next((q[j] for j, nm in enumerate(joints) if nm.endswith("Left_1_Joint")), None)
    return {
        "current_action": [float(v) for v in np.atleast_1d(g.current_action)],
        "Left_1": l1,
        "gripper_qpos": [round(v, 6) for v in q],
        "aperture_mm": None if EX.jaw_gap_mm(env) is None else round(EX.jaw_gap_mm(env), 3),
        "ctrl": [float(v) for v in np.atleast_1d(d.ctrl[6])],
        "nc_actuator_force": [float(v) for v in np.atleast_1d(d.actuator_force)],
        "qfrc_actuator": [round(float(d.qfrc_actuator[i]), 4) for i in idx],
        "ncon": int(d.ncon),
    }


def build_env(gripper=None):
    env, task = N4.build(a.suite, a.task_id, robot=a.robot,
                         gripper=gripper or a.gripper, seed=a.seed)
    EX.install_generic_pad_geoms(lambda m: None)
    EX.install_generic_state_gripper(env, lambda m: None)
    return env


def d1_closing_stiffness(env):
    """把 ctrl 逐步压向闭合端：找净距不再变化的 ctrl（卡住点）。"""
    import mujoco
    sim, m, d, gid = N.ids(env)
    g = env.env.robots[0].gripper
    orig = g.format_action
    rows = []
    for ctrl in [0.0, -0.10, -0.20, -0.30, -0.40, -0.50, -0.60, -0.70, -0.80, -0.91]:
        env.reset()
        for _ in range(300):
            step_raw(env, -1.0)                # 先全开
        g.format_action = (lambda v: (lambda act: np.array([v], dtype=float)))(ctrl)
        for i in range(300):
            step_raw(env, 1.0)
            if i % 100 == 0:
                pass
        g.format_action = orig
        s = snap(env)
        s["ctrl_forced"] = ctrl
        rows.append(s)
        log("  ctrl=%+.3f → Left_1=%+.4f  净距=%8.3f mm  qfrc_act=%s  接触=%d"
            % (ctrl, s["Left_1"] or 0, s["aperture_mm"] or -1,
               s["qfrc_actuator"][:1], s["ncon"]))
    return rows


def d2_grip_force(env, width_mm):
    """空载闭合后，直接在物体上施加沿闭合轴的外力，找拽出阈值。"""
    import mujoco
    env.reset()
    for _ in range(a.open_steps):
        step_raw(env, -1.0)                    # 全开
    sim, m, d, gid = N.ids(env)
    cat, obj, region = N3.target_category(env)
    bid, bname = N4.find_body(sim, obj)
    axis = EX.closing_axis_world_generic(env)
    p_grip = np.array(N.gripsite(sim, m, d, gid)[0], dtype=float)
    out = {"object": bname, "block_width_mm": width_mm,
           "closing_axis": None if axis is None else [float(v) for v in axis],
           "grip_site": [float(v) for v in p_grip]}
    # 让物体落在两指之间（物体已经在抓取区域内就不再挪动，避免破坏 free joint）
    bb = N4.bbox(sim, bid)
    c = None if bb is None else 0.5 * (np.asarray(bb[0]) + np.asarray(bb[1]))
    out["object_center_at_reset"] = None if c is None else [float(v) for v in c]
    out["object_center_offset_to_grip_site_mm"] = (
        None if c is None else [float((c[i] - p_grip[i]) * 1000) for i in range(3)])
    mass = float(sum(m.body_mass[b] for b in range(m.nbody)
                     if int(m.body_rootid[b]) == int(m.body_rootid[bid])))
    out["object_mass_kg"] = mass

    # 空载闭合到 ctrl 端点
    for _ in range(a.close_steps):
        step_raw(env, 1.0)
    s0 = snap(env)
    out["after_close"] = s0
    bb = N4.bbox(sim, bid)
    c0 = None if bb is None else 0.5 * (np.asarray(bb[0]) + np.asarray(bb[1]))
    out["object_center_after_close"] = None if c0 is None else [float(v) for v in c0]
    log("  闭合后 净距=%.3f mm Left_1=%+.4f 物心偏移=%s mm 质量=%.4f kg"
        % (s0["aperture_mm"] or -1, s0["Left_1"] or 0,
           out["object_center_offset_to_grip_site_mm"], mass))

    pulls = []
    pull_out = None
    force = 1.0
    while force <= a.pull_max_n:
        if axis is None:
            break
        fvec = np.asarray(axis, dtype=float) * force
        for _ in range(a.pull_steps):
            d.xfrc_applied[bid][:3] = fvec
            step_raw(env, 1.0)
        bb = N4.bbox(sim, bid)
        c = None if bb is None else 0.5 * (np.asarray(bb[0]) + np.asarray(bb[1]))
        slip = None if (c is None or c0 is None) else float(np.linalg.norm(c - c0) * 1000)
        s = snap(env)
        pulls.append({"force_N": round(force, 2), "slip_mm": slip,
                      "aperture_mm": s["aperture_mm"], "Left_1": s["Left_1"]})
        log("  外力 %5.2f N → 滑移 %8.3f mm  净距=%8.3f mm  Left_1=%+.4f"
            % (force, slip if slip is not None else -1, s["aperture_mm"] or -1,
               s["Left_1"] or 0))
        if slip is not None and slip > 15.0:
            pull_out = force
            break
        force += 1.0 if force < 10 else 2.0
    d.xfrc_applied[bid] = 0.0
    out["pulls"] = pulls
    out["pull_out_force_N"] = pull_out
    out["static_friction_coeff_est"] = (
        None if (pull_out is None or not mass) else round(pull_out / (mass * 9.81), 3))
    return out


def main():
    env = build_env()
    sim, m, d, gid = N.ids(env)
    g = env.env.robots[0].gripper
    RES["gripper_info"] = {"class": type(g).__name__, "speed": float(g.speed),
                           "joints": [str(j) for j in g.joints], "dof": int(g.dof)}
    RES["aperture_reset_mm"] = EX.jaw_gap_mm(env)
    if not a.skip_d1:
        log("----- D1. 闭合刚度（ctrl → 净距） -----")
        RES["D1_closing_stiffness"] = d1_closing_stiffness(env)

    log("----- D2. 握持力（空载闭合 + 外力拽出） -----")
    # ★ 先驱动到全开再量开口：reset 时 ZhiXing 停在 init_qpos（17.8 mm），
    #   直接拿来当预算会把宽度算成负数（早期版本踩到：size must be positive）。
    env.reset()
    for _ in range(a.open_steps):
        step_raw(env, -1.0)
    open_gap = EX.jaw_gap_mm(env)
    RES["open_aperture_mm_at_full_open"] = open_gap
    width_mm = min(a.width_mm, float(open_gap or a.width_mm) - 4.0)
    log("  全开净开口 = %.3f mm → 试块宽度 = %.1f mm" % (open_gap or -1, width_mm))
    try:
        env.close()
    except Exception:                                          # noqa: BLE001
        pass
    env = build_env()
    env.reset()
    cat, obj, region = N3.target_category(env)
    N3.install_object_substitution(
        cat, N3.block_object_class(width_mm / 1000.0, N3.BLOCK_DEPTH_M, N3.BLOCK_HEIGHT_M),
        log)
    env.close()
    env = build_env()
    RES["D2_grip_force"] = d2_grip_force(env, width_mm)

    if getattr(a, "compare_panda", False):
        try:
            env.close()
        except Exception:                                      # noqa: BLE001
            pass
        envp = build_env("PandaGripper")
        RES["Panda_D1"] = d1_closing_stiffness(envp)
        try:
            envp.close()
        except Exception:                                      # noqa: BLE001
            pass
        envp = build_env("PandaGripper")
        envp.reset()
        N3.install_object_substitution(
            *N3.target_category(envp)[:1],
            N3.block_object_class(width_mm / 1000.0, N3.BLOCK_DEPTH_M, N3.BLOCK_HEIGHT_M),
            log)
        envp.close()
        envp = build_env("PandaGripper")
        RES["Panda_D2"] = d2_grip_force(envp, width_mm)

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RES, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log("[out] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
