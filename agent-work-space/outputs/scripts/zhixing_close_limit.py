#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZhiXing 夹爪闭合行程定因（只读）：18 mm 的“合到底”是几何限制还是模型/接触阻塞？

依次测：
  1. reset 后直接**写 qpos**到约束自洽闭合位（Left_1=-0.91，其余按 equality 反解）
     → 记录指垫净距（对照 N2 契约的 0.57 mm）
  2. env.reset() 后的初始位（Left_1=-0.91）→ 同样记录
  3. 命令 +1 动态闭合到停住所达到的构型 → 记录
  4. 动态闭合停住时，**强制把 ctrl 钉在 -0.91**（覆盖 format_action），看 500 步后
     关节是否被推到位、指垫是否继续靠近、作动器力是否饱和
  5. 停住构型下的**自接触**清单（哪些 geom 互相穿透）

输出 JSON，不做任何纠错。
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
import n4_collect as N4                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_object")
ap.add_argument("--task-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="/app/agent-work-space/outputs/diag/zhixing_close_limit.json")
a = ap.parse_args()
os.environ.setdefault("ZHIXING_GRIPPER_SPEED", "0.01")
RES = {"node": "ZHIXING-CLOSE-LIMIT", "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
       "note": "只读定因：未修改任何代码/参数"}


def log(m):
    print(m, flush=True)


def step_raw(env, cmd):
    try:
        env.env.done = False
    except Exception:                                          # noqa: BLE001
        pass
    return env.step(np.concatenate([np.zeros(6), [cmd]]))


def snap(env, tag):
    import mujoco
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    idx = [int(i) for i in r._ref_gripper_joint_pos_indexes]
    joints = [str(j) for j in r.gripper.joints]
    q = [float(d.qpos[i]) for i in idx]
    pads = EX.pad_geoms_generic(sim)
    pad_pos = {k: [float(v) for v in d.geom_xpos[pads[k]]] for k in ("left", "right")}
    ctr_mm = float(np.linalg.norm(np.asarray(pad_pos["left"]) - np.asarray(pad_pos["right"])) * 1000)
    # 自接触：两侧同属夹爪 body 的接触对
    self_pen = []
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or str(g)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        n1, n2 = gname(g1), gname(g2)
        if ("gripper0" in n1) and ("gripper0" in n2) and float(c.dist) < 0:
            self_pen.append({"g1": n1, "g2": n2, "pen_mm": round(-float(c.dist) * 1000, 4)})
    return {"tag": tag,
            "Left_1": next((q[j] for j, nm in enumerate(joints) if nm.endswith("Left_1_Joint")), None),
            "gripper_qpos": [round(v, 6) for v in q],
            "aperture_mm": None if EX.jaw_gap_mm(env) is None else round(EX.jaw_gap_mm(env), 4),
            "min_clearance_mm": None if EX.min_clearance_mm(env) is None else round(EX.min_clearance_mm(env), 4),
            "pad_center_distance_mm": round(ctr_mm, 4),
            "pad_left_pos": pad_pos["left"], "pad_right_pos": pad_pos["right"],
            "ctrl": [float(v) for v in np.atleast_1d(d.ctrl[6])],
            "actuator_force": [float(v) for v in np.atleast_1d(d.actuator_force)],
            "ncon": int(d.ncon),
            "ncon_self": len(self_pen),
            "self_penetrations": self_pen[:12],
            "current_action": [float(v) for v in np.atleast_1d(r.gripper.current_action)]}


def main():
    env, task = N4.build(a.suite, a.task_id, robot="ECO65", gripper="ZhixingGripper", seed=a.seed)
    EX.install_generic_pad_geoms(lambda m: None)
    EX.install_generic_state_gripper(env, lambda m: None)
    import mujoco
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    joints = [str(j) for j in r.gripper.joints]
    idx = [int(i) for i in r._ref_gripper_joint_pos_indexes]
    RES["joints"] = joints
    RES["joint_limits_local"] = {nm: [float(v) for v in m.jnt_range[j]]
                                 for j, nm in enumerate(
                                     [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, k) or "" for k in range(m.njnt)])
                                 if nm.endswith("Joint")}
    RES["ncon_at_reset_before_any_step"] = int(d.ncon)

    # ---- 1. 直接写 qpos 到约束自洽闭合位 ----
    env.reset()
    sim, m, d, gid = N.ids(env)
    q_close = [-0.91, 0.91, -0.91, 0.91, 0.91, 0.91]
    for i, v in zip(idx, q_close):
        d.qpos[i] = v
    mujoco.mj_forward(m, d)
    s1 = snap(env, "qpos_written_closed")
    RES["S1_qpos_written_closed"] = s1
    log("[S1] 直接写 qpos 到闭合位：Left_1=%+.4f 指垫净距=%.4f mm 指垫中心距=%.4f mm 自接触=%d"
        % (s1["Left_1"], s1["aperture_mm"], s1["pad_center_distance_mm"], s1["ncon_self"]))

    # ---- 2. reset 后的初始位 ----
    env.reset()
    s2 = snap(env, "after_reset")
    RES["S2_after_reset"] = s2
    log("[S2] reset 初始位：Left_1=%+.4f 指垫净距=%.4f mm 自接触=%d 总接触=%d"
        % (s2["Left_1"], s2["aperture_mm"], s2["ncon_self"], s2["ncon"]))

    # ---- 3. 动态闭合到停住 ----
    for _ in range(500):
        step_raw(env, 1.0)
    s3 = snap(env, "dynamic_closed")
    RES["S3_dynamic_closed"] = s3
    log("[S3] 动态 +1 闭合 500 步：Left_1=%+.4f 指垫净距=%.4f mm 指垫中心距=%.4f mm "
        "act_force=%s 自接触=%d" % (s3["Left_1"], s3["aperture_mm"], s3["pad_center_distance_mm"],
                                    [round(v, 4) for v in s3["actuator_force"]], s3["ncon_self"]))

    # ---- 4. 强制 ctrl=-0.91 继续推 500 步 ----
    g = r.gripper
    orig = g.format_action

    def stub(action):
        return np.array([-0.91], dtype=float)

    g.format_action = stub
    hist = []
    for i in range(500):
        step_raw(env, 1.0)
        if i % 100 == 0 or i == 499:
            hist.append({"i": i, "Left_1": None, "aperture_mm": None})
            st = snap(env, "forced_%d" % i)
            hist[-1] = {"i": i, "Left_1": st["Left_1"], "aperture_mm": st["aperture_mm"],
                        "pad_center_distance_mm": st["pad_center_distance_mm"],
                        "actuator_force": st["actuator_force"], "ncon_self": st["ncon_self"]}
    g.format_action = orig
    s4 = snap(env, "forced_ctrl_minus0p91")
    RES["S4_forced_ctrl"] = {"traj": hist, "final": s4}
    log("[S4] 强制 ctrl=-0.91 再推 500 步：Left_1=%+.4f 指垫净距=%.4f mm 指垫中心距=%.4f mm "
        "act_force=%s 自接触=%d" % (s4["Left_1"], s4["aperture_mm"], s4["pad_center_distance_mm"],
                                    [round(v, 4) for v in s4["actuator_force"]], s4["ncon_self"]))
    for h in hist:
        log("      i=%-4d Left_1=%+.4f 净距=%.4f mm 中心距=%.4f mm force=%s 自接触=%d"
            % (h["i"], h["Left_1"], h["aperture_mm"], h["pad_center_distance_mm"],
               [round(v, 4) for v in h["actuator_force"]], h["ncon_self"]))

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RES, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log("[out] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
