#!/usr/bin/env python3
"""验证两个默认关闭的开关确实生效，且默认路径行为不变。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gripper_polarity import (make_env, get_robot, live_sim, jaw,  # noqa: E402
                              gripper_actuator_ids, fmt)

JOINTS = ["Left_1", "Left_Support", "Left_2", "Right_2", "Right_1", "Right_Support"]
EQS = [("Left_2", "Left_1", 1.0), ("Right_1", "Left_1", -1.0), ("Right_2", "Left_1", -1.0),
       ("Left_Support", "Left_1", -1.0), ("Right_Support", "Left_1", -1.0)]


def drive(env, cmd, steps=25):
    for _ in range(steps):
        a = np.zeros(7, dtype=float)
        a[6] = cmd
        env.step(a.tolist())
    r = get_robot(env)
    return jaw(live_sim(env, r), "zhixing", r)[0], r


def residuals(q):
    d = {J: q[i] for i, J in enumerate(JOINTS)}
    return {j1: round(d[j1] - c * d[j2], 5) for j1, j2, c in EQS}


def conn_left(sim, robot):
    from gripper_polarity import sites_ends
    return sites_ends(sim, "Left_Support_tip", "Left_2_connect")


def main():
    out = {}
    # --- 默认（无开关）---
    for k in ("ZHIXING_GRIPPER_INVERT", "ZHIXING_GRIPPER_CONSISTENT_INIT"):
        os.environ.pop(k, None)
    env = make_env("ECO65", "ZhixingGripper")
    env.reset()
    r = get_robot(env)
    sim = live_sim(env, r)
    q0 = np.asarray([sim.data.qpos[i] for i in r._ref_gripper_joint_pos_indexes], dtype=float)
    out["default_init_qpos"] = q0.round(5).tolist()
    out["default_init_residuals"] = residuals(q0)
    out["default_conn_left_m"] = round(conn_left(sim, r), 6)
    jm, _ = drive(env, -1.0)
    jp, _ = drive(env, +1.0)
    out["default_jaw_minus1_m"] = round(jm, 5)
    out["default_jaw_plus1_m"] = round(jp, 5)

    # --- CONSISTENT_INIT=1 ---
    os.environ["ZHIXING_GRIPPER_CONSISTENT_INIT"] = "1"
    env.reset()
    r = get_robot(env)
    sim = live_sim(env, r)
    q1 = np.asarray([sim.data.qpos[i] for i in r._ref_gripper_joint_pos_indexes], dtype=float)
    out["consistent_init_qpos"] = q1.round(5).tolist()
    out["consistent_init_residuals"] = residuals(q1)
    out["consistent_conn_left_m"] = round(conn_left(sim, r), 6)
    os.environ.pop("ZHIXING_GRIPPER_CONSISTENT_INIT", None)

    # --- INVERT=1 ---
    os.environ["ZHIXING_GRIPPER_INVERT"] = "1"
    env.reset()
    jm2, _ = drive(env, -1.0)
    jp2, _ = drive(env, +1.0)
    out["invert_jaw_minus1_m"] = round(jm2, 5)
    out["invert_jaw_plus1_m"] = round(jp2, 5)
    os.environ.pop("ZHIXING_GRIPPER_INVERT", None)

    out["PASS_default_unchanged"] = bool(
        out["default_init_qpos"] == [-0.91] * 6 and out["default_jaw_plus1_m"] > out["default_jaw_minus1_m"])
    out["PASS_consistent_init_fixes_eq"] = bool(
        all(abs(v) < 1e-4 for v in out["consistent_init_residuals"].values())
        and out["consistent_conn_left_m"] < 1e-3
        and abs(out["default_init_residuals"]["Right_1"]) > 1.0)
    out["PASS_invert_flips_polarity"] = bool(
        out["invert_jaw_plus1_m"] < out["invert_jaw_minus1_m"])

    for k, v in out.items():
        print("  %-32s %s" % (k, v))
    print("\nALL_PASS =", all(v for k, v in out.items() if k.startswith("PASS_")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
