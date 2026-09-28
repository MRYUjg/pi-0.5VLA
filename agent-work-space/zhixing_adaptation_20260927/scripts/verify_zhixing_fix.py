#!/usr/bin/env python3
"""验证 ZhiXing 夹爪的两项修复已正式落地（不再是环境变量开关）。

期望（不带任何环境变量）：
  1. init_qpos 返回 6 个元素，且与 XML 的 5 条 joint equality 完全自洽（残差 0.0）
     左侧四连杆 connect 残差 → 0
  2. action +1 = 闭合（开口变小）、-1 = 张开（开口变大）—— 与 Panda 约定一致
  3. speed 默认仍为 0.2（未改动）
  4. 已移除的开关环境变量不再有任何效果
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gripper_polarity import (make_env, get_robot, live_sim, jaw,  # noqa: E402
                              sites_ends, fmt)

JOINTS = ["Left_1", "Left_Support", "Left_2", "Right_2", "Right_1", "Right_Support"]
EQS = [("Left_2", "Left_1", 1.0), ("Right_1", "Left_1", -1.0), ("Right_2", "Left_1", -1.0),
       ("Left_Support", "Left_1", -1.0), ("Right_Support", "Left_1", -1.0)]
EXPECTED_INIT = [-0.91, 0.91, -0.91, 0.91, 0.91, 0.91]


def drive(env, cmd, steps=25):
    for _ in range(steps):
        a = np.zeros(7, dtype=float)
        a[6] = cmd
        env.step(a.tolist())
    r = get_robot(env)
    return jaw(live_sim(env, r), "zhixing", r)[0]


def residuals(q):
    d = {J: q[i] for i, J in enumerate(JOINTS)}
    return {j1: round(float(d[j1] - c * d[j2]), 6) for j1, j2, c in EQS}


def main():
    for k in ("ZHIXING_GRIPPER_INVERT", "ZHIXING_GRIPPER_CONSISTENT_INIT", "ZHIXING_GRIPPER_SPEED"):
        os.environ.pop(k, None)

    out = {}
    env = make_env("ECO65", "ZhixingGripper")
    env.reset()
    r = get_robot(env)
    sim = live_sim(env, r)

    q = np.asarray([sim.data.qpos[i] for i in r._ref_gripper_joint_pos_indexes], dtype=float)
    out["init_qpos_len_declared"] = int(np.asarray(r.gripper.init_qpos).size)
    out["init_qpos_declared"] = np.asarray(r.gripper.init_qpos, dtype=float).round(5).tolist()
    out["qpos_after_reset"] = q.round(5).tolist()
    out["joint_eq_residuals"] = residuals(q)
    out["conn_left_m"] = round(float(sites_ends(sim, "Left_Support_tip", "Left_2_connect")), 8)
    out["conn_right_m"] = round(float(sites_ends(sim, "Right_Support_tip", "Right_2_connect")), 8)
    out["speed_default"] = float(r.gripper.speed)

    j_minus = drive(env, -1.0)
    j_plus = drive(env, +1.0)
    out["jaw_action_minus1_m"] = round(float(j_minus), 5)
    out["jaw_action_plus1_m"] = round(float(j_plus), 5)

    # 已移除的开关不应再有效果
    os.environ["ZHIXING_GRIPPER_INVERT"] = "1"
    j_plus_flagged = drive(env, +1.0)
    out["jaw_plus1_with_legacy_invert_flag_m"] = round(float(j_plus_flagged), 5)
    os.environ.pop("ZHIXING_GRIPPER_INVERT", None)

    out["PASS_init_is_6elem"] = out["init_qpos_len_declared"] == 6
    out["PASS_init_equals_expected"] = out["init_qpos_declared"] == EXPECTED_INIT
    out["PASS_reset_matches_init"] = out["qpos_after_reset"] == EXPECTED_INIT
    out["PASS_eq_residuals_zero"] = all(abs(v) < 1e-6 for v in out["joint_eq_residuals"].values())
    out["PASS_fourbar_closed"] = out["conn_left_m"] < 1e-5 and out["conn_right_m"] < 1e-5
    out["PASS_polarity_plus1_closes"] = out["jaw_action_plus1_m"] < out["jaw_action_minus1_m"]
    out["PASS_speed_unchanged_0p2"] = abs(out["speed_default"] - 0.2) < 1e-9
    out["PASS_legacy_flag_no_effect"] = abs(
        out["jaw_plus1_with_legacy_invert_flag_m"] - out["jaw_action_plus1_m"]) < 1e-6

    for k, v in out.items():
        print("  %-40s %s" % (k, v))
    ok = all(v for k, v in out.items() if k.startswith("PASS_"))
    print("\nALL_PASS =", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
