#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
判别「夹爪开合重复性」与「测试给的时间够不够」是两件事。

背景：落地 ±2 N·m 力上限后（= 厂商 URDF 的 effort=2），空载全行程需 **488 子步 ≈ 0.98 s
≈ 100 mm/s**，与厂商规格 `<100 mm/s` 一致。而既有 N3 的 A 项每相位只给 25 个 env step
（625 子步），留量过小 → 读数在 65.9~90.1 mm 间跳动，被判 FAIL。

本脚本把两件事分开测：
  1. **给足时间**（每相位 settle 子步可配）下的开合重复性 —— 这才是夹爪本身的重复性；
  2. 同时报告每次实际用到的子步数，验证 25 env step 是否真的不够。
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402


def main():
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--settle", type=int, default=1500,
                    help="每个相位给的子步数（env.step 数 = settle/25）")
    a = ap.parse_args()

    env, task = N.build(task_id=0)
    env.reset()
    sim, m, d, gid = N.ids(env)
    aid = N.gripper_actuator(env)
    print("[actuator] forcelimited=%d forcerange=%s"
          % (int(m.actuator_forcelimited[aid]),
             np.round(m.actuator_forcerange[aid], 3).tolist()))

    opens, closes, used_open, used_close = [], [], [], []
    for i in range(a.trials):
        # 开（ctrl = 0.0）：跑到停住或超时
        st = _drive(env, 0.0, a.settle)
        used_open.append(st)
        opens.append(N.jaw_face_separation_mm(env))
        # 闭（ctrl = -0.91）
        st = _drive(env, -0.91, a.settle)
        used_close.append(st)
        closes.append(N.jaw_face_separation_mm(env))

    def stat(v):
        return (float(np.mean(v)), float(np.std(v)), float(np.min(v)), float(np.max(v)))

    for name, v, used in (("全开", opens, used_open), ("闭合", closes, used_close)):
        mu, sd, lo, hi = stat(v)
        print("  %s 净距 = %7.2f ± %.3f mm  (min %.2f / max %.2f)   子步 %s"
              % (name, mu, sd, lo, hi, used))
    mu_o, sd_o, _, _ = stat(opens)
    mu_c, sd_c, _, _ = stat(closes)
    ok = sd_o < 0.5 and sd_c < 0.5
    print("\n  判据 std < 0.5 mm  ->  %s" % ("PASS" if ok else "FAIL"))
    print("  对照：既有 A 项每相位 25 env step = 625 子步（留量不足）")
    env.close()
    return 0


def _drive(env, ctrl, max_substeps):
    """直驱夹爪到 ctrl，锁臂；返回用掉的子步数。"""
    import mujoco

    sim, m, d, gid, aj, aadr, adof = N._arm_idx(env)
    aid = N.gripper_actuator(env)
    q_hold = np.array([d.qpos[x] for x in aadr], dtype=float)
    jl1 = None
    for j in range(m.njnt):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("Left_1_Joint"):
            jl1 = j
    prev, stable = None, 0
    for k in range(1, max_substeps + 1):
        N.set_arm(sim, m, d, aadr, adof, q_hold)
        d.ctrl[aid] = float(ctrl)
        sim.step()
        cur = float(d.qpos[int(m.jnt_qposadr[jl1])])
        if prev is not None and abs(cur - prev) < 1e-5:
            stable += 1
            if stable >= 25:
                return k
        else:
            stable = 0
        prev = cur
    return max_substeps


if __name__ == "__main__":
    sys.exit(main())
