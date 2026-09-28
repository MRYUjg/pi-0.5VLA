#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为什么加了力上限后开合变得又慢又不稳？
（Part A 复验：闭合净距 0.56~67.23 mm，std 19.8 —— 之前是 0.56+0.000）

实测现象：无限力时 ~600 子步走完全行程；限力 1.33 时需 ~3650 子步才到 0.67 行程
（≈8 mm/s，而厂商规格是 <100 mm/s，慢约 12 倍）。
本脚本查清「阻力来自哪里」，并给出开合时间的量化对照。

打印：
  * 夹爪 6 个关节的 damping / frictionloss / armature / stiffness
  * equality（四连杆 + connect）约束的 solref / solimp
  * 不同力矩上限下的开合时间（子步数 / 秒 / 等效 mm/s）
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402


def main():
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="none,8,4,2,1")
    ap.add_argument("--max-steps", type=int, default=40000)
    a = ap.parse_args()

    env, task = N.build(task_id=0)
    env.reset()
    sim, m, d, gid = N.ids(env)
    aid = N.gripper_actuator(env)

    print("=== 夹爪关节属性 ===")
    gjs = [j for j in range(m.njnt)
           if "gripper0" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")]
    for j in gjs:
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        print("  %-42s damping=%.4g frictionloss=%.4g armature=%.4g stiffness=%.4g "
              "range=%s" % (nm, m.dof_damping[int(m.jnt_dofadr[j])],
                            m.dof_frictionloss[int(m.jnt_dofadr[j])],
                            m.dof_armature[int(m.jnt_dofadr[j])],
                            m.jnt_stiffness[j], np.round(m.jnt_range[j], 3).tolist()))

    print("=== equality 约束 ===")
    for e in range(m.neq):
        print("  eq[%d] type=%d solref=%s solimp=%s"
              % (e, int(m.eq_type[e]), np.round(m.eq_solref[e], 4).tolist(),
                 np.round(m.eq_solimp[e], 4).tolist()))

    print("=== 执行器 ===")
    print("  kp=%.1f forcelimited=%d forcerange=%s"
          % (m.actuator_gainprm[aid][0], int(m.actuator_forcelimited[aid]),
             np.round(m.actuator_forcerange[aid], 3).tolist()))

    # 关节限位范围（用于换算行程比例）
    jl1 = None
    for j in gjs:
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").endswith("Left_1_Joint"):
            jl1 = j
    lo, hi = float(m.jnt_range[jl1][0]), float(m.jnt_range[jl1][1])
    span = abs(hi - lo)
    print("  Left_1_Joint range=[%.3f, %.3f] span=%.3f rad\n" % (lo, hi, span))

    def q1():
        return float(d.qpos[int(m.jnt_qposadr[jl1])])

    def drive(ctrl, cap, max_steps):
        """把夹爪驱到 ctrl，返回 (子步数, 终值, 是否停住)。"""
        if cap is None:
            m.actuator_forcelimited[aid] = 0
        else:
            m.actuator_forcelimited[aid] = 1
            m.actuator_forcerange[aid] = [-cap, cap]
        aadr = N._arm_idx(env)[5]
        q_hold = np.array([d.qpos[x] for x in aadr], dtype=float)
        prev, stable = None, 0
        for k in range(1, max_steps + 1):
            N.set_arm(sim, m, d, aadr, N._arm_idx(env)[6], q_hold)
            d.ctrl[aid] = float(ctrl)
            sim.step()
            cur = q1()
            if prev is not None and abs(cur - prev) < 1e-5:
                stable += 1
                if stable >= 20:
                    return k, cur, True
            else:
                stable = 0
            prev = cur
        return max_steps, q1(), False

    print("=== 开合时间对照（空载，锁臂直驱）===")
    print("  %-8s %-10s %-12s %-12s %-10s" % ("cap", "方向", "子步数", "秒", "等效 mm/s"))
    for cap_s in a.caps.split(","):
        cap_s = cap_s.strip()
        cap = None if cap_s == "none" else float(cap_s)
        # 先开到底
        drive(0.0, cap, a.max_steps)
        d.qpos[int(m.jnt_qposadr[jl1])] = hi
        mujoco.mj_forward(m, d)
        n_steps, q_end, stopped = drive(lo, cap, a.max_steps)
        frac = abs(q_end - hi) / span
        mm = frac * 90.0                     # 全行程按 90 mm 计
        sec = n_steps * float(m.opt.timestep)
        print("  %-8s %-10s %-12d %-12.3f %-10.1f%s"
              % (cap_s, "开->闭", n_steps, sec, mm / sec if sec > 0 else 0,
                 "" if stopped else "  (未停住)"))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
