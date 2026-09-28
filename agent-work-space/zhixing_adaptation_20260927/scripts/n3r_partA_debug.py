#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
诊断：为什么 A 项经由 **action 路径**（env.step + format_action）闭合只到 ~40.6 mm，
而**直驱**（d.ctrl）能到 0.52 mm？

逐相位打印：Left_1_Joint 角度、夹爪净距、grip_site z、以及涉及指垫的接触。
"""
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402


def left1(env):
    import mujoco

    sim, m, d, _ = N.ids(env)
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.endswith("Left_1_Joint"):
            return float(d.qpos[int(m.jnt_qposadr[j])])
    return float("nan")


def ctrl_now(env):
    sim, m, d, _ = N.ids(env)
    aid = N.gripper_actuator(env)
    return float(d.ctrl[aid]), float(d.actuator_force[aid]), \
        float(N.ids(env)[2].qpos[int(m.jnt_qposadr[0])]) if False else None


def pad_contacts(env):
    import mujoco

    sim, m, d, _ = N.ids(env)
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        n2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        if "pad" in n1 + n2 or "fingerpad" in n1 + n2:
            out.append("%s<->%s dist=%+.2fmm pos=%s"
                       % (n1, n2, float(c.dist) * 1000,
                          np.round(np.array(c.pos, dtype=float), 4).tolist()))
    return out


def report(env, tag):
    import mujoco

    sim, m, d, gid = N.ids(env)
    aid = N.gripper_actuator(env)
    gz = float(d.site_xpos[gid][2])
    jaw = N.jaw_face_separation_mm(env)
    print("  %-16s L1=%+.4f  ctrl=%+.4f  actF=%+7.3f  jaw=%7.2f mm  grip_z=%.4f  接触=%s"
          % (tag, left1(env), float(d.ctrl[aid]), float(d.actuator_force[aid]),
             jaw if jaw is not None else float("nan"), gz, pad_contacts(env)))


def main():
    env, task = N.build(task_id=0)
    env.reset()
    report(env, "reset 后")
    for phase, cmd, settle in (("开(-1.0)", -1.0, 80), ("闭(+1.0)", +1.0, 80),
                               ("再开(-1.0)", -1.0, 80), ("再闭(+1.0)", +1.0, 80)):
        N.hold(env, cmd, settle)
        report(env, phase)

    # 对照：直驱同一 cmd 的等效 ctrl
    print("\n[对照] 直驱 ctrl：")
    for c in (0.0, -0.455, -0.91):
        aadr = N._arm_idx(env)[5]
        d0 = N.ids(env)[2]
        q = np.array([d0.qpos[x] for x in aadr], dtype=float)
        N.step_hold(env, q, c, n=1500)
        jaw = N.jaw_face_separation_mm(env)
        print("  ctrl=%+.3f  L1=%+.4f  jaw=%7.2f mm  %s"
              % (c, left1(env), jaw, pad_contacts(env)))

    # 几何核对：两个 pad box 的世界位置/尺寸，以及它们与**对侧 mesh** 的关系
    print("\n[几何] pad box 世界信息：")
    import mujoco
    sim, m, d, gid = N.ids(env)
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "pad_collision" not in nm:
            continue
        R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
        sz = np.array(m.geom_size[g], dtype=float)
        corners = np.array([[sx, sy, sz2] for sx in (-1, 1) for sy in (-1, 1)
                            for sz2 in (-1, 1)]) * sz
        P = (R @ corners.T).T + np.array(d.geom_xpos[g], dtype=float)
        print("  %-34s xpos=%s  world bbox size(mm)=%s"
              % (nm, np.round(d.geom_xpos[g], 4).tolist(),
                 np.round((P.max(axis=0) - P.min(axis=0)) * 1000, 1).tolist()))

    # 关掉力上限再走 action 路径（判别是否 forcerange 引起）
    print("\n[对照] 临时关掉 forcerange，再走 action 路径：")
    sim, m, d, _ = N.ids(env)
    aid = N.gripper_actuator(env)
    m.actuator_forcelimited[aid] = 0
    env.reset()
    N.hold(env, +1.0, 80)
    jaw = N.jaw_face_separation_mm(env)
    print("  无限力 action 闭合: L1=%+.4f  jaw=%7.2f mm  %s"
          % (left1(env), jaw, pad_contacts(env)))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
