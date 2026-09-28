#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3 诊断：实测 LIBERO/robosuite OSC 动作的**参考系与尺度**。

不靠猜：施加单位动作，直接看 grip_site 在世界系的位移方向，
从而确定 action → 世界位移 的映射，用于脚本专家的闭环控制。
"""
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402


def probe(env, act7, n=10):
    import mujoco

    env.reset()
    sim, m, d, gid = N.ids(env)
    p0, R0 = N.gripsite(sim, m, d, gid)
    for _ in range(n):
        env.step(np.array(act7, dtype=float))
    p1, R1 = N.gripsite(sim, m, d, gid)
    return p0, p1, R0, R1


def main():
    env, task = N.build(task_id=0)
    env.reset()
    sim, m, d, gid = N.ids(env)
    _, R = N.gripsite(sim, m, d, gid)
    print("初始 grip_site 姿态（列 = 局部轴在世界系）:")
    for i, nm in enumerate("xyz"):
        print(f"  {nm}: {np.round(R[:,i],4).tolist()}")

    print("\n=== 平移维（每次动作重复 10 步）===")
    for i, nm in enumerate("xyz"):
        a = [0.0] * 7
        a[i] = 1.0
        a[6] = -1.0
        p0, p1, R0, R1 = probe(env, a)
        dw = p1 - p0
        print(f"  action[{i}]=+1 → 世界位移 {np.round(dw,5).tolist()}  |d|={np.linalg.norm(dw)*1000:.1f} mm"
              f"   在初始 eef 系下 = {np.round(R0.T @ dw,5).tolist()}")

    print("\n=== 旋转维 ===")
    for i, nm in enumerate("xyz"):
        a = [0.0] * 7
        a[3 + i] = 1.0
        a[6] = -1.0
        p0, p1, R0, R1 = probe(env, a, n=10)
        # 相对旋转的轴角
        Rrel = R1 @ R0.T
        ang = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
        print(f"  action[{3+i}]=+1 → 姿态变化 {ang:.2f} deg  世界位移 {np.round(p1-p0,5).tolist()}")

    print("\n=== 夹爪维 ===")
    for v in (+1.0, -1.0):
        a = [0.0] * 6 + [v]
        p0, p1, R0, R1 = probe(env, a, n=25)
        print(f"  action[6]={v:+.0f} ×25 步 → 夹爪净距 {N.jaw_face_separation_mm(env):.2f} mm")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
