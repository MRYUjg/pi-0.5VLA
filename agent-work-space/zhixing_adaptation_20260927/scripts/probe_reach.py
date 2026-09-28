#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3 诊断：ECO65(+ZhiXing) 在「工具垂直向下」姿态下的**可达工作空间**。

背景：脚本专家的抓取目标若不可达，IK 会收敛到最近点并留下几十 mm 残差，
表现为「夹爪悬在半空、夹爪闭合但夹空」。先量出可达区域，再把物体放到可达处。
"""
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402


def main():
    import mujoco

    env, _ = N.build(task_id=0)
    env.reset()
    sim, m, d, gid = N.ids(env)
    _, _, _, _, _, aadr, adof = N._arm_idx(env)
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    q0 = np.array([d.qpos[x] for x in aadr], dtype=float)
    print(f"参考姿态 R[:,2]（工具轴）= {np.round(R_ref[:,2],4).tolist()}")
    print(f"reset grip_site = {np.round(N.gripsite(sim,m,d,gid)[0],4).tolist()}")

    print("\n可达性网格：残差 mm（x 行 × y 列），z=1.01，工具垂直向下")
    xs = np.arange(-0.30, 0.11, 0.05)
    ys = np.arange(-0.05, 0.36, 0.05)
    hdr = "   x\\y  " + "".join(f"{y:7.2f}" for y in ys)
    print(hdr)
    best = []
    for x in xs:
        row = f"  {x:6.2f} "
        for y in ys:
            for z in (1.01, 1.06):
                pass
            pinch_t = np.array([x, y, 1.01])
            gs = pinch_t - R_ref @ N.GRASP_OFFSET_CLOSED   # 正确反推 grip_site
            q, _, _ = N.solve_ik_gripsite(env, gs, R_ref, q0, iters=1200)
            for a, v in zip(aadr, q):
                d.qpos[a] = v
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = np.array(d.site_xpos[gid], dtype=float)
            R = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)
            res = float(np.linalg.norm(p + R @ N.GRASP_OFFSET_CLOSED - pinch_t) * 1000)
            row += f"{res:7.1f}"
            if res < 3.0:
                best.append((res, float(x), float(y)))
        print(row)

    best.sort()
    print(f"\n可达点（残差 <3 mm）共 {len(best)} 个，最优 10 个：")
    for r, x, y in best[:10]:
        print(f"   ({x:+.2f},{y:+.2f})  残差 {r:.2f} mm")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
