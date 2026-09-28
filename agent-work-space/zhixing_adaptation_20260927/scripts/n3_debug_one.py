#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3 诊断：单次抓取全程逐阶段打印，定位「物体完全不动」的原因。
检查项：OSC 是否到位 / 夹持点模型是否成立 / 闭合后是否真的夹住 / 是否发生接触
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

TASK = 0


def main():
    import mujoco

    env, task = N.build(task_id=TASK)
    env.reset()
    sim, m, d, gid = N.ids(env)

    p, R = N.gripsite(sim, m, d, gid)
    print("=== 初始状态 ===")
    print(f"grip_site pos = {np.round(p,5).tolist()}")
    print(f"grip_site R[:,0]={np.round(R[:,0],4).tolist()}  R[:,1]={np.round(R[:,1],4).tolist()}  "
          f"R[:,2]={np.round(R[:,2],4).tolist()}")
    print(f"局部 +z 与世界 -z 夹角 = {np.degrees(np.arccos(np.clip(-R[2,2],-1,1))):.2f} deg")

    bodies = N.find_object_bodies(sim, "akita_black_bowl_1")
    jname = N.free_joint_of(sim, bodies[0][1])
    pts = N.body_world_pts(sim, bodies[0][0])
    obj0 = pts.mean(axis=0)
    print(f"\n=== 物体 ===\n  joint={jname}")
    print(f"顶点数={pts.shape[0]}  中心={np.round(obj0,4).tolist()}  "
          f"z范围=[{pts[:,2].min():.4f},{pts[:,2].max():.4f}]")

    # 外力测试：直接闭合看能否夹住（不做任何移动）
    print("\n=== A. 原地开合（不移动）===")
    N.hold(env, -1.0, 25)
    print(f"张开后 夹爪净距 = {N.jaw_face_separation_mm(env):.2f} mm")
    N.hold(env, +1.0, 25)
    print(f"闭合后 夹爪净距 = {N.jaw_face_separation_mm(env):.2f} mm")

    # 目标点
    z_top = float(pts[:, 2].max())
    z_g = z_top - 0.012
    r_out = float(pts[:, 1].max() - obj0[1])
    pin = np.array([obj0[0], obj0[1] + r_out - 0.004, z_g])
    gs = pin + np.array([0, 0, N.GRASP_OFFSET_CLOSED[2]])
    print(f"\n=== B. 目标 ===\nz_top={z_top:.4f} z_grasp={z_g:.4f} r_out={r_out*1000:.1f}mm")
    print(f"pinch 目标 = {np.round(pin,4).tolist()}")
    print(f"grip_site 目标 = {np.round(gs,4).tolist()}")

    # 接近
    print("\n=== C. 接近（上方 6 cm）===")
    st = N.drive_to(env, gs + np.array([0, 0, 0.06]), -1.0, max_steps=200, tol=0.003)
    p1, R1 = N.gripsite(sim, m, d, gid)
    print(f"  drive_to: {st}")
    print(f"  实际 grip_site = {np.round(p1,4).tolist()}")
    print(f"  实际 pinch     = {np.round(p1 + R1 @ N.GRASP_OFFSET_CLOSED,4).tolist()}")

    # 下降
    print("\n=== D. 下降到抓取位 ===")
    st = N.drive_to(env, gs, -1.0, max_steps=200, tol=0.003)
    p2, R2 = N.gripsite(sim, m, d, gid)
    pinch2 = p2 + R2 @ N.GRASP_OFFSET_CLOSED
    print(f"  drive_to: {st}")
    print(f"  实际 grip_site = {np.round(p2,4).tolist()}")
    print(f"  实际 pinch     = {np.round(pinch2,4).tolist()}   目标={np.round(pin,4).tolist()}")
    print(f"  pinch 误差 = {np.linalg.norm(pinch2-pin)*1000:.2f} mm")
    print(f"  夹爪净距 = {N.jaw_face_separation_mm(env):.2f} mm")

    # 闭合
    print("\n=== E. 闭合 ===")
    N.hold(env, +1.0, 30)
    print(f"  闭合后 夹爪净距 = {N.jaw_face_separation_mm(env):.2f} mm")
    pts2 = N.body_world_pts(sim, bodies[0][0])
    print(f"  物体中心位移 = {np.linalg.norm(pts2.mean(axis=0)-obj0)*1000:.2f} mm")

    # 接触统计
    print("\n=== F. 接触 ===")
    ng = {}
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1)
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2)
        if "gripper" in g1 or "gripper" in g2 or "pad" in g1 or "pad" in g2:
            key = tuple(sorted([g1, g2]))
            ng[key] = ng.get(key, 0) + 1
    if ng:
        for k, v in sorted(ng.items(), key=lambda x: -x[1])[:12]:
            print(f"  {v:3d}x  {k[0]}  <->  {k[1]}")
    else:
        print("  （无夹爪相关接触）")

    # 抬升
    print("\n=== G. 抬升 5 cm ===")
    st = N.drive_to(env, gs + np.array([0, 0, 0.05]), +1.0, max_steps=200, tol=0.003)
    print(f"  drive_to: {st}")
    pts3 = N.body_world_pts(sim, bodies[0][0])
    print(f"  物体 z 变化 = {(pts3.mean(axis=0)[2]-obj0[2])*1000:+.2f} mm")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
