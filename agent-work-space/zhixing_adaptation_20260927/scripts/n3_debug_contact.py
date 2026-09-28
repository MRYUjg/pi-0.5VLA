#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3 诊断：为什么夹爪闭合时**穿过**碗壁（jaw 收到 0.56 mm 却无接触）。

检查：物体与指垫的 geom 类型 / contype / conaffinity / 尺寸，
并在碗壁处缓慢闭合，逐步打印夹爪净距与接触数。
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

    print("=== ZhiXing 指垫 geom ===")
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "fingerpad" in n:
            print(f"  {n:<30s} type={m.geom_type[g]} contype={m.geom_contype[g]} "
                  f"conaffinity={m.geom_conaffinity[g]} size={np.round(m.geom_size[g],4).tolist()}")

    b = N.find_object_bodies(sim, "akita_black_bowl_1")[0][0]
    print(f"\n=== 碗的 geom（body id={b}）===")
    cnt = {}
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != b:
            continue
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        cnt[(int(m.geom_type[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g]))] = \
            cnt.get((int(m.geom_type[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g])), 0) + 1
    print("  (type, contype, conaffinity) -> 个数")
    for k, v in sorted(cnt.items()):
        print(f"   {k} -> {v}")
    # 碰撞 geom 的实际世界范围
    col = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and m.geom_contype[g] != 0]
    vis = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b and m.geom_contype[g] == 0]
    print(f"  碰撞 geom {len(col)} 个, 视觉 geom {len(vis)} 个")
    for label, gs in (("碰撞", col), ("视觉", vis)):
        if not gs:
            continue
        P = np.vstack([N.body_world_pts(sim, b)])  # 占位
        # 逐 geom 世界包围盒
        lo = np.array([1e9] * 3)
        hi = np.array([-1e9] * 3)
        for g in gs:
            R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                a, nn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                v = np.array(m.mesh_vert[a:a + nn], dtype=float)
            else:
                s = np.array(m.geom_size[g], dtype=float)
                v = s * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            W = (R @ v.T).T + np.array(d.geom_xpos[g], dtype=float)
            lo = np.minimum(lo, W.min(axis=0))
            hi = np.maximum(hi, W.max(axis=0))
        print(f"  {label}范围 x=[{lo[0]:.4f},{hi[0]:.4f}] y=[{lo[1]:.4f},{hi[1]:.4f}] "
              f"z=[{lo[2]:.4f},{hi[2]:.4f}]")

    # 缓慢闭合测试
    print("\n=== 缓慢闭合（在碗壁处）===")
    jname = N.free_joint_of(sim, "akita_black_bowl_1_main")
    N.set_free_joint(sim, jname, [-0.20, 0.10, 0.97])
    for _ in range(25):
        env.step(np.concatenate([np.zeros(6), [-1.0]]))
    sim, m, d, gid = N.ids(env)
    _, _, _, _, _, aadr, adof = N._arm_idx(env)
    q0 = np.array([d.qpos[x] for x in aadr], dtype=float)
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    pts = N.body_world_pts(sim, b)
    cy = pts[:, 1].mean()
    r_out = float(pts[:, 1].max() - cy)
    print(f"  碗 y 中心≈{cy:.4f}  r_out={r_out*1000:.1f} mm")
    pin = np.array([-0.20, cy + r_out - 0.004, 1.0106])
    q_g, res = N.ik_goto(env, q0, pin, R_ref, 0.0, settle=15)
    print(f"  IK 残差 {res:.2f} mm, 夹爪净距 {N.jaw_face_separation_mm(env):.2f} mm")
    aid = N.gripper_actuator(env)
    for i in range(0, 601, 75):
        N.step_hold(env, q_g, -0.91, n=75) if i else None
        sim, m, d, gid = N.ids(env)
        nc = 0
        pairs = []
        for ci in range(d.ncon):
            c = d.contact[ci]
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or str(c.geom1)
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or str(c.geom2)
            if int(m.geom_bodyid[c.geom1]) == b or int(m.geom_bodyid[c.geom2]) == b:
                nc += 1
                pairs.append(tuple(sorted([g1, g2])))
        sep = N.jaw_face_separation_mm(env)
        obj = N.body_world_pts(sim, b).mean(axis=0)
        print(f"  step {i:4d}  净距={sep:7.2f} mm  碗相关接触={nc:3d}  碗中心y={obj[1]:.4f}  {pairs[:2]}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
