#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3 判定实验：在碗壁处**极慢**闭合，区分两种失败机理：

  (甲) 碰撞体粒度不足 → 指垫凸包穿透薄壁，净距仍收到 ~0.56 mm（=空载闭合值）
  (乙) 闭合过快把碗推走 → 慢速闭合时指垫会停在壁厚附近（~6 mm）

做法：把夹持点开到碗壁中点，然后以 200 级斜坡把 ctrl 从 0 缓慢推到 -0.91，
逐级记录净距、碗的位移、以及指垫↔碗的接触数。
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
    bid = N.find_object_bodies(sim, "akita_black_bowl_1")[0][0]
    jname = N.free_joint_of(sim, "akita_black_bowl_1_main")

    N.set_free_joint(sim, jname, [-0.20, 0.10, 0.97])
    for _ in range(25):
        env.step(np.concatenate([np.zeros(6), [-1.0]]))
    sim, m, d, gid = N.ids(env)
    _, _, _, _, _, aadr, adof = N._arm_idx(env)
    q0 = np.array([d.qpos[x] for x in aadr], dtype=float)
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()

    pts = N.body_world_pts(sim, bid, collision_only=True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    z_g = 1.0101
    print(f"碰撞顶点 {pts.shape[0]} 个  x=[{lo[0]:.4f},{hi[0]:.4f}] y=[{lo[1]:.4f},{hi[1]:.4f}] "
          f"z=[{lo[2]:.4f},{hi[2]:.4f}]  轴心≈({cx:.4f},{cy:.4f})")
    sel = None
    for wx, wz, dy in ((0.030, 0.020, 0.005), (0.045, 0.030, 0.002), (0.060, 0.040, -0.005)):
        cand = pts[(np.abs(pts[:, 0] - cx) < wx) & (np.abs(pts[:, 2] - z_g) < wz)
                   & (pts[:, 1] > cy + dy)]
        print(f"  窗口 x±{wx*1000:.0f} z±{wz*1000:.0f} y>轴+{dy*1000:.0f}mm → {cand.shape[0]} 点")
        if cand.shape[0] >= 3:
            sel = cand
            break
    if sel is None:
        print("  ✗ 无法取到 +y 侧壁截面，改用全物体最大 y")
        r_out = float(hi[1] - cy)
        r_in = r_out - 0.006
    else:
        r_in = float(sel[:, 1].min() - cy)
        r_out = float(sel[:, 1].max() - cy)
    pin = np.array([cx, cy + 0.5 * (r_in + r_out), z_g])
    print(f"壁厚 {(r_out-r_in)*1000:.1f} mm (内 {r_in*1000:.1f} / 外 {r_out*1000:.1f})  目标壁中点 y={pin[1]:.4f}")

    q_g, res = N.ik_goto(env, q0, pin, R_ref, 0.0, settle=15)
    print(f"IK 残差 {res:.3f} mm   起始净距 {N.jaw_face_separation_mm(env):.2f} mm")

    pad_ids = [g for g in range(m.ngeom)
               if "fingerpad" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]

    def pad_bowl_contacts():
        n = 0
        for ci in range(d.ncon):
            c = d.contact[ci]
            s = {int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])}
            if bid in s and any(int(m.geom_bodyid[p]) in s for p in pad_ids):
                n += 1
        return n

    print("\n  ctrl     净距(mm)   碗位移(mm)  指垫↔碗接触")
    base = None
    for k in range(0, 201):
        ctrl = -0.91 * k / 200.0
        N.step_hold(env, q_g, ctrl, n=25)
        sim, m, d, gid = N.ids(env)
        sep = N.jaw_face_separation_mm(env)
        obj = N.body_world_pts(sim, bid).mean(axis=0)
        if base is None:
            base = obj.copy()
        shift = float(np.linalg.norm(obj - base) * 1000)
        nc = pad_bowl_contacts()
        if k % 20 == 0 or k == 200:
            print(f"  {ctrl:+.3f}   {sep:8.2f}   {shift:9.2f}   {nc:6d}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
