#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3-redo 抬升逐步追踪：为什么「拉拔能撑住 8 N」但「抬升带不走 1.44 N 的试块」？

已知（n3r_why.py，限力 23 N 等效）：
  * 夹持正常：法向 [0,±1,0]、ΣfN≈26 N、穿透 3.6 mm；
  * 给试块施加**向上外力**时，8.0 N 才把它拉动（自重仅 1.44 N）。
  * 但 n3r_block_test 的「抬升 5 cm」仍然带不走试块。

本脚本逐步爬升并逐帧记录：目标 / grip_site / 指垫内面 / 试块 / 接触，
定位断点到底在"指垫没动""接触丢了"还是"摩擦没传上来"。
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N      # noqa: E402
import n3r_block_test as B     # noqa: E402
import n3r_why as W            # noqa: E402


def pad_face_z(env):
    """两个指垫内面中点的世界 z。"""
    import mujoco

    sim, m, d, _ = N.ids(env)
    ids = W.pad_ids(sim)
    P = {}
    for g in ids:
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = np.array(m.mesh_vert[a:a + n], dtype=float)
        else:
            v = np.zeros((1, 3))
        R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
        P[g] = (R @ v.T).T + np.array(d.geom_xpos[g], dtype=float)
    zs = []
    for g in ids:
        u = np.mean([P[h].mean(axis=0) for h in ids if h != g], axis=0) - P[g].mean(axis=0)
        u /= np.linalg.norm(u)
        proj = P[g] @ u
        zs.append(float(P[g][proj >= proj.max() - 0.002].mean(axis=0)[2]))
    return float(np.mean(zs)), zs


def contact_summary(env, bid):
    import mujoco

    sim, m, d, _ = N.ids(env)
    ids = set(W.pad_ids(sim))
    n_ct, sum_fN, sum_fTz, nz = 0, 0.0, 0.0, []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if int(m.geom_bodyid[g1]) != bid and int(m.geom_bodyid[g2]) != bid:
            continue
        if g1 not in ids and g2 not in ids:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        n_ct += 1
        sum_fN += float(f[0])
        nz.append(float(c.frame[2]))
        # 切向力在 z 方向的分量（frame 的后 6 个分量是两条切线）
        frame = np.array(c.frame, dtype=float).reshape(3, 3)
        sum_fTz += float(f[1] * frame[1][2] + f[2] * frame[2][2])
    return {"n_ct": n_ct, "sum_fN": sum_fN, "sum_fTz": sum_fTz,
            "normal_z": nz}


def left1(env):
    import mujoco

    sim, m, d, _ = N.ids(env)
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.endswith("Left_1_Joint"):
            return float(d.qpos[int(m.jnt_qposadr[j])])
    return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mm", type=float, default=50.0)
    ap.add_argument("--height-mm", type=float, default=70.0)
    ap.add_argument("--depth-mm", type=float, default=60.0)
    ap.add_argument("--torque-cap", type=float, default=1.33)
    ap.add_argument("--ramp", type=int, default=25)
    ap.add_argument("--ramp-settle", type=int, default=8)
    ap.add_argument("--lift-m", type=float, default=0.05)
    ap.add_argument("--grasp-above-table", type=float, default=0.035)
    ap.add_argument("--place-x", type=float, default=-0.22)
    ap.add_argument("--place-y", type=float, default=0.10)
    ap.add_argument("--drop-z", type=float, default=1.30)
    ap.add_argument("--settle-steps", type=int, default=400)
    ap.add_argument("--close-steps", type=int, default=600)
    ap.add_argument("--close-max-steps", type=int, default=8000)
    ap.add_argument("--close-chunk", type=int, default=50)
    a = ap.parse_args()

    hy, hz = a.width_mm / 2000.0, a.height_mm / 2000.0
    hx = a.depth_mm / 2000.0
    B.install_block_injection((hx, hy, hz), (a.place_x, a.place_y, a.drop_z))
    env, task = N.build(task_id=0)
    env.reset()
    sim, m, d, gid = N.ids(env)
    aid = N.gripper_actuator(env)
    m.actuator_forcerange[aid] = [-a.torque_cap, a.torque_cap]
    m.actuator_forcelimited[aid] = 1
    bid = B.block_id(sim)

    for _ in range(a.settle_steps):
        env.step(np.concatenate([np.zeros(6), [0.0]]))
    sim, m, d, gid = N.ids(env)
    pts = N.body_world_pts(sim, bid, collision_only=True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1]))
    z_g = float(lo[2]) + a.grasp_above_table
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    aadr = N._arm_idx(env)[5]
    q_home = np.array([d.qpos[x] for x in aadr], dtype=float)
    obj0 = N.body_world_pts(sim, bid).mean(axis=0)
    pin = np.array([cx, cy, z_g])
    print("[setup] block z=[%.4f,%.4f]  z_g=%.4f  cap=±%.2f" % (lo[2], hi[2], z_g, a.torque_cap))

    q_pre, r1 = N.ik_goto(env, q_home, pin + np.array([0, 0, 0.06]), R_ref, 0.0, settle=15)
    q_g, r2 = N.ik_goto(env, q_pre, pin, R_ref, 0.0, settle=15)
    ca = B.close_until_stall(env, q_g, a)
    print("[close] 闭合到停住：用了 %d 步，Left_1=%.4f" % (ca["close_steps_used"], ca["left_1_qpos_after_close"]))

    def row(tag, tgt_z):
        s, mm, dd, gg = N.ids(env)
        gsz = float(dd.site_xpos[gg][2])
        pf, zs = pad_face_z(env)
        bz = float(N.body_world_pts(s, bid).mean(axis=0)[2])
        cs = contact_summary(env, bid)
        print("  %-14s tgt_z=%.4f  grip_site_z=%.4f  指垫内面z=%.4f  试块z=%.4f (Δ%+6.2fmm)  "
              "L1=%+.4f  接触=%d ΣfN=%6.2f ΣfTz=%+7.3f  normal_z=%s"
              % (tag, tgt_z, gsz, pf, bz, (bz - obj0[2]) * 1000, left1(env),
                 cs["n_ct"], cs["sum_fN"], cs["sum_fTz"],
                 np.round(cs["normal_z"], 2).tolist()))
        return bz

    row("闭合后", z_g)
    q_prev = q_g
    for k in range(1, a.ramp + 1):
        tgt = pin + np.array([0, 0, a.lift_m * k / a.ramp])
        q_prev, res = N.ik_goto(env, q_prev, tgt, R_ref, -0.91, settle=a.ramp_settle)
        row("ramp %d/%d" % (k, a.ramp), tgt[2])
    row("抬升末", pin[2] + a.lift_m)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
