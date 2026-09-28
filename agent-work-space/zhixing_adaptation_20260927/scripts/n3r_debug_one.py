#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3-redo 逐阶段诊断：抓取全程打印「试块 ↔ 指垫」的真实接触与几何关系，
定位「闭合净距正常但试块不跟随抬升」的原因。

诊断点（接近后 / 闭合后 / 抬升后）：
  - 试块世界位姿、沿闭合轴的跨度
  - 两个指垫 body 的世界位置与夹角轴
  - 所有涉及试块的接触：geom 名、有符号距离、法向力
  - 夹爪中心线是否穿过试块
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N      # noqa: E402
import n3r_block_test as B     # noqa: E402

PAD_KEYS = ("fingerpad_col", "pad_collision")


def pad_geoms(sim):
    import mujoco

    m = sim.model._model
    out = {}
    for g in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if not any(k in n for k in PAD_KEYS):
            continue
        if "left" in n:
            out.setdefault("left", []).append(g)
        elif "right" in n:
            out.setdefault("right", []).append(g)
    return out


def gname(m, g):
    import mujoco

    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"


def stage_report(env, bid, label):
    import mujoco

    sim, m, d, gid = N.ids(env)
    pads = pad_geoms(sim)
    blk = N.body_world_pts(sim, bid, collision_only=True)
    bp = blk.mean(axis=0)
    print(f"\n--- [{label}] block_mean={np.round(bp,4).tolist()} "
          f"z_range=[{blk[:,2].min():.4f},{blk[:,2].max():.4f}]")

    # 指垫 body 位置
    body_w = {}
    for side, gl in pads.items():
        bs = sorted({int(m.geom_bodyid[g]) for g in gl})
        for b in bs:
            body_w[side] = np.array(d.xpos[b], dtype=float)
    if "left" in body_w and "right" in body_w:
        axis = body_w["right"] - body_w["left"]
        L = float(np.linalg.norm(axis))
        axis_n = axis / L
        print(f"    pad bodies: L={np.round(body_w['left'],4).tolist()} "
              f"R={np.round(body_w['right'],4).tolist()}  |LR|={L*1000:.1f}mm "
              f"axis={np.round(axis_n,3).tolist()}")
        mid = 0.5 * (body_w["left"] + body_w["right"])
        proj = (blk - mid) @ axis_n
        print(f"    闭合轴跨度: block=[{proj.min()*1000:+.1f},{proj.max()*1000:+.1f}]mm "
              f"中心偏移={float((bp-mid)@axis_n)*1000:+.1f}mm; "
              f"垂直偏移={np.linalg.norm((bp-mid)-((bp-mid)@axis_n)*axis_n)*1000:.1f}mm")

    # 接触
    n_ct = 0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if int(m.geom_bodyid[g1]) != bid and int(m.geom_bodyid[g2]) != bid:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        n_ct += 1
        nrm = np.array(c.frame[:3], dtype=float)
        print(f"    contact[{i}] {gname(m,g1)} <-> {gname(m,g2)}  "
              f"dist={float(c.dist)*1000:+7.3f}mm  |F|={np.linalg.norm(f[:3]):7.3f}N  "
              f"fN={f[0]:7.3f} fT=({f[1]:+.2f},{f[2]:+.2f})  "
              f"normal={np.round(nrm,3).tolist()}  pos={np.round(c.pos,4).tolist()}")
    if n_ct == 0:
        print("    （无涉及试块的接触）")
    return bp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mm", type=float, default=50.0)
    ap.add_argument("--height-mm", type=float, default=70.0)
    ap.add_argument("--depth-mm", type=float, default=60.0)
    ap.add_argument("--gripper", default="ZhixingGripper")
    ap.add_argument("--padfix-mode", choices=["none", "add", "replace"], default="none")
    ap.add_argument("--place-x", type=float, default=-0.22)
    ap.add_argument("--place-y", type=float, default=0.10)
    ap.add_argument("--drop-z", type=float, default=1.30)
    ap.add_argument("--settle-steps", type=int, default=400)
    ap.add_argument("--close-steps", type=int, default=600)
    ap.add_argument("--grasp-above-table", type=float, default=0.035)
    a = ap.parse_args()

    gname_use = a.gripper
    if a.padfix_mode != "none":
        import json
        import pathlib
        import n3r_padfix as PF
        pj = json.loads((pathlib.Path(PF.OUT) / "n3r_pad_collision.json").read_text())
        path, notes = PF.build_variant_xml(a.padfix_mode, pj)
        gname_use = PF.register_variant(path)
        print(f"[padfix] mode={a.padfix_mode}")
        for n in notes:
            print("   ", n)

    hy, hz = a.width_mm / 2000.0, a.height_mm / 2000.0
    hx = a.depth_mm / 2000.0
    B.install_block_injection((hx, hy, hz), (a.place_x, a.place_y, a.drop_z))
    env, task = N.build(task_id=0, gripper=gname_use)
    env.reset()
    for _ in range(a.settle_steps):
        env.step(np.concatenate([np.zeros(6), [0.0]]))

    sim, m, d, gid = N.ids(env)
    bid = B.block_id(sim)
    pts = N.body_world_pts(sim, bid, collision_only=True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1]))
    z_g = float(lo[2]) + a.grasp_above_table
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    print(f"[axis] grip_site 局部轴在世界系（列）:\n{np.round(R_ref,3)}")
    print(f"       接近轴(local +z) -> world {np.round(R_ref[:,2],3).tolist()}")
    print(f"       闭合轴(local +y) -> world {np.round(R_ref[:,1],3).tolist()}")
    print(f"[geom] block xy=({cx:.4f},{cy:.4f}) z_g={z_g:.4f}")

    _, _, _, _, _, aadr, adof = N._arm_idx(env)
    q_home = np.array([d.qpos[x] for x in aadr], dtype=float)
    obj0 = N.body_world_pts(sim, bid).mean(axis=0)
    pin = np.array([cx, cy, z_g])
    stage_report(env, bid, "reset/settled")

    q_pre, r1 = N.ik_goto(env, q_home, pin + np.array([0, 0, 0.06]), R_ref, 0.0, settle=15)
    print(f"\n[ik] pre  res={r1:.3f}mm")
    stage_report(env, bid, "after approach(pre)")

    q_g, r2 = N.ik_goto(env, q_pre, pin, R_ref, 0.0, settle=15)
    print(f"\n[ik] grasp res={r2:.3f}mm")
    stage_report(env, bid, "descended")

    N.step_hold(env, q_g, -0.91, n=a.close_steps)
    bp_close = stage_report(env, bid, "after close")

    q_up, r3 = N.ik_goto(env, q_g, pin + np.array([0, 0, 0.05]), R_ref, -0.91, settle=15)
    print(f"\n[ik] lift res={r3:.3f}mm")
    bp_up = stage_report(env, bid, "after lift")

    N.step_hold(env, q_up, -0.91, n=40)
    bp_hold = stage_report(env, bid, "after hold 2s")
    print(f"\n[result] dz_close={float(bp_close[2]-obj0[2])*1000:+.1f}mm "
          f"dz_lift={float(bp_up[2]-obj0[2])*1000:+.1f}mm "
          f"dz_hold={float(bp_hold[2]-obj0[2])*1000:+.1f}mm")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
