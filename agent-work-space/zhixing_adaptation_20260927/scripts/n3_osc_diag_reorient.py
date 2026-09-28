#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重定向抓取失败的**机理诊断**（回答审核的三个问题）。

1. 重定向后夹爪的**闭合轴**相对试块宽度方向是什么关系？夹取面是否平行于目标截面？
2. 深穿透（最高 10.4 mm）发生在哪个阶段？撞的是试块**顶面**还是**侧面**？
3. 重定向是否改变了 `grasp_pinch_target` 的计算？
4. 归因：是"当前策略在带姿态偏转时不可用"（真实局限），还是"探针实现有缺陷"（脚本问题）？

做法：与主脚本同一链路（同一 Episode/servo/OSC 动作），但逐控制步打印
grip_site / 指垫世界位置 / 试块 bbox / 臂关节值与限位余量 / 指垫↔试块接触
（geom 名、dist、接触点世界坐标与试块顶面 z 的比较）。
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N          # noqa: E402
import n4_collect as N4            # noqa: E402
import n3_osc_full_chain as F      # noqa: E402


def arm_joints(env):
    """返回 [(关节名, qpos, lo, hi, 距最近限位)]。"""
    import mujoco

    sim, m, d, _ = N.ids(env)
    out = []
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not nm.startswith("robot0_joint"):
            continue
        a = int(m.jnt_qposadr[j])
        lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
        q = float(d.qpos[a])
        out.append((nm.replace("robot0_joint", "J"), q, lo, hi,
                    min(q - lo, hi - q)))
    return out


def contacts_pad_block(env, bid):
    """指垫↔试块接触：geom 名、dist(mm)、接触点世界坐标。"""
    import mujoco

    sim, m, d, _ = N.ids(env)
    pads = F.pad_geoms(sim)
    pset = set(pads.values())
    rows = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if not ((g1 in pset) ^ (g2 in pset)):
            continue
        other = g2 if g1 in pset else g1
        if int(m.geom_bodyid[other]) != bid:
            continue
        pid = g1 if g1 in pset else g2
        n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, pid) or "?"
        rows.append({"pad": n1.split("_")[0],
                     "dist_mm": float(c.dist) * 1000.0,
                     "pos": [float(v) for v in c.pos]})
    return rows


def snap(env, bid, tag, extra=""):
    sim, m, d, gid = N.ids(env)
    p, R = N.gripsite(sim, m, d, gid)
    pads = F.pad_geoms(sim)
    pl = np.array(d.geom_xpos[pads["left"]], dtype=float)
    pr = np.array(d.geom_xpos[pads["right"]], dtype=float)
    bb = N4.bbox(sim, bid)
    ct = contacts_pad_block(env, bid)
    ax = F.closing_axis_world(env)
    print("    %-11s gs=%s pad_mid_z=%.4f" % (tag, np.round(p, 4).tolist(),
                                              0.5 * (pl[2] + pr[2])))
    print("                block z=[%.4f,%.4f] objxy=%s 闭合轴=%s"
          % (bb[0][2], bb[1][2], np.round(0.5 * (bb[0] + bb[1]), 4).tolist(),
             "n/a" if ax is None else np.round(ax, 3).tolist()))
    print("                " + "  ".join("%s=%.3f(余%.3f)" % (n, q, mg)
                                        for n, q, lo, hi, mg in arm_joints(env)))
    for c in ct[:3]:
        where = ("顶面附近" if abs(c["pos"][2] - bb[1][2]) < 0.008
                 else "侧面" if c["pos"][2] < bb[1][2] - 0.008 else "顶面上方")
        print("                CONTACT %s dist=%+.3fmm pos_z=%.4f top=%.4f → %s"
              % (c["pad"], c["dist_mm"], c["pos"][2], bb[1][2], where))
    if not ct:
        print("                CONTACT none")
    if extra:
        print("                " + extra)
    return p, R, bb, ax


def descend_step(ep, env, pin, R_ref, grip):
    sim, m, d, gid = N.ids(env)
    p, R = N.gripsite(sim, m, d, gid)
    nd = float(np.linalg.norm(pin - p))
    dv = pin - p
    if nd > F.SERVO_MAX_STEP_POS_M:
        dv = dv * (F.SERVO_MAX_STEP_POS_M / nd)
    ap = np.clip(dv / F.TEACHER_GAIN_POS_M, -1, 1)
    ap = np.clip(ap, -F.SERVO_MAX_STEP_POS_M / F.OSC_POS_UNIT_M,
                 F.SERVO_MAX_STEP_POS_M / F.OSC_POS_UNIT_M)
    er = N.rot_err_vec(R_ref, R)
    ne = float(np.linalg.norm(er))
    if ne > F.SERVO_MAX_STEP_ROT_RAD:
        er = er * (F.SERVO_MAX_STEP_ROT_RAD / ne)
    ar = np.clip(er / F.TEACHER_GAIN_ROT_RAD, -1, 1)
    ar = np.clip(ar, -F.SERVO_MAX_STEP_ROT_RAD / F.OSC_ROT_UNIT_RAD,
                 F.SERVO_MAX_STEP_ROT_RAD / F.OSC_ROT_UNIT_RAD)
    ep.act(np.concatenate([ap, ar, [grip]]), "descend")
    return nd, ne


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mm", type=float, default=70.0)
    ap.add_argument("--yaw-deg", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=41900)
    ap.add_argument("--max-descend", type=int, default=90)
    ap.add_argument("--max-close", type=int, default=400)
    a = ap.parse_args()

    F.install_object_substitution(
        "alphabet_soup",
        F.block_object_class(a.width_mm / 1000.0, F.BLOCK_DEPTH_M, F.BLOCK_HEIGHT_M),
        lambda s: print(s))
    env, task = N4.build("libero_object", 0)
    ep = F.ChainEpisode(env)
    env.seed(a.seed)
    ep.start()
    obs = N4.quiet_hold(env, 120, F.GRIP_OPEN)
    ep.raw_steps += 120
    ep.refresh_obs(obs, "prep")
    sim, m, d, gid = N.ids(env)
    obj_name, region_name = N4.goal_objects(env)
    bid, bname = N4.find_body(sim, obj_name)
    ep.bid = bid
    ep.act(np.concatenate([np.zeros(6), [F.GRIP_OPEN]]), "settle")
    F.arm_ik_guard()

    sim, m, d, gid = N.ids(env)
    ob, oh = N4.bbox(sim, bid)
    obj_c0 = 0.5 * (ob + oh)
    R_home = N.gripsite(sim, m, d, gid)[1].copy()
    axis_w = F.closing_axis_world(env)
    phi = float(np.arctan2(axis_w[1], axis_w[0]))
    yaw = float(np.radians(a.yaw_deg))
    obj_yaw = float(phi + yaw - np.pi / 2.0)
    print("=" * 78)
    print("[0] 试块 bbox = %s .. %s  (%.1f x %.1f x %.1f mm)"
          % (np.round(ob, 4).tolist(), np.round(oh, 4).tolist(),
             (oh[0] - ob[0]) * 1000, (oh[1] - ob[1]) * 1000, (oh[2] - ob[2]) * 1000))
    print("    R_home[:,2](工具轴) = %s   闭合轴(home) = %s"
          % (np.round(R_home[:, 2], 4).tolist(), np.round(axis_w, 4).tolist()))
    print("    pinch_target(pin) = %s   ← 只由物体 bbox 决定，与夹爪姿态无关"
          % np.round(obj_c0, 4).tolist())
    F.set_object_yaw(env, bid, obj_c0[:2], obj_yaw)
    sim, m, d, gid = N.ids(env)
    ob, oh = N4.bbox(sim, bid)
    obj_c0 = 0.5 * (ob + oh)
    yaw_now = F.object_yaw(env, bid)
    wdir = np.array([-np.sin(yaw_now), np.cos(yaw_now), 0.0])
    print("[1] 物体偏航设为 %.3f rad (%.2f°) → bbox = %.1f x %.1f x %.1f mm"
          % (obj_yaw, np.degrees(obj_yaw), (oh[0] - ob[0]) * 1000,
             (oh[1] - ob[1]) * 1000, (oh[2] - ob[2]) * 1000))
    print("    试块宽度方向(局部 y) 世界系 = %s" % np.round(wdir, 4).tolist())
    h = float(oh[2] - ob[2])
    z_g = float(min(ob[2] + max(0.028, min(0.5 * h, h - 0.006)), oh[2] - 0.004))
    pin = np.array([obj_c0[0], obj_c0[1], z_g])
    print("    pin(z_g=%.4f) = %s" % (z_g, np.round(pin, 4).tolist()))
    snap(env, bid, "pre-reorient")

    R_ref = F.rotz(yaw) @ R_home
    print("\n[2] 重定向目标 = rotz(%.0f°) @ R_home（世界系偏航）；期望闭合轴 = %s"
          % (a.yaw_deg, np.round(F.rotz(yaw) @ axis_w, 4).tolist()))
    p0 = N.gripsite(*N.ids(env))[0].copy()
    ok, k = ep.servo_limited(p0, R_ref, F.GRIP_OPEN, "reorient", 300,
                             tol=0.005, rot_tol=0.03)
    R_now = N.gripsite(*N.ids(env))[1]
    print("    收敛=%s(rot=%s) 步数=%d 姿态残差=%.4f rad"
          % (ok, ep.conv["rot"], k,
             float(np.linalg.norm(N.rot_err_vec(R_ref, R_now)))))
    p, R, bb, ax = snap(env, bid, "post-reorient")
    if ax is not None:
        cosang = float(np.clip(abs(np.dot(ax, wdir)), 0, 1))
        print("    ★ 闭合轴 vs 试块宽度方向夹角 = %.2f°（0° = 夹取面正好平行于目标截面）"
              % np.degrees(np.arccos(cosang)))

    # 对照：不做重定向，同样下降，看是否也漂
    print("\n[3] 下降（逐步；7 维 OSC 动作，无锁臂、无 IK）")
    reached = False
    for st in range(a.max_descend):
        nd, ne = descend_step(ep, env, pin, R_ref, F.GRIP_OPEN)
        sim, m, d, gid = N.ids(env)
        p2 = np.array(d.site_xpos[gid], dtype=float)
        dxy = float(np.linalg.norm(p2[:2] - pin[:2]))
        ct = contacts_pad_block(env, bid)
        if st < 4 or st % 4 == 0 or ct or nd < 0.008:
            snap(env, bid, "desc%02d" % st,
                 "|xy-pin|=%.1fmm |z-pin|=%.1fmm |e|=%.1fmm |rot_e|=%.4f"
                 % (dxy * 1000, (p2[2] - pin[2]) * 1000, nd * 1000, ne))
        if nd < 0.008:
            reached = True
            print("    → 下降到位")
            break
    if not reached:
        print("    → 下降停滞（%d 步未到位）" % a.max_descend)

    print("\n[4] 闭合（逐步；检查指垫是夹在侧面还是压在顶面）")
    for st in range(a.max_close):
        sim, m, d, gid = N.ids(env)
        p, R = N.gripsite(sim, m, d, gid)
        dv = pin - p
        nd = float(np.linalg.norm(dv))
        if nd > F.SERVO_MAX_STEP_POS_M:
            dv = dv * (F.SERVO_MAX_STEP_POS_M / nd)
        ap = np.clip(dv / F.TEACHER_GAIN_POS_M, -1, 1)
        ap = np.clip(ap, -F.SERVO_MAX_STEP_POS_M / F.OSC_POS_UNIT_M,
                     F.SERVO_MAX_STEP_POS_M / F.OSC_POS_UNIT_M)
        er = N.rot_err_vec(R_ref, R)
        ne = float(np.linalg.norm(er))
        if ne > F.SERVO_MAX_STEP_ROT_RAD:
            er = er * (F.SERVO_MAX_STEP_ROT_RAD / ne)
        ar = np.clip(er / F.TEACHER_GAIN_ROT_RAD, -1, 1)
        ar = np.clip(ar, -F.SERVO_MAX_STEP_ROT_RAD / F.OSC_ROT_UNIT_RAD,
                     F.SERVO_MAX_STEP_ROT_RAD / F.OSC_ROT_UNIT_RAD)
        ep.act(np.concatenate([ap, ar, [F.GRIP_CLOSE]]), "close")
        if st < 3 or st % 15 == 0:
            snap(env, bid, "close%03d" % st,
                 "Left_1=%.4f jaw=%.1fmm" % (N4.left1(env),
                                             N.jaw_face_separation_mm(env) or -1))
        l1 = N4.left1(env)
        if st > 60 and abs(l1 + 1.0) < 0.03:
            print("    → 夹爪已到行程末端（夹空/被挡）")
            break
    p, R, bb, ax = snap(env, bid, "close-end")
    print("\n[5] 要点：Left_1=%.4f  jaw=%.2f mm  物体中心=%s"
          % (N4.left1(env), N.jaw_face_separation_mm(env) or -1,
             np.round(0.5 * (bb[0] + bb[1]), 4).tolist()))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
