#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为什么 ZhiXing 抓不住？—— 受控机理测量（不再靠推断）

已确立的事实
------------
* 运动学/执行器健康：空载开闭 std=0、六向跟踪 ≤2 mm；
* 闭合时**确实**形成横向对夹：法向 ±y、法向力 ~30 N、接触点落在试块两侧面之间；
* 但对刚体穿透 6~20 mm，抬升 5 cm 时试块只动 ~2 mm，指垫最终滑到试块顶棱。

核心疑问
--------
30 N 横向夹持力，按 μ≈1 计摩擦容量约 30 N，而试块仅重 **1.44 N** —— 本应轻松提起。
所以要么「名义夹持力不是真的」，要么「有效摩擦远小于名义值」。

本脚本用**受控外力**直接测，不靠推断：
  1. 静力夹持读数：执行器真实输出力、关节角、接触的有符号距离、
     **MuJoCo 实际生效的接触摩擦系数 `contact.friction`**；
  2. 给试块施加**已知的竖直外力** `d.xfrc_applied`，逐步加大，
     找到「试块被拉动」的阈值 → **真实保持力 F_hold**；
  3. μ_eff = F_hold / Σ法向力；
  4. 与试块自重的 1.44 N 对比 —— 直接回答「为什么抬升时滑脱」。

用法
----
  --mode clamp    只做静力夹持 + 竖直拉拔（默认）
  --configs ...   顺带跑参数隔离（见 CONFIGS）
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N      # noqa: E402
import n3r_block_test as B     # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"
PAD_KEYS = ("fingerpad_col", "pad_collision")

# 参数隔离配置：对**已编译模型**直接改写 geom 参数（MuJoCo model 数组可写）
CONFIGS = {
    "baseline": {},
    "stiff": {"pad_solref": [0.002, 1.0], "pad_solimp": [0.99, 0.999, 0.0001, 0.5, 2.0]},
    "fric": {"pad_friction": [2.0, 0.05, 0.0001],
             "block_friction": [2.0, 0.05, 0.0001]},
    "stiff_fric": {"pad_solref": [0.002, 1.0], "pad_solimp": [0.99, 0.999, 0.0001, 0.5, 2.0],
                   "pad_friction": [2.0, 0.05, 0.0001],
                   "block_friction": [2.0, 0.05, 0.0001]},
    "stiff_fric_condim6": {"pad_solref": [0.002, 1.0],
                           "pad_solimp": [0.99, 0.999, 0.0001, 0.5, 2.0],
                           "pad_friction": [2.0, 0.05, 0.0001],
                           "block_friction": [2.0, 0.05, 0.0001],
                           "pad_condim": 6},
    # 真机 CTAG2F90 最大夹持力 23 N；仿真**没有力上限**，会一路压到穿透。
    # 下面按实测比值（ΣfN/torque ≈ 17.3 N per unit）折算等效力矩上限。
    "cap23N": {"torque_cap": 1.33},
    "cap12N": {"torque_cap": 0.70},
    "cap23N_stiff_fric": {"torque_cap": 1.33, "pad_solref": [0.002, 1.0],
                          "pad_solimp": [0.99, 0.999, 0.0001, 0.5, 2.0],
                          "pad_friction": [2.0, 0.05, 0.0001],
                          "block_friction": [2.0, 0.05, 0.0001]},
}


def pad_ids(sim):
    import mujoco

    m = sim.model._model
    out = []
    for g in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if any(k in n for k in PAD_KEYS):
            out.append(g)
    return out


def block_geoms(sim, bid):
    m = sim.model._model
    return [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid]


def apply_cfg(env, bid, cfg):
    """把配置写到已编译的 model 数组上（不需要重编译）。"""
    sim, m, d, _ = N.ids(env)
    note = []
    if "pad_solref" in cfg:
        for g in pad_ids(sim):
            m.geom_solref[g] = cfg["pad_solref"]
        note.append("pad_solref=%s" % cfg["pad_solref"])
    if "pad_solimp" in cfg:
        for g in pad_ids(sim):
            m.geom_solimp[g] = cfg["pad_solimp"]
        note.append("pad_solimp=%s" % cfg["pad_solimp"])
    if "pad_friction" in cfg:
        for g in pad_ids(sim):
            m.geom_friction[g] = cfg["pad_friction"]
        note.append("pad_friction=%s" % cfg["pad_friction"])
    if "pad_condim" in cfg:
        for g in pad_ids(sim):
            m.geom_condim[g] = cfg["pad_condim"]
        note.append("pad_condim=%d" % cfg["pad_condim"])
    if "block_friction" in cfg:
        for g in block_geoms(sim, bid):
            m.geom_friction[g] = cfg["block_friction"]
        note.append("block_friction=%s" % cfg["block_friction"])
    if cfg.get("torque_cap") is not None:
        aid = N.gripper_actuator(env)
        cap = float(cfg["torque_cap"])
        m.actuator_forcerange[aid] = [-cap, cap]
        m.actuator_forcelimited[aid] = 1
        note.append("torque_cap=%.3f（等效夹持力约 %.1f N）" % (cap, cap * 17.3))
    return note


def pad_face_report(env, bid, label, verbose=True):
    """指垫**内面中点**与试块的几何关系（判断指垫到底在试块侧面还是上面）。"""
    import mujoco

    sim, m, d, gid = N.ids(env)
    p_gs = np.array(d.site_xpos[gid], dtype=float)
    R_gs = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)

    def mesh_world(g):
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = np.array(m.mesh_vert[a:a + n], dtype=float)
        R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
        return (R @ v.T).T + np.array(d.geom_xpos[g], dtype=float)

    ids = pad_ids(sim)
    P = {g: mesh_world(g) for g in ids}
    faces = {}
    for g in ids:
        u = (np.mean([P[h].mean(axis=0) for h in ids if h != g], axis=0) - P[g].mean(axis=0))
        u /= np.linalg.norm(u)
        proj = P[g] @ u
        faces[g] = P[g][proj >= proj.max() - 0.002].mean(axis=0)

    blk = N.body_world_pts(sim, bid, collision_only=True)
    bz0, bz1 = float(blk[:, 2].min()), float(blk[:, 2].max())
    rep = {"label": label, "block_z": [bz0, bz1], "pads": []}
    if verbose:
        print("      [几何] 试块 z∈[%.4f, %.4f]  grip_site_z=%.4f" % (bz0, bz1, p_gs[2]))
    for g in ids:
        f = faces[g]
        loc = R_gs.T @ (f - p_gs)
        rep["pads"].append({"geom": gname(m, g),
                            "face_world": [float(v) for v in f],
                            "local_z_mm": float(loc[2]) * 1000,
                            "face_z": float(f[2])})
        if verbose:
            print("         %-34s 内面中点 z=%.4f  相对 grip_site 局部 z=%+7.2f mm  "
                  "%s" % (gname(m, g), f[2], loc[2] * 1000,
                          "在试块高度范围内" if bz0 <= f[2] <= bz1 else "★ 在试块高度范围之外"))
    return rep


def gname(m, g):
    import mujoco

    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ("geom%d" % g)


def contact_dump(env, bid, label, verbose=True):
    """打印/汇总涉及试块的接触，含 MuJoCo 实际生效的摩擦系数。"""
    import mujoco

    sim, m, d, _ = N.ids(env)
    rows, sum_fN = [], 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if int(m.geom_bodyid[g1]) != bid and int(m.geom_bodyid[g2]) != bid:
            continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, d, i, f)
        try:
            fr = [float(v) for v in c.friction]
        except Exception:
            fr = None
        row = {"geoms": [gname(m, g1), gname(m, g2)],
               "dist_mm": float(c.dist) * 1000,
               "fN": float(f[0]), "fT": [float(f[1]), float(f[2])],
               "normal": [float(v) for v in c.frame[:3]],
               "friction": fr, "dim": int(c.dim)}
        rows.append(row)
        if "fingerpad" in row["geoms"][0] + row["geoms"][1] or \
           "pad_collision" in row["geoms"][0] + row["geoms"][1]:
            sum_fN += float(f[0])
        if verbose:
            print("      %s <-> %s  dist=%+7.3fmm  fN=%7.3fN |fT|=%6.3f  "
                  "normal=%s  c.friction=%s" %
                  (row["geoms"][0], row["geoms"][1], row["dist_mm"], row["fN"],
                   float(np.linalg.norm(row["fT"])), np.round(row["normal"], 3).tolist(),
                   row["friction"]))
    if verbose and not rows:
        print("      （无涉及试块的接触）")
    return rows, sum_fN


def step_locked(env, q_arm, grip_ctrl, bid, force=None, n=1):
    """锁臂 + 驱动夹爪 + 可选给试块施加外力，步进 n 次。"""
    sim, m, d, gid, aj, aadr, adof = N._arm_idx(env)
    aid = N.gripper_actuator(env)
    for _ in range(n):
        N.set_arm(sim, m, d, aadr, adof, q_arm)
        d.ctrl[aid] = float(grip_ctrl)
        if force is not None:
            d.xfrc_applied[bid] = force
        sim.step()


def block_mean(env, bid):
    sim = N.ids(env)[0]
    return N.body_world_pts(sim, bid).mean(axis=0)


def run_config(env, bid, cfg_name, cfg, args):
    import mujoco

    env.reset()
    for _ in range(args.settle_steps):
        env.step(np.concatenate([np.zeros(6), [0.0]]))

    sim, m, d, gid = N.ids(env)
    bid = B.block_id(sim)
    pts = N.body_world_pts(sim, bid, collision_only=True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1]))
    z_g = float(lo[2]) + args.grasp_above_table

    notes = apply_cfg(env, bid, cfg)
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    aadr = N._arm_idx(env)[5]
    q_home = np.array([d.qpos[x] for x in aadr], dtype=float)
    obj0 = N.body_world_pts(sim, bid).mean(axis=0)
    pin = np.array([cx, cy, z_g])

    print("\n" + "=" * 74)
    print("[cfg %s] %s" % (cfg_name, "; ".join(notes) if notes else "（未改动）"))
    print("   试块体积质量 = %.4f kg  自重 = %.3f N" %
          (m.body_mass[bid], m.body_mass[bid] * 9.81))

    q_pre, r1 = N.ik_goto(env, q_home, pin + np.array([0, 0, 0.06]), R_ref, 0.0, settle=15)
    q_g, r2 = N.ik_goto(env, q_pre, pin, R_ref, 0.0, settle=15)
    step_locked(env, q_g, -0.91, bid, force=None, n=args.close_steps)

    sim, m, d, _ = N.ids(env)
    aid = N.gripper_actuator(env)
    l1 = None
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.endswith("Left_1_Joint"):
            l1 = float(d.qpos[int(m.jnt_qposadr[j])])
    print("  [solver] solver=%d iterations=%d tolerance=%.1e noslip_iter=%d  "
          "impratio=%.1f  cone=%d" % (int(m.opt.solver), int(m.opt.iterations),
          float(m.opt.tolerance), int(m.opt.noslip_iterations),
          float(m.opt.impratio), int(m.opt.cone)))
    print("  [静力夹持] IK残差=%.3fmm  Left_1_Joint qpos=%.5f (ctrl=-0.91000)  "
          "执行器输出力=%.3f" % (r2, l1 if l1 is not None else float("nan"),
                              float(d.actuator_force[aid])))
    print("             试块位移(闭合后) = %+.3f mm" %
          (float((block_mean(env, bid)[2] - obj0[2]) * 1000)))
    rows, sum_fN = contact_dump(env, bid, "clamp")
    print("             Σ法向力(指垫↔试块) = %.3f N" % sum_fN)
    geo = pad_face_report(env, bid, "clamp")

    # ---- 竖直拉拔：施加已知外力，找保持力阈值 ----
    print("  [竖直拉拔] 逐步施加向上外力，记录试块抬升")
    z_prev = float(block_mean(env, bid)[2])
    hold_N, hold_sum_fN = None, None
    trace = []
    F = 0.0
    while F < args.max_pull_N:
        F += args.pull_step_N
        step_locked(env, q_g, -0.91, bid, force=[0.0, 0.0, F, 0.0, 0.0, 0.0],
                    n=args.pull_steps_per)
        sim, m, d, _ = N.ids(env)
        d.xfrc_applied[bid] = np.zeros(6)
        z = float(block_mean(env, bid)[2])
        dz = (z - z_prev) * 1000
        _, sfn = contact_dump(env, bid, "pull", verbose=False)
        trace.append({"F_N": F, "block_dz_mm": (z - obj0[2]) * 1000,
                      "step_dz_mm": dz, "sum_fN": sfn})
        if (z - obj0[2]) * 1000 > args.lift_threshold_mm:
            hold_N, hold_sum_fN = F, sfn
            print("      F=%5.2f N → 试块累计抬升 %+6.2f mm  ★ 判定为被抓动"
                  % (F, (z - obj0[2]) * 1000))
            break
        z_prev = z
        if int(F / args.pull_step_N) % 5 == 0:
            print("      F=%5.2f N → 试块累计抬升 %+6.3f mm (ΣfN=%.2f N)"
                  % (F, (z - obj0[2]) * 1000, sfn))

    weight = float(m.body_mass[bid] * 9.81)
    res = {"config": cfg_name, "notes": notes, "ik_res_mm": r2,
           "left_1_qpos": l1, "actuator_force": float(d.actuator_force[aid]),
           "sum_fN_clamp": sum_fN, "contacts_clamp": rows, "geometry_clamp": geo,
           "block_weight_N": weight, "hold_force_N": hold_N,
           "sum_fN_at_hold": hold_sum_fN,
           "mu_eff": (hold_N / hold_sum_fN) if (hold_N and hold_sum_fN) else None,
           "trace": trace}
    if hold_N is None:
        print("      → **在 %.1f N 内未能拉动试块**（保持力 > %.1f N）"
              % (args.max_pull_N, args.max_pull_N))
    else:
        print("      → 保持力 F_hold = %.2f N   试块自重 = %.2f N   "
              "比值 = %.2f×" % (hold_N, weight, hold_N / weight))
        if hold_sum_fN:
            print("      → 有效摩擦 μ_eff = F_hold / ΣfN = %.2f / %.2f = **%.3f**"
                  % (hold_N, hold_sum_fN, hold_N / hold_sum_fN))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mm", type=float, default=50.0)
    ap.add_argument("--height-mm", type=float, default=70.0)
    ap.add_argument("--depth-mm", type=float, default=60.0)
    ap.add_argument("--place-x", type=float, default=-0.22)
    ap.add_argument("--place-y", type=float, default=0.10)
    ap.add_argument("--drop-z", type=float, default=1.30)
    ap.add_argument("--settle-steps", type=int, default=400)
    ap.add_argument("--close-steps", type=int, default=600)
    ap.add_argument("--grasp-above-table", type=float, default=0.035)
    ap.add_argument("--max-pull-N", type=float, default=30.0)
    ap.add_argument("--pull-step-N", type=float, default=0.5)
    ap.add_argument("--pull-steps-per", type=int, default=40)
    ap.add_argument("--lift-threshold-mm", type=float, default=5.0)
    ap.add_argument("--configs", default="baseline")
    ap.add_argument("--tag", default="why")
    a = ap.parse_args()

    hy, hz = a.width_mm / 2000.0, a.height_mm / 2000.0
    hx = a.depth_mm / 2000.0
    B.install_block_injection((hx, hy, hz), (a.place_x, a.place_y, a.drop_z))

    names = [s.strip() for s in a.configs.split(",") if s.strip()]
    out = {"node": "N3-redo", "tag": a.tag, "purpose": "why_cannot_grasp",
           "block_mm": [a.depth_mm, a.width_mm, a.height_mm], "results": []}

    # 每个配置一个独立 env：model 数组被就地改写，不能跨配置复用
    for nm in names:
        cfg = CONFIGS.get(nm)
        if cfg is None:
            print("[skip] 未知配置 %s（可选：%s）" % (nm, list(CONFIGS)))
            continue
        env, task = N.build(task_id=0)
        env.reset()
        sim = N.ids(env)[0]
        bid = B.block_id(sim)
        if bid is None:
            raise RuntimeError("试块未注入")
        out["results"].append(run_config(env, bid, nm, cfg, a))
        env.close()

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(OUT) / ("n3r_%s.json" % a.tag)
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[out] %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
