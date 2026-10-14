#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3：ZhiXing 夹爪抓持能力验证（G0→1 闸门）。

方法学铁律
==========
**不得用 π0.5 策略的成功率判定夹爪好坏。** 策略在 ZhiXing 上有视觉 OOD
（腕部画面分布不同 → 不产生抓取时序），0/10 反映的是策略、不是夹爪。
因此本脚本**完全不加载策略**，改用**脚本专家 + 底层 OSC 控制**产生动作，
把「夹爪能不能抓」与「策略想不想抓」彻底分开。

判据（来自 `实现节点与阶段性验证.md` N3）
  (a) 抓持 ≥ 9/10
  (b) 无穿透、无数值发散
  (c) 成功是真抓持（抬升并保持），不是推碰/摆放伪成功
  (d) 空载开闭重复性一致

三个部分
  A. 空载开闭 ×10 重复性
  B. 6 向小幅位姿跟踪（闭环 plant 检查）
  C. 抓取—抬升 5 cm—保持 2 s—放回 ×10

专家不走 π0.5。专家的目标点直接用**实测夹持点**表达，因此本测试与
「TCP 定义是否已修」无关 —— 测的是夹爪本身。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

RES = 256
TASK_SUITE = "libero_spatial"
DEFAULT_OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"

# 实测（N2，闭合位）：夹持点相对 grip_site 的局部偏移
GRASP_OFFSET_CLOSED = np.array([-0.00006, 0.00002, 0.08254])
GRASP_OFFSET_OPEN = np.array([-0.00006, 0.00002, 0.06511])
FINGERTIP_BEYOND_PINCH_M = 0.0199          # 指尖比夹持点再深 ~19.9 mm
# 实测（probe_action_frame.py）：LIBERO/robosuite OSC 的动作
#   action[0:3] = **世界系**平移增量（不是 eef 系！），满动作约 10 mm/env-step
#   action[3:6] = 世界系旋转增量，满动作约 0.078 rad/env-step
#   action[6]   = 夹爪：+1 闭合 / -1 张开（N1 极性修复后）
OSC_POS_SCALE = 0.02
OSC_ROT_SCALE = 0.05


# --------------------------------------------------------------------------- #
def build(robot="ECO65", gripper="ZhixingGripper", task_id=0, seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kw = {"bddl_file_name": bddl, "camera_heights": RES, "camera_widths": RES, "robots": [robot]}
    if gripper != "default":
        kw["gripper_types"] = gripper
    env = OffScreenRenderEnv(**kw)
    env.seed(seed)
    return env, task


def ids(env):
    import mujoco

    sim = env.env.sim
    m, d = sim.model._model, sim.data._data
    gid = next(i for i in range(m.nsite)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or "").endswith("grip_site"))
    return sim, m, d, gid


def gripsite(sim, m, d, gid):
    return (np.array(d.site_xpos[gid], dtype=float),
            np.array(d.site_xmat[gid], dtype=float).reshape(3, 3))


def rot_err_vec(R_target, R_cur):
    R = R_target @ R_cur.T
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    s = np.linalg.norm(v)
    if s < 1e-12:
        return v
    ang = np.arctan2(s, (np.trace(R) - 1.0) / 2.0)
    return v / s * ang


def body_world_pts(sim, bid, collision_only=False):
    """物体该 body 下所有 geom 的世界顶点。

    collision_only=True 时只取 **碰撞 geom**（contype!=0）—— 抓取规划必须用
    碰撞几何，视觉网格（往往是大 mesh）与碰撞分解（常为多个 box）范围可能不同，
    用视觉网格规划会让手指从碰撞体旁边滑过去（实测踩到过）。
    """
    import mujoco

    m, d = sim.model._model, sim.data._data
    pts = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) != bid:
            continue
        if collision_only and m.geom_contype[g] == 0:
            continue
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = np.array(m.mesh_vert[a:a + n], dtype=float)
        else:
            s = np.array(m.geom_size[g], dtype=float)
            v = s * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
        pts.append((R @ v.T).T + np.array(d.geom_xpos[g], dtype=float))
    return np.vstack(pts) if pts else None


def find_object_bodies(sim, keyword):
    import mujoco

    m = sim.model._model
    out = []
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if keyword in n and n.endswith("_main"):
            out.append((b, n))
    return out


def free_joint_of(sim, body_name):
    """按名字前缀找该物体的 free joint 名。"""
    import mujoco

    m = sim.model._model
    stem = body_name.replace("_main", "")
    for j in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if n.startswith(stem) and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return n
    return None


def set_free_joint(sim, jname, xyz):
    import mujoco

    m, d = sim.model._model, sim.data._data
    jid = sim.model.joint_name2id(jname)
    a = int(m.jnt_qposadr[jid])
    d.qpos[a] = xyz[0]
    d.qpos[a + 1] = xyz[1]
    d.qpos[a + 2] = xyz[2]
    mujoco.mj_forward(m, d)



# --------------------------------------------------------------------------- #
# IK 控制（脚本专家用；与 π0.5 无关）
# --------------------------------------------------------------------------- #
def _arm_idx(env):
    import mujoco

    sim, m, d, gid = ids(env)
    aj = [i for i in range(m.njnt)
          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("robot0_joint")]
    aadr = [int(m.jnt_qposadr[j]) for j in aj]
    adof = [int(m.jnt_dofadr[j]) for j in aj]
    return sim, m, d, gid, aj, aadr, adof


def gripper_actuator(env):
    import mujoco

    sim, m, d, gid = ids(env)
    for i in range(m.nu):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        if n.startswith("gripper0_"):
            return i
    raise RuntimeError("找不到夹爪执行器")


def set_arm(sim, m, d, aadr, adof, q):
    for a, v in zip(aadr, q):
        d.qpos[a] = float(v)
    for a in adof:
        d.qvel[a] = 0.0


def step_hold(env, q_arm, gripper_ctrl, n=1):
    """锁住臂关节、驱动夹爪执行器、步进 n 次（物理/接触正常演化）。"""
    import mujoco

    sim, m, d, gid, aj, aadr, adof = _arm_idx(env)
    aid = gripper_actuator(env)
    for _ in range(n):
        set_arm(sim, m, d, aadr, adof, q_arm)
        d.ctrl[aid] = float(gripper_ctrl)
        sim.step()
    mujoco.mj_forward(m, d)


def solve_ik_gripsite(env, target_pos, target_mat, q_seed, iters=800, lam=0.05, alpha=0.6):
    """DLS IK：解 6 个臂关节，使 grip_site 达到 target 6D 位姿。"""
    import mujoco

    sim, m, d, gid, aj, aadr, adof = _arm_idx(env)
    jids = [sim.model.joint_name2id(
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) for j in aj]
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    q = np.clip(np.array(q_seed, dtype=float), lo + 1e-9, hi - 1e-9)
    for _ in range(iters):
        for a, v in zip(aadr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        cur_p = np.array(d.site_xpos[gid], dtype=float)
        cur_R = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)
        e = np.concatenate([target_pos - cur_p, rot_err_vec(target_mat, cur_R)])
        if np.linalg.norm(e) < 1e-6:
            break
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, gid)
        J = np.vstack([jacp[:, adof], jacr[:, adof]])
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), e)
        q = np.clip(q + alpha * dq, lo, hi)
    for a, v in zip(aadr, q):
        d.qpos[a] = v
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    return q, lo, hi


def ik_goto(env, q_seed, pinch_target, R_ref, gripper_ctrl, settle=12, n_seeds=6):
    """把夹持点开到 pinch_target（姿态保持 R_ref）。多初值重启，取残差最小者。"""
    import mujoco

    gs = np.asarray(pinch_target, dtype=float) - R_ref @ GRASP_OFFSET_CLOSED
    sim, m, d, gid, aj, aadr, adof = _arm_idx(env)
    best = None
    rng = np.random.default_rng(0)
    for k in range(n_seeds):
        seed = np.array(q_seed, dtype=float)
        if k > 0:
            seed = seed + (rng.random(len(seed)) - 0.5) * 0.6
        q, lo, hi = solve_ik_gripsite(env, gs, R_ref, seed, iters=1200)
        for a, v in zip(aadr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = np.array(d.site_xpos[gid], dtype=float)
        R = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)
        res = float(np.linalg.norm(p + R @ GRASP_OFFSET_CLOSED - np.asarray(pinch_target)) * 1000)
        if best is None or res < best[1]:
            best = (q.copy(), res)
    q = best[0]
    step_hold(env, q, gripper_ctrl, n=settle)
    sim, m, d, gid = ids(env)
    p = np.array(d.site_xpos[gid], dtype=float)
    R = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)
    return q, float(np.linalg.norm((p + R @ GRASP_OFFSET_CLOSED) - np.asarray(pinch_target)) * 1000)


# --------------------------------------------------------------------------- #
# 闭环移动（OSC）
# --------------------------------------------------------------------------- #
def drive_to(env, target_pos, gripper_cmd, max_steps=200, tol=0.004, rot_hold=True):
    """用 OSC 增量把 grip_site 开到 target_pos（保持当前姿态）。返回轨迹统计。"""
    sim, m, d, gid = ids(env)
    R0 = gripsite(sim, m, d, gid)[1].copy()
    errs, nan = [], False
    steps = 0
    for steps in range(1, max_steps + 1):
        p, R = gripsite(sim, m, d, gid)
        e_w = target_pos - p                       # 世界系
        act_p = np.clip(e_w / OSC_POS_SCALE, -1.0, 1.0)
        if rot_hold:
            er_w = rot_err_vec(R0, R)              # 世界系
            act_r = np.clip(er_w / OSC_ROT_SCALE, -1.0, 1.0)
        else:
            act_r = np.zeros(3)
        a = np.concatenate([act_p, act_r, [gripper_cmd]])
        env.step(a)
        p2 = gripsite(sim, m, d, gid)[0]
        if not np.all(np.isfinite(p2)):
            nan = True
            break
        errs.append(float(np.linalg.norm(target_pos - p2)))
        if errs[-1] < tol:
            break
    return {"steps": steps, "final_err_mm": errs[-1] * 1000 if errs else None,
            "nan": nan, "converged": bool(errs and errs[-1] < tol)}


def hold(env, gripper_cmd, steps):
    sim, m, d, gid = ids(env)
    p0 = gripsite(sim, m, d, gid)[0].copy()
    sim, m, d, gid = ids(env)
    R0 = gripsite(sim, m, d, gid)[1].copy()
    for _ in range(steps):
        p, R = gripsite(sim, m, d, gid)
        act_p = np.clip((p0 - p) / OSC_POS_SCALE, -1, 1)
        act_r = np.clip(rot_err_vec(R0, R) / OSC_ROT_SCALE, -1, 1)
        env.step(np.concatenate([act_p, act_r, [gripper_cmd]]))


def jaw_face_separation_mm(env):
    """两指垫内侧面净距（N2 权威口径的精简版：最近点对在此仅用于开合重复性对比）。"""
    import mujoco

    sim, m, d, _ = ids(env)
    gl = gr = None
    for g in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if "left_fingerpad_col" in n:
            gl = g
        if "right_fingerpad_col" in n:
            gr = g
    if gl is None or gr is None:
        return None
    def ctr(g):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = np.array(m.mesh_vert[a:a + n], dtype=float)
        else:
            v = np.zeros((1, 3))
        R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
        P = (R @ v.T).T + np.array(d.geom_xpos[g], dtype=float)
        return P
    Pl, Pr = ctr(gl), ctr(gr)
    # 各自朝对方最近的一薄层质心
    u = Pr.mean(axis=0) - Pl.mean(axis=0)
    u /= np.linalg.norm(u)
    fl = Pl[(Pl @ u) >= (Pl @ u).max() - 0.005].mean(axis=0)
    fr = Pr[(Pr @ (-u)) >= (Pr @ (-u)).max() - 0.005].mean(axis=0)
    return float(np.linalg.norm(fl - fr) * 1000)


# --------------------------------------------------------------------------- #
# 抓取可行性分析
# --------------------------------------------------------------------------- #
def grasp_plan(sim, bodies, closing_axis=1, opening_mm=89.96, margin_mm=6.0, band_m=0.006):
    """
    在物体几何上找可行抓取高度：沿闭合轴(y)的截面宽度 < 开口-裕度，
    且指尖（比夹持点深 19.9 mm）不会插到桌面以下。

    用 ±6 mm 的厚带来统计，避免稀疏网格顶点导致某些高度没有样本。
    """
    allv = np.vstack([body_world_pts(sim, b, collision_only=True) for b, _ in bodies])
    zmin, zmax = float(allv[:, 2].min()), float(allv[:, 2].max())
    table_z = zmin
    budget = (opening_mm - margin_mm) / 1000.0
    best, rows = None, []
    for z in np.arange(zmax, zmin - 1e-9, -0.002):
        slab = allv[np.abs(allv[:, 2] - z) <= band_m]
        if slab.shape[0] < 2:
            continue
        ext = slab.max(axis=0) - slab.min(axis=0)
        w_close = float(ext[closing_axis])
        fingertip_z = z - FINGERTIP_BEYOND_PINCH_M
        clear = fingertip_z - table_z
        ok = bool(w_close <= budget and clear > 0.002)
        rows.append([round(float(z), 4), round(w_close * 1000, 1), round(clear * 1000, 1), ok])
        if ok and (best is None or z < best[0]):
            best = (float(z), w_close, clear,
                    float((slab[:, 0].max() + slab[:, 0].min()) / 2),
                    float((slab[:, 1].max() + slab[:, 1].min()) / 2))
    return {"object_z_range": [zmin, zmax], "table_z": table_z, "n_vertices": int(allv.shape[0]),
            "opening_budget_mm": budget * 1000, "best": best, "scan": rows}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--part", default="all", choices=["all", "A", "B", "C", "plan"])
    ap.add_argument("--object", default="akita_black_bowl_1")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--strategy", default="rim", choices=["rim", "side"])
    ap.add_argument("--place-x", type=float, default=-0.20,
                    help="把物体摆到该 x（ECO65 垂直工具下的可达区）")
    ap.add_argument("--place-y", type=float, default=0.10)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--tag", default="", help="结果文件名后缀；默认按 part/strategy 生成")
    # [2026-09-27] A 项的 settle 预算。落地 forcerange=-2（厂商 URDF effort=2）后，
    # 空载全行程实测需 ~1450-1500 子步（开）/~510 子步（闭），而 20 Hz 下
    # 1 env step = 25 子步。原值 25 env step（=625 子步）留量不足，会让读数在
    # 65.9~90.1 mm 间跳动而被误判 FAIL。**判据本身（std < 0.5 mm）未改动**，
    # 只把 settle 时间调到足以走完全行程：80 env step = 2000 子步。
    ap.add_argument("--a-settle", type=int, default=80,
                    help="A 项每个相位给的 env step 数（默认 80 = 2000 子步）")
    a = ap.parse_args()

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or (f"{a.part}_{a.strategy}" if a.part == "C" else a.part)
    out_file = out_dir / f"n3_{tag}.json"
    result = {"node": "N3", "tag": tag, "object": a.object,
              "strategy": a.strategy, "trials_requested": a.trials,
              "place_xy": [a.place_x, a.place_y], "parts": {}}

    env, task = build(task_id=a.task_id)
    env.reset()
    sim, m, d, gid = ids(env)
    print(f"[env] task: {task.language}")
    print(f"[env] robot: ECO65 + ZhixingGripper  (策略未加载 —— 脚本专家)")

    bodies = find_object_bodies(sim, a.object)
    if not bodies:
        print(f"[!] 找不到物体 {a.object}")
        return 1
    print(f"[obj] {[n for _, n in bodies]}")

    # ---------- 可行性分析 ----------
    plan = grasp_plan(sim, bodies)
    result["parts"]["plan"] = plan
    print(f"\n[plan] 物体 z 范围 {np.round(plan['object_z_range'],4).tolist()}  "
          f"桌面 z={plan['table_z']:.4f}  开口预算={plan['opening_budget_mm']:.1f} mm")
    if plan["best"]:
        z, w, cl, cx, cy = plan["best"]
        print(f"[plan] ★ 可行抓取: z={z:.4f}  该高度截面宽={w*1000:.1f} mm  "
              f"指尖离桌={cl*1000:.1f} mm  中心=({cx:.4f},{cy:.4f})")
    else:
        print("[plan] ✗ 未找到可行抓取高度（开口不足或指尖会插到桌面以下）")
        wmin = min(r[1] for r in plan["scan"]) if plan["scan"] else None
        if wmin:
            print(f"[plan]   物体最小截面宽 = {wmin:.1f} mm vs 开口预算 "
                  f"{plan['opening_budget_mm']:.1f} mm")
        print("[plan]   剖面（自顶向下，每 ~4 mm 一行）:  z / 截面宽 / 指尖离桌 / 可行")
        rows = plan["scan"]
        for i in range(0, len(rows), 3):
            z, w, cl, ok = rows[i]
            print(f"[plan]     z={z:.4f}  宽={w:6.1f}mm  指尖离桌={cl:+7.1f}mm  "
                  f"{'OK' if ok else '--'}")

    if a.part == "plan":
        env.close()
        out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[out] {out_file}")
        return 0

    # ---------- A. 空载开闭 ×10 ----------
    if a.part in ("all", "A"):
        print("\n===== A. 空载开闭重复性 ×10 =====")
        opens, closes = [], []
        for i in range(10):
            # 每个 trial 先 reset：settle 加大后累计步数会超过 episode horizon，
            # 否则 env.step 抛 "executing action in terminated episode"。
            # A 项只量夹爪净距，reset 不影响被测量。
            env.reset()
            hold(env, -1.0, a.a_settle)
            opens.append(jaw_face_separation_mm(env))
            hold(env, +1.0, a.a_settle)
            closes.append(jaw_face_separation_mm(env))
        A = {
            "open_mm": {"mean": float(np.mean(opens)), "std": float(np.std(opens)),
                        "min": float(np.min(opens)), "max": float(np.max(opens))},
            "closed_mm": {"mean": float(np.mean(closes)), "std": float(np.std(closes)),
                          "min": float(np.min(closes)), "max": float(np.max(closes))},
            "n": 10,
        }
        A["repeatable"] = bool(A["open_mm"]["std"] < 0.5 and A["closed_mm"]["std"] < 0.5)
        A["pass"] = A["repeatable"]
        result["parts"]["A"] = A
        print(f"  全开 净距 = {A['open_mm']['mean']:.2f} ± {A['open_mm']['std']:.3f} mm "
              f"(min {A['open_mm']['min']:.2f} / max {A['open_mm']['max']:.2f})")
        print(f"  闭合 净距 = {A['closed_mm']['mean']:.2f} ± {A['closed_mm']['std']:.3f} mm "
              f"(min {A['closed_mm']['min']:.2f} / max {A['closed_mm']['max']:.2f})")
        print(f"  → 重复性 {'PASS' if A['pass'] else 'FAIL'}")

    # ---------- B. 6 向小幅位姿跟踪 ----------
    if a.part in ("all", "B"):
        print("\n===== B. 6 向小幅位移跟踪（2 cm / 10°）=====")
        import mujoco
        arm = [i for i in range(m.njnt)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("robot0_joint")]
        aadr = [int(m.jnt_qposadr[j]) for j in arm]
        q0 = np.array([d.qpos[x] for x in aadr], dtype=float)
        R0 = gripsite(sim, m, d, gid)[1].copy()
        rows = []
        for label, dv in (("+x", [0.02, 0, 0]), ("-x", [-0.02, 0, 0]),
                          ("+y", [0, 0.02, 0]), ("-y", [0, -0.02, 0]),
                          ("+z", [0, 0, 0.02]), ("-z", [0, 0, -0.02])):
            for x, v in zip(aadr, q0):
                d.qpos[x] = v
            d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            p0 = gripsite(sim, m, d, gid)[0].copy()
            st = drive_to(env, p0 + np.array(dv), -1.0, max_steps=80, tol=0.002)
            p1 = gripsite(sim, m, d, gid)[0]
            rows.append({"dir": label, "commanded_mm": 20.0,
                         "achieved_mm": float(np.linalg.norm(p1 - p0) * 1000),
                         "err_mm": float(np.linalg.norm(p1 - (p0 + np.array(dv))) * 1000),
                         "steps": st["steps"], "nan": st["nan"]})
        errs = [r["err_mm"] for r in rows]
        B = {"rows": rows, "max_err_mm": float(max(errs)), "pass": bool(max(errs) < 5.0)}
        result["parts"]["B"] = B
        for r in rows:
            print(f"  {r['dir']:<3s} 命令 {r['commanded_mm']:.0f} mm → 实际 "
                  f"{r['achieved_mm']:6.1f} mm  残差 {r['err_mm']:5.2f} mm  ({r['steps']} 步)")
        print(f"  → 最大残差 {B['max_err_mm']:.2f} mm  {'PASS' if B['pass'] else 'FAIL'}")

    # ---------- C. 抓取—抬升—保持—放回 ×10 ----------
    if a.part in ("all", "C"):
        print(f"\n===== C. 抓取—抬升 5 cm—保持 2 s—放回 ×10  (策略={a.strategy}) =====")
        jname = free_joint_of(sim, bodies[0][1])
        print(f"  free joint: {jname}")
        scan = plan["scan"]
        z_top, z_tab = plan["object_z_range"][1], plan["table_z"]

        def width_at(z):
            best = None
            for zz, w, cl, okk in scan:
                if best is None or abs(zz - z) < abs(best[0] - z):
                    best = (zz, w)
            return best[1]

        if a.strategy == "side" and plan["best"] is None:
            print("  ✗ 侧向抓取可行性分析未找到窗口 → 该策略几何上不可行，不执行")
            result["parts"]["C"] = {"skipped": True, "strategy": "side",
                                    "reason": "no_feasible_grasp_window"}
        else:
            z_g = z_top - 0.012 if a.strategy == "rim" else plan["best"][0]
            if a.strategy == "rim":
                print(f"  [rim] 抓取高度：每轮按物体**摆放后**的真实碰撞几何重算（顶面下 12 mm）")
            else:
                print(f"  [side] 抓取高度 z={z_g:.4f} 截面宽={plan['best'][1]*1000:.1f} mm")

            trials = []
            for t in range(a.trials):
                env.reset()
                sim, m, d, gid = ids(env)
                # 摆到臂可达区（抓取能力测试允许把物体放到方便处）
                dx = 0.006 * np.cos(2 * np.pi * t / a.trials)
                dy = 0.006 * np.sin(2 * np.pi * t / a.trials)
                set_free_joint(sim, jname, [a.place_x + dx, a.place_y + dy, z_tab])
                for _ in range(60):
                    env.step(np.concatenate([np.zeros(6), [-1.0]]))
                sim, m, d, gid = ids(env)

                pts = body_world_pts(sim, bodies[0][0], collision_only=True)
                obj0 = pts.mean(axis=0)
                obj_all = body_world_pts(sim, bodies[0][0])
                # ★ 必须按「摆放后的真实几何」重算抓取高度：
                # 物体换位置后落点高度会变；沿用 plan 阶段（默认位姿）的 z 会抓空。
                z_top_act = float(pts[:, 2].max())
                z_g = z_top_act - 0.012
                if t == 0:
                    print(f"  [check] 摆放后物体碰撞几何 z 范围 "
                          f"[{pts[:,2].min():.4f}, {pts[:,2].max():.4f}]  → 抓取高度 z={z_g:.4f}")
                if a.strategy == "rim":
                    # 在抓取高度取「近侧(+y)碗壁」的横截面，量出壁的内/外表面，
                    # 夹持点放在**壁厚中点** —— 否则两指会一路夹过头，把碗挤飞。
                    cx_, cy_ = obj0[0], obj0[1]
                    sel = pts[(np.abs(pts[:, 0] - cx_) < 0.016)
                              & (np.abs(pts[:, 2] - z_g) < 0.012)
                              & (pts[:, 1] > cy_)]
                    if sel.shape[0] >= 3:
                        r_in = float(sel[:, 1].min() - cy_)
                        r_out = float(sel[:, 1].max() - cy_)
                    else:
                        r_out = float(pts[:, 1].max() - cy_)
                        r_in = r_out - 0.006
                    pin = np.array([cx_, cy_ + 0.5 * (r_in + r_out), z_g])
                    aim = (f"碗壁捏取 内{r_in*1000:.1f}/外{r_out*1000:.1f} "
                           f"壁厚{(r_out-r_in)*1000:.1f}mm")
                else:
                    pin = np.array([obj0[0], obj0[1], z_g])
                    aim = "侧向夹取"

                # ---- 脚本专家：IK 控制，臂关节锁死，只用夹爪执行器动作 ----
                CTRL_OPEN, CTRL_CLOSED = 0.0, -0.91   # ZhiXing actuator ctrlrange [-0.91, 0]
                R_ref = gripsite(sim, m, d, gid)[1].copy()
                _, _, _, _, _, aadr, adof = _arm_idx(env)
                q_seed = np.array([d.qpos[x] for x in aadr], dtype=float)

                nan = False
                res = {}
                q_pre, res["pre"] = ik_goto(env, q_seed, pin + np.array([0, 0, 0.06]),
                                            R_ref, CTRL_OPEN, settle=15)
                q_g, res["grasp"] = ik_goto(env, q_pre, pin, R_ref, CTRL_OPEN, settle=15)
                # 闭合（臂锁在抓取位）
                step_hold(env, q_g, CTRL_CLOSED, n=600)
                jaw_close = jaw_face_separation_mm(env)
                obj_close = body_world_pts(sim, bodies[0][0]).mean(axis=0)
                # 抬升 5 cm
                q_up, res["lift"] = ik_goto(env, q_g, pin + np.array([0, 0, 0.05]),
                                            R_ref, CTRL_CLOSED, settle=15)
                obj_lift = body_world_pts(sim, bodies[0][0]).mean(axis=0)
                # 保持 2 s（40 步 @20 Hz）
                step_hold(env, q_up, CTRL_CLOSED, n=60)
                sim, m, d, gid = ids(env)
                obj_hold = body_world_pts(sim, bodies[0][0]).mean(axis=0)
                dz_lift = float(obj_lift[2] - obj0[2])
                dz_hold = float(obj_hold[2] - obj0[2])
                success = bool(dz_lift > 0.03 and dz_hold > 0.03)
                # 放回 + 张开
                ik_goto(env, q_up, pin, R_ref, CTRL_CLOSED, settle=15)
                step_hold(env, q_g, CTRL_OPEN, n=600)
                sim, m, d, gid = ids(env)
                obj_final = body_world_pts(sim, bodies[0][0]).mean(axis=0)
                qv = float(np.max(np.abs(d.qvel)))
                nan = not (np.all(np.isfinite(obj_hold)) and np.all(np.isfinite(obj_final)))

                trials.append({
                    "trial": t + 1, "strategy": a.strategy, "aim": aim,
                    "object_xy": [float(obj0[0]), float(obj0[1])],
                    "grasp_z": float(z_g), "pin": pin.tolist(),
                    "ik_res_mm": res, "jaw_close_mm": jaw_close,
                    "obj_shift_on_close_mm": float(np.linalg.norm(obj_close - obj0) * 1000),
                    "dz_lift_mm": dz_lift * 1000, "dz_hold_mm": dz_hold * 1000,
                    "success": success, "nan": bool(nan), "max_qvel": qv,
                    "dz_final_mm": float(obj_final[2] - obj0[2]) * 1000,
                })
                print(f"  [{t+1:2d}/{a.trials}] {aim}  抬升 {dz_lift*1000:+7.1f} mm  "
                      f"保持 {dz_hold*1000:+7.1f} mm  {'OK 成功' if success else 'XX 失败'}  "
                      f"max|qvel|={qv:.2f}")

            n_ok = sum(1 for r in trials if r["success"])
            C = {
                "strategy": a.strategy, "trials": trials,
                "n_success": n_ok, "n_total": len(trials),
                "success_rate": n_ok / len(trials) if trials else 0.0,
                "any_nan": any(r["nan"] for r in trials),
                "max_qvel": max((r["max_qvel"] for r in trials), default=None),
                "threshold": 9,
                "pass": bool(n_ok >= 9 and not any(r["nan"] for r in trials)),
            }
            result["parts"]["C"] = C
            print(f"\n  → 抓持成功 {n_ok}/{len(trials)}   NaN={C['any_nan']}   "
                  f"判据 >=9/10 -> {'PASS' if C['pass'] else 'FAIL'}")


    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n[out] {out_file}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
