#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4 前置：为**地面挂载**的 `OnTheGroundECO65` 求解初始位姿，并检查地面工作区可达性。

为什么不能沿用 MountedECO65 的 init_qpos
----------------------------------------
`MountedECO65` 的 init_qpos 是 N2 按「夹持点对齐 **桌面** Panda」解出来的——
工作区在 `main_table` (z=0.90)。`libero_object` / `libero_10` 走
`Libero_Floor_Manipulation`：机器人站在地面（`base_xpos_offset["empty"]=(-0.6,0,0)`，
`workspace_offset=floor_offset=(0,0,-0.035)`），物体在**地面**上。
两者几何完全不同，桌面解不能照搬。

做法（与 N2 同一套方法学）
--------------------------
1. 建 `Panda`（经 libero 前缀 → `OnTheGroundPanda`）的**地面**参考环境，
   实测其 reset 时刻的**夹持点**（两指垫内侧面中点）位姿——这是策略预训练
   时见过的分布。
2. 建 `ECO65`（→ `OnTheGroundECO65`）环境，用迭代 IK 把**它的夹持点**开到
   同一位置、同一姿态。
3. 报告残差、关节余量与地面可达性。

为什么用迭代而不是一次解
------------------------
夹持点**不是 site**，没有解析 Jacobian。但 TCP 修复后 `grip_site` 已与
闭合位夹持点重合（偏差 ~0），故可用 `solve_ik_gripsite` 对 `grip_site`
求解，再实测夹持点残差并回代修正目标——通常 2~3 次即收敛。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N                      # noqa: E402
import verify_tcp_fix as V                     # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"
RES = 128
SETTLE = 20


def build(suite, task_id, robot, gripper, seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    s = benchmark.get_benchmark_dict()[suite]()
    task = s.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kw = {"bddl_file_name": bddl, "camera_heights": RES, "camera_widths": RES,
          "robots": [robot]}
    if gripper != "default":
        kw["gripper_types"] = gripper
    env = OffScreenRenderEnv(**kw)
    env.seed(seed)
    return env, task


def pinch_pose(env):
    """当前夹持点世界位置 + grip_site 姿态矩阵。"""
    sim, m, d, gid = N.ids(env)
    p, R = N.gripsite(sim, m, d, gid)
    r = V.pad_face_pts(env)
    if r is None:
        return None, R
    fl, fr, _ = r
    return 0.5 * (fl + fr), R


def arm_q(env):
    _, _, d, _, _, aadr, _ = N._arm_idx(env)
    return np.array([d.qpos[a] for a in aadr], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    import mujoco

    res = {"node": "N4-pre", "check": "floor_init_pose", "suite": a.suite,
           "task_id": a.task_id}

    # ---------- 1. 地面 Panda 参考 ----------
    print("=" * 78)
    print("[ref] 地面 Panda (OnTheGroundPanda) 参考")
    env_p, task_p = build(a.suite, a.task_id, "Panda", "PandaGripper")
    env_p.reset()
    root_p = env_p.env.robots[0]
    print("  机器人类 = %s   挂载 = %s"
          % (type(root_p.robot_model).__name__, root_p.robot_model.default_mount))
    p_ref, R_ref = pinch_pose(env_p)
    print("  task: %s" % task_p.language)
    print("  Panda 夹持点 = %s" % np.round(p_ref, 5).tolist())
    print("  Panda base   = %s" % np.round(
        env_p.env.sim.data.body_xpos[env_p.env.sim.model.body_name2id("robot0_base")], 5).tolist())
    res["panda_reference"] = {"pinch": [float(v) for v in p_ref],
                              "R": [[float(x) for x in row] for row in R_ref]}
    env_p.close()

    # ---------- 2. ECO65 地面环境 ----------
    print("\n" + "=" * 78)
    print("[tgt] 地面 ECO65 (OnTheGroundECO65)")
    env, task = build(a.suite, a.task_id, "ECO65", "ZhixingGripper")
    env.reset()
    root = env.env.robots[0]
    print("  机器人类 = %s   挂载 = %s"
          % (type(root.robot_model).__name__, root.robot_model.default_mount))
    res["robot_class"] = type(root.robot_model).__name__
    res["default_mount"] = str(root.robot_model.default_mount)

    bid = env.env.sim.model.body_name2id("robot0_base")
    print("  ECO65 base   = %s" % np.round(
        env.env.sim.data.body_xpos[bid], 5).tolist())

    p_cur, _ = pinch_pose(env)
    res["init_pinch_before"] = [float(v) for v in p_cur]
    d0 = float(np.linalg.norm(p_cur - p_ref) * 1000)
    print("  ECO65 夹持点(当前 init) = %s" % np.round(p_cur, 5).tolist())
    print("  → 与 Panda 夹持点偏差 = %.2f mm" % d0)
    res["residual_before_mm"] = d0

    # ---------- 3. 迭代 IK ----------
    # 实测：N2 的对齐解在**地面** arena 下同样成立（残差 <0.1 mm）——因为
    # MountedPanda 与 OnTheGroundPanda 的 init_qpos 相同，且两 arena 里
    # 机器人 base 都在 (-0.6,0,0)，故「夹持点相对 base 的位置」与 arena 无关。
    # 因此这里只在残差超阈值时才真正求解；否则直接采用现值。
    print("\n[ik ] 检查是否需要重解（阈值 0.1 mm）")
    target = p_ref.copy()
    q_seed = arm_q(env)
    hist = []
    if d0 < 0.1:
        print("  当前 init 已满足（残差 %.4f mm）→ 跳过 IK，沿用现值" % d0)
        hist.append({"iter": 0, "err_mm": d0,
                     "q": [round(float(v), 6) for v in q_seed]})
    else:
        for it in range(1, a.iters + 1):
            q, _lo, _hi = N.solve_ik_gripsite(env, target, R_ref, q_seed, iters=1500)
            N.step_hold(env, q, -0.91, n=SETTLE)          # 闭合位（TCP 即夹持点）
            p_now, _ = pinch_pose(env)
            err = p_ref - p_now
            dist = float(np.linalg.norm(err) * 1000)
            hist.append({"iter": it, "err_mm": dist,
                         "q": [round(float(v), 6) for v in q]})
            print("  iter %d: 残差 = %7.4f mm" % (it, dist))
            if dist < 0.1:
                break
            target = target + err                            # 回代修正目标
            q_seed = q
    q_best = np.array(hist[-1]["q"], dtype=float)
    res["ik_history"] = hist
    res["q_candidate"] = [float(v) for v in q_best]
    res["residual_after_mm"] = hist[-1]["err_mm"]

    # 关节余量
    sim, m, d, gid, aj, aadr, adof = N._arm_idx(env)
    for x, v in zip(aadr, q_best):
        d.qpos[x] = float(v)
    mujoco.mj_forward(m, d)
    jids = [env.env.sim.model.joint_name2id(
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) for j in aj]
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    margin = np.minimum(q_best - lo, hi - q_best)
    res["joint_margins_rad"] = [round(float(v), 4) for v in margin]
    res["min_joint_margin_rad"] = round(float(margin.min()), 4)
    print("\n  候选 init_qpos = %s" % [round(float(v), 6) for v in q_best])
    print("  最小关节余量 = %.4f rad" % margin.min())

    # ---------- 4. 地面可达性：能否把夹持点开到物体抓取位 ----------
    print("\n" + "=" * 78)
    print("[reach] 地面可达性检查（按物体真实抓取位试解 IK）")
    objs = []
    for b in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not nm or any(s in nm for s in ("robot0", "gripper0", "mount",
                                           "floor", "wall", "world")):
            continue
        has_free = any(int(m.jnt_bodyid[j]) == b
                       and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
                       for j in range(m.njnt))
        if not has_free:
            continue
        pts = N.body_world_pts(sim, b, collision_only=True)
        if pts is None or pts.shape[0] == 0:
            continue
        lo_o, hi_o = pts.min(axis=0), pts.max(axis=0)
        size = (hi_o - lo_o) * 1000
        z_g = float(hi_o[2]) - 0.012
        pin = np.array([0.5 * (lo_o[0] + hi_o[0]), 0.5 * (lo_o[1] + hi_o[1]), z_g])
        q_s, r_mm = N.ik_goto(env, q_best, pin, R_ref, -0.91, settle=8)
        row = {"body": nm, "size_mm": [round(float(v), 1) for v in size],
               "pinch_target": [round(float(v), 4) for v in pin],
               "ik_res_mm": round(float(r_mm), 2)}
        objs.append(row)
        print("  %-34s size=%-24s IK残差=%7.2f mm  %s"
              % (nm[:34], row["size_mm"], r_mm,
                 "可达" if r_mm < 5.0 else "**不可达/残差大**"))
    res["reachability"] = objs
    env.close()

    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.out) / "n4_floor_init_pose.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    ok = hist[-1]["err_mm"] < 1.0
    print("\n" + "=" * 78)
    print("[verdict] 初始位姿对齐 %s（残差 %.4f mm，最小关节余量 %.4f rad）"
          % ("PASS" if ok else "FAIL", hist[-1]["err_mm"], margin.min()))
    print("[out] %s" % p)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
