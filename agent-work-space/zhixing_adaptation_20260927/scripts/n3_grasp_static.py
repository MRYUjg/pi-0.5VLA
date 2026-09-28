#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3 决定性测试：抓**不需要搬动**的物体（默认位置就在 ECO65 可达区内）。

动机
----
`probe_placement.py` 证明本环境里用自由关节重摆物体会掉高 20~972 mm，
导致专家按"摆放后真实几何"算出的抓取高度落在碰撞体之外（抓空）。
要干净地测夹爪，必须选一个**默认位姿就在可达区内**的物体，完全不动它。

LIBERO Spatial task0 的可达物体：**ramekin**（(-0.2034, 0.187)，89.1×89.3×43.3 mm）。

策略：壁捏（pinch the wall）——在物体近侧(+y)壁厚中点闭合，抬升 5 cm、保持 2 s、放回。
10 次试验通过**微调抓取高度/深度**构造不同初始条件（不搬动物体）。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"


def main():
    import mujoco

    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="glazed_rim_porcelain_ramekin_1")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--drop", type=float, default=0.012,
                    help="抓取高度 = 顶面 - drop")
    ap.add_argument("--tag", default="C_static_rim")
    a = ap.parse_args()

    env, task = N.build(task_id=0)
    env.reset()
    sim, m, d, gid = N.ids(env)
    bodies = N.find_object_bodies(sim, a.object)
    if not bodies:
        print(f"找不到物体 {a.object}")
        return 1
    bid = bodies[0][0]
    print(f"[env] task: {task.language}")
    print(f"[env] robot: ECO65 + ZhixingGripper   (策略未加载 —— 脚本专家)")
    print(f"[obj] {[n for _, n in bodies]}  —— **不搬动**，用默认位姿")

    pts = N.body_world_pts(sim, bid, collision_only=True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
    print(f"[obj] 碰撞几何 x=[{lo[0]:.4f},{hi[0]:.4f}] y=[{lo[1]:.4f},{hi[1]:.4f}] "
          f"z=[{lo[2]:.4f},{hi[2]:.4f}]  轴心≈({cx:.4f},{cy:.4f})")

    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    _, _, _, _, _, aadr, adof = N._arm_idx(env)
    q_home = np.array([d.qpos[x] for x in aadr], dtype=float)

    trials, n_ok = [], 0
    for t in range(a.trials):
        # ★ 每轮 env.reset()：让物体回到**环境自身的默认位姿**。
        # 不这样做的话，上一轮抓取会把物体碰翻/推到桌下（实测第 2 轮起物体掉到 z≈0.03），
        # 后续轮次的几何全错。注意：reset 恢复的是环境默认位姿（稳定），
        # 与「手工搬运物体」不同——后者在本环境中不可靠（见 probe_placement.py）。
        env.reset()
        sim, m, d, gid = N.ids(env)
        bid = N.find_object_bodies(sim, a.object)[0][0]
        R_ref = N.gripsite(sim, m, d, gid)[1].copy()
        for _ in range(60):
            env.step(np.concatenate([np.zeros(6), [-1.0]]))
        sim, m, d, gid = N.ids(env)
        pts = N.body_world_pts(sim, bid, collision_only=True)
        obj0 = pts.mean(axis=0)
        z_top = float(pts[:, 2].max())
        # 10 次试验：抓取高度在 ±4 mm 内变化，构造不同初始条件
        dz = 0.004 * np.sin(2 * np.pi * t / a.trials)
        z_g = z_top - a.drop + dz

        # 近侧(+y)壁：稳健做法——只在 z 高度的水平层里取外表面，
        # 再按标称壁厚往内推，夹持点放在「外表面内侧 3 mm」。
        # （不要用 x 窗口 + y>axis 的复合筛选：稀疏碰撞顶点会筛空，
        #   一旦筛空退化成整体极值就会得到负半径、抓空。）
        for wz in (0.015, 0.025, 0.040):
            selz = pts[np.abs(pts[:, 2] - z_g) < wz]
            if selz.shape[0] >= 3:
                break
        r_out = float(selz[:, 1].max() - cy)
        r_in = max(r_out - 0.006, 0.002)
        pin = np.array([cx, cy + r_out - 0.003, z_g])
        aim = f"壁捏 内{r_in*1000:.1f}/外{r_out*1000:.1f} 厚{(r_out-r_in)*1000:.1f}mm"

        _, _, _, _, _, aadr, adof = N._arm_idx(env)
        q0 = np.array([d.qpos[x] for x in aadr], dtype=float)
        q_pre, res_pre = N.ik_goto(env, q0, pin + np.array([0, 0, 0.06]), R_ref, 0.0, settle=15)
        q_g, res_g = N.ik_goto(env, q_pre, pin, R_ref, 0.0, settle=15)
        N.step_hold(env, q_g, -0.91, n=600)
        jaw_close = N.jaw_face_separation_mm(env)
        obj_close = N.body_world_pts(sim, bid).mean(axis=0)
        q_up, res_up = N.ik_goto(env, q_g, pin + np.array([0, 0, 0.05]), R_ref, -0.91, settle=15)
        obj_lift = N.body_world_pts(sim, bid).mean(axis=0)
        N.step_hold(env, q_up, -0.91, n=40)
        sim, m, d, gid = N.ids(env)
        obj_hold = N.body_world_pts(sim, bid).mean(axis=0)
        dz_lift = float(obj_lift[2] - obj0[2])
        dz_hold = float(obj_hold[2] - obj0[2])
        success = bool(dz_lift > 0.03 and dz_hold > 0.03)
        n_ok += int(success)
        N.ik_goto(env, q_up, pin, R_ref, -0.91, settle=15)
        N.step_hold(env, q_g, 0.0, n=600)

        trials.append({
            "trial": t + 1, "aim": aim, "z_g": float(z_g), "pin": pin.tolist(),
            "wall_in_mm": r_in * 1000, "wall_out_mm": r_out * 1000,
            "ik_res_mm": {"pre": res_pre, "grasp": res_g, "lift": res_up},
            "jaw_close_mm": jaw_close,
            "obj_shift_on_close_mm": float(np.linalg.norm(obj_close - obj0) * 1000),
            "dz_lift_mm": dz_lift * 1000, "dz_hold_mm": dz_hold * 1000,
            "success": success,
        })
        print(f"  [{t+1:2d}/{a.trials}] {aim}  z_g={z_g:.4f}  IK={res_g:.2f}mm  "
              f"闭合净距={jaw_close:6.2f}mm  位移={trials[-1]['obj_shift_on_close_mm']:5.2f}mm  "
              f"抬升={dz_lift*1000:+7.1f}mm  {'OK 成功' if success else 'XX 失败'}")

    out = {"node": "N3", "tag": a.tag, "object": a.object, "relocated": False,
           "strategy": "wall_pinch", "n_success": n_ok, "n_total": a.trials,
           "success_rate": n_ok / a.trials, "threshold": 9,
           "pass": bool(n_ok >= 9), "trials": trials}
    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT) / f"n3_{a.tag}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  → 成功 {n_ok}/{a.trials}   判据 >=9/10 -> {'PASS' if out['pass'] else 'FAIL'}")
    print(f"[out] {OUT}/n3_{a.tag}.json")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
