#!/usr/bin/env python3
"""N1 追加确认：夹爪开合符号（极性）与开口曲线。

背景：audit_gripper.py 实测发现 ZhiXing 的 action 极性可能与代码注释相反。
本脚本用**两条独立路径**交叉验证，避免单一测量口径出错：

  路径 A（几何真值）：直接写 sim.data.ctrl 扫过 actuator 全量程，
                     测「关节角 → 实际开口宽度」的物理曲线。
  路径 B（真实动作路径）：走 env.step(7 维动作) 的正式链路，
                     测 action=±1 最终落到哪个 ctrl、开口多少。

结论口径：+1 应该是「闭合」（对齐 Panda / LIBERO 约定）。
参考：PandaGripper 也跑一遍路径 B 作为极性基准。

用法（容器内）：
    python gripper_polarity.py --out /app/agent-work-space/zhixing_adaptation_20260927/results
"""
import argparse
import datetime
import json
import os
import pathlib
import sys

import numpy as np


def log(m):
    print(m, flush=True)


def get_robot(env):
    return env.env.robots[0]


def live_sim(env, robot):
    s = getattr(robot, "sim", None)
    if s is not None and hasattr(s, "data"):
        return s
    return get_robot(env).sim


def sites_ends(sim, a, b):
    ia = ib = None
    for i in range(sim.model.nsite):
        n = sim.model.site_id2name(i) or ""
        if n.endswith("_" + a) or n == a:
            ia = i
        if n.endswith("_" + b) or n == b:
            ib = i
    if ia is None or ib is None:
        return None
    return float(np.linalg.norm(np.asarray(sim.data.site_xpos[ia], dtype=float)
                                - np.asarray(sim.data.site_xpos[ib], dtype=float)))


def geom_ends(sim, key):
    """按关键字找左右两个 geom，返回其原点间距（退化口径，用于交叉验证）。"""
    ids = []
    for i in range(sim.model.ngeom):
        n = sim.model.geom_id2name(i) or ""
        if key in n:
            ids.append(i)
    if len(ids) < 2:
        return None, [sim.model.geom_id2name(i) for i in ids]
    p = [np.asarray(sim.data.geom_xpos[i], dtype=float) for i in ids]
    d = min(float(np.linalg.norm(p[i] - p[j]))
            for i in range(len(p)) for j in range(i + 1, len(p)))
    return d, [sim.model.geom_id2name(i) for i in ids]


def jaw(sim, kind, robot=None):
    """开口宽度（m）+ 口径说明。

    kind 必须显式是 'zhixing' 或 'panda'；**不做静默回退**——曾经因为传错字符串
    掉进 Panda 分支，把 |q[0]-q[1]| 当成开口，得到 1.82 m 这种不可能的值。
    """
    if kind == "zhixing":
        d = sites_ends(sim, "Left_Support_tip", "Right_Support_tip")
        return d, "site:Left_Support_tip<->Right_Support_tip"
    if kind != "panda":
        raise ValueError("jaw(): kind 必须是 'zhixing' 或 'panda'，收到 %r" % (kind,))
    # Panda：两个平动指关节，开口 = 两关节行程之差
    if robot is not None:
        try:
            q = np.asarray([sim.data.qpos[i] for i in robot._ref_gripper_joint_pos_indexes],
                           dtype=float)
            if q.size >= 2:
                return float(abs(q[0] - q[1])), "prismatic:q1-q2"
        except Exception:  # noqa: BLE001
            pass
    d, names = geom_ends(sim, "fingerpad")
    return d, "geom:fingerpad* min-pair %s" % names


def fmt(x, nd=5):
    return "None" if x is None else ("%." + str(nd) + "f") % x


def gripper_actuator_ids(sim, robot):
    ids = []
    for a in getattr(robot.gripper, "actuators", []):
        try:
            ids.append(int(sim.model.actuator_name2id(a)))
        except Exception:  # noqa: BLE001
            pass
    return ids


def make_env(robot_name, gripper, suite="libero_spatial", task_id=0, seed=7, res=256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(robots=[robot_name], bddl_file_name=bddl,
                             camera_heights=res, camera_widths=res,
                             gripper_types=gripper)
    env.seed(seed)
    return env


def path_a_ctrl_sweep(env, robot, kind, npts=13, settle=40):
    """路径 A：直接写 ctrl，测关节角 → 开口宽度的物理曲线。"""
    sim = live_sim(env, robot)
    acts = gripper_actuator_ids(sim, robot)
    if not acts:
        return {"error": "no actuator found"}
    ctrlrange = np.asarray(sim.model.actuator_ctrlrange[acts[0]], dtype=float)
    lo, hi = float(ctrlrange[0]), float(ctrlrange[1])
    rows = []
    for v in np.linspace(lo, hi, npts):
        try:
            sim.reset()
        except Exception:  # noqa: BLE001
            pass
        for _ in range(settle):
            for k, aid in enumerate(acts):
                # 多指夹爪：第二个 actuator 取反（Panda 风格对称驱动）
                sim.data.ctrl[aid] = v if k == 0 else -v
            sim.step()
        jq = np.asarray([sim.data.qpos[i] for i in robot._ref_gripper_joint_pos_indexes], dtype=float)
        w, how = jaw(sim, kind, robot)
        rows.append({"ctrl": float(v), "jaw_m": w, "jaw_how": how,
                     "gripper_qpos": jq.tolist()})
    return {"actuator_ids": acts, "ctrlrange": [lo, hi], "rows": rows}


def path_b_action(env, robot, kind, cmd, steps=40):
    """路径 B：走正式动作链路，看 action 落到哪个 ctrl、开口多少。"""
    obs = env.reset()
    robot = get_robot(env)
    sim = live_sim(env, robot)
    acts = gripper_actuator_ids(sim, robot)
    for _ in range(steps):
        a = np.zeros(7, dtype=float)
        a[6] = cmd
        obs, _, _, _ = env.step(a.tolist())
    sim = live_sim(env, robot)
    ctrl = [float(sim.data.ctrl[i]) for i in acts]
    jq = np.asarray([sim.data.qpos[i] for i in robot._ref_gripper_joint_pos_indexes], dtype=float)
    w, how = jaw(sim, kind, robot)
    return {"cmd": cmd, "applied_ctrl": ctrl, "jaw_m": w, "jaw_how": how,
            "gripper_qpos": jq.tolist(),
            "current_action": np.asarray(robot.gripper.current_action, dtype=float).tolist()}


def path_b_sweep(env, kind, speed="0.01", maxn=4):
    """真实动作路径上的开合曲线：慢速累积拿到中间开度（不经直接写 ctrl）。"""
    rows = []
    os.environ["ZHIXING_GRIPPER_SPEED"] = speed
    try:
        for cmd in (-1.0, 1.0):
            env.reset()
            r = get_robot(env)
            sim = live_sim(env, r)
            acts = gripper_actuator_ids(sim, r)
            w0, how = jaw(sim, kind, r)
            rows.append({"cmd": cmd, "n_steps": 0, "current_action": 0.0,
                         "ctrl": [float(sim.data.ctrl[i]) for i in acts],
                         "jaw_m": w0, "jaw_how": how})
            for n in range(1, maxn + 1):
                a = np.zeros(7, dtype=float)
                a[6] = cmd
                env.step(a.tolist())
                sim = live_sim(env, r)
                w, _ = jaw(sim, kind, r)
                ca = np.asarray(r.gripper.current_action, dtype=float).ravel()
                rows.append({"cmd": cmd, "n_steps": n,
                             "current_action": float(ca[0]) if ca.size else None,
                             "ctrl": [float(sim.data.ctrl[i]) for i in acts],
                             "jaw_m": w, "jaw_how": how})
    finally:
        os.environ.pop("ZHIXING_GRIPPER_SPEED", None)
    return {"speed": speed, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/app/agent-work-space/zhixing_adaptation_20260927/results")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rep = {"meta": {"generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
                    "seed": args.seed}}

    # ---------- ZhiXing ----------
    log("### ZhiXing (ECO65 + ZhixingGripper) ###")
    env = make_env("ECO65", "ZhixingGripper", seed=args.seed)
    robot = get_robot(env)
    env.reset()
    robot = get_robot(env)

    rep["zhixing_path_a_ctrl_sweep"] = path_a_ctrl_sweep(env, robot, "zhixing")
    log("\n[A] ctrl → 开口（直接驱动 actuator）")
    log("     ctrl      jaw(m)   gripper_qpos[0]")
    for r in rep["zhixing_path_a_ctrl_sweep"]["rows"]:
        log("   %8.4f   %s   %s" % (r["ctrl"], fmt(r["jaw_m"]), fmt(r["gripper_qpos"][0])))

    rep["zhixing_path_b_action"] = {}
    for cmd in (-1.0, 1.0):
        res = path_b_action(env, robot, "zhixing", cmd)
        rep["zhixing_path_b_action"]["cmd_%+d" % int(cmd)] = res
        log("[B] action=%+0.0f → ctrl=%s  qpos[0]=%s  jaw=%s m"
            % (cmd, [round(c, 4) for c in res["applied_ctrl"]],
               fmt(res["gripper_qpos"][0]), fmt(res["jaw_m"])))

    # 真实动作路径上的开合曲线（慢速累积拿中间开度）
    rep["zhixing_path_b_sweep"] = path_b_sweep(env, "zhixing")
    log("\n[B2] 真实动作路径开合曲线 (speed=0.01)")
    log("     cmd   n   current_action   ctrl      jaw(m)")
    for r in rep["zhixing_path_b_sweep"]["rows"]:
        log("   %+5.1f  %2d   %s        %s   %s"
            % (r["cmd"], r["n_steps"], fmt(r["current_action"], 3),
               fmt(r["ctrl"][0], 4), fmt(r["jaw_m"])))

    # ---------- Panda 基准 ----------
    log("\n### PandaGripper (ECO65 + PandaGripper) — 极性基准 ###")
    env2 = make_env("ECO65", "PandaGripper", seed=args.seed)
    robot2 = get_robot(env2)
    env2.reset()
    robot2 = get_robot(env2)
    rep["panda_path_b_action"] = {}
    for cmd in (-1.0, 1.0):
        res = path_b_action(env2, robot2, "panda", cmd)
        rep["panda_path_b_action"]["cmd_%+d" % int(cmd)] = res
        log("[B] action=%+0.0f → ctrl=%s  jaw=%s m"
            % (cmd, [round(c, 4) for c in res["applied_ctrl"]], fmt(res["jaw_m"])))

    # ---------- 判定 ----------
    zx = rep["zhixing_path_b_action"]
    pd = rep["panda_path_b_action"]
    rep["verdict"] = {
        "zhixing_action_minus1_jaw_m": zx["cmd_-1"]["jaw_m"],
        "zhixing_action_plus1_jaw_m": zx["cmd_+1"]["jaw_m"],
        "panda_action_minus1_jaw_m": pd["cmd_-1"]["jaw_m"],
        "panda_action_plus1_jaw_m": pd["cmd_+1"]["jaw_m"],
        "zhixing_plus1_opens": bool(zx["cmd_+1"]["jaw_m"] > zx["cmd_-1"]["jaw_m"]),
        "panda_plus1_closes": bool(pd["cmd_+1"]["jaw_m"] < pd["cmd_-1"]["jaw_m"]),
    }
    rep["verdict"]["zhixing_polarity_inverted_vs_panda"] = bool(
        rep["verdict"]["zhixing_plus1_opens"] and rep["verdict"]["panda_plus1_closes"])
    log("\n=== 判定 ===")
    for k, v in rep["verdict"].items():
        log("  %s: %s" % (k, v))

    p = os.path.join(args.out, "gripper_polarity.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    log("\nWROTE " + p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
