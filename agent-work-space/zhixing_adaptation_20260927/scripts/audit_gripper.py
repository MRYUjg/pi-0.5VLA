#!/usr/bin/env python3
"""N1 — ZhiXing 夹爪本体体检（audit_gripper.py）。

回答任务方案 P0 第 3 项要求的全部问题，全部以**实测**为准：
  1. 6 关节实际初始化与 XML 5 条 joint equality 的闭环残差
  2. init_qpos 单值广播是否自洽；与「自洽 6 元素」对照（**进程内开关，不改文件**）
  3. 开合符号：-1 是否开、+1 是否合（用指尖 site 间距直接量测）
  4. 全开/全闭的实际开口范围
  5. 按秒标定的开合时间；speed 0.2 vs 0.01 对照
  6. 四连杆 connect 闭环残差、闭合穿透、数值发散

用法（容器内）：
    python audit_gripper.py --robot ECO65 --gripper ZhixingGripper \
        --out /app/agent-work-space/zhixing_adaptation_20260927/results

产物：gripper_audit.json + gripper_audit_steps.csv + 控制台摘要

注意：本环境下 `MjSim.free()` 会 `del self.data`，而 `env.reset()` 会重建
robot/sim。因此**每次 reset 后必须重新获取 robot**，否则拿到已释放的旧对象。
"""
import argparse
import csv
import datetime
import json
import os
import pathlib
import sys
import traceback

import numpy as np

# XML 里声明的 5 条 joint equality：joint1 = c0 + c1*joint2
# （抄录自 robosuite/models/assets/grippers/zhixing_gripper.xml）
JOINT_EQS = [
    ("eq_Left_2", "Left_2_Joint", "Left_1_Joint", 0.0, 1.0),
    ("eq_Right_1", "Right_1_Joint", "Left_1_Joint", 0.0, -1.0),
    ("eq_Right_2", "Right_2_Joint", "Left_1_Joint", 0.0, -1.0),
    ("eq_Left_Support", "Left_Support_Joint", "Left_1_Joint", 0.0, -1.0),
    ("eq_Right_Support", "Right_Support_Joint", "Left_1_Joint", 0.0, -1.0),
]
# 2 条 connect（四连杆闭合）：site1 与 site2 应重合
CONNECT_EQS = [
    ("conn_left", "Left_Support_tip", "Left_2_connect"),
    ("conn_right", "Right_Support_tip", "Right_2_connect"),
]
# 关节声明顺序 = gripper_qpos 索引顺序（XML 中出现顺序）
JOINT_ORDER = [
    "Left_1_Joint", "Left_Support_Joint", "Left_2_Joint",
    "Right_2_Joint", "Right_1_Joint", "Right_Support_Joint",
]


def log(msg):
    print(msg, flush=True)


def get_robot(env):
    """main.py 用 env.env.robots[0]，这里兼容两种包装。"""
    for path in ("env.robots", "robots"):
        obj = env
        ok = True
        for attr in path.split("."):
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok and len(obj) > 0:
            return obj[0]
    raise RuntimeError("找不到 robot 对象")


def live_sim(env, robot):
    """返回仍然有效的 sim（reset 后旧 robot 的 sim 已被 free）。"""
    s = getattr(robot, "sim", None)
    if s is not None and hasattr(s, "data"):
        return s
    r = get_robot(env)
    return r.sim


def reset_env(env):
    """reset 并返回 (obs, 重新获取的 robot)。"""
    obs = env.reset()
    return obs, get_robot(env)


def site_id(sim, name):
    """按后缀匹配 site id，规避 gripper0_ 前缀差异。"""
    m = sim.model
    for i in range(getattr(m, "nsite", 0)):
        n = m.site_id2name(i)
        if n == name or (n and n.endswith("_" + name)):
            return i
    return None


def joint_qpos_index(sim, name):
    m = sim.model
    for i in range(getattr(m, "njnt", 0)):
        n = m.joint_id2name(i)
        if n == name or (n and n.endswith("_" + name)):
            return int(m.jnt_qposadr[i])
    return None


def gripper_qpos_from_obs(obs):
    for k in ("robot0_gripper_qpos",):
        if k in obs:
            return np.asarray(obs[k], dtype=float).ravel()
    return None


def read_gripper_qpos(env, robot, obs=None):
    """优先用观测（与真实入口路径一致），否则回退到 sim。"""
    if obs is not None:
        q = gripper_qpos_from_obs(obs)
        if q is not None:
            return q
    sim = live_sim(env, robot)
    idx = list(robot._ref_gripper_joint_pos_indexes)
    return np.array([sim.data.qpos[i] for i in idx], dtype=float)


def joint_eq_residuals(qpos_by_name):
    out = []
    for name, j1, j2, c0, c1 in JOINT_EQS:
        if j1 in qpos_by_name and j2 in qpos_by_name:
            pred = c0 + c1 * qpos_by_name[j2]
            out.append({"eq": name, "joint1": j1, "joint2": j2,
                        "predicted": float(pred), "actual": float(qpos_by_name[j1]),
                        "residual_rad": float(qpos_by_name[j1] - pred)})
    return out


def connect_residuals(sim, prefix=""):
    out = []
    for name, s1, s2 in CONNECT_EQS:
        i, j = site_id(sim, s1), site_id(sim, s2)
        if i is None or j is None:
            out.append({"eq": name, "error": "site not found"})
            continue
        p1 = np.array(sim.data.site_xpos[i], dtype=float)
        p2 = np.array(sim.data.site_xpos[j], dtype=float)
        out.append({"eq": name, "site1": s1, "site2": s2,
                    "distance_m": float(np.linalg.norm(p1 - p2))})
    return out


def tip_separation(sim):
    """指尖开口：左右 Support_tip 的间距（m）。"""
    i, j = site_id(sim, "Left_Support_tip"), site_id(sim, "Right_Support_tip")
    if i is None or j is None:
        return None
    p1 = np.array(sim.data.site_xpos[i], dtype=float)
    p2 = np.array(sim.data.site_xpos[j], dtype=float)
    return float(np.linalg.norm(p1 - p2))


def fingerpad_penetration(sim):
    """闭合后左右 fingerpad 之间是否有穿透（负间距）。"""
    res = {"max_penetration_m": None, "n_fingerpad_contacts": 0, "raw": []}
    try:
        data = getattr(sim.data, "_data", sim.data)
        for c in data.contact:
            n1 = sim.model.geom_id2name(int(c.geom1)) or ""
            n2 = sim.model.geom_id2name(int(c.geom2)) or ""
            if "fingerpad" in n1 or "fingerpad" in n2:
                dist = float(c.dist)
                res["n_fingerpad_contacts"] += 1
                res["raw"].append({"geom1": n1, "geom2": n2, "dist": dist})
                if res["max_penetration_m"] is None or dist < res["max_penetration_m"]:
                    res["max_penetration_m"] = dist
    except Exception as e:  # noqa: BLE001
        res["error"] = f"{type(e).__name__}: {e}"
    return res


def make_env(robot_name, gripper, suite_name, task_id, seed, resolution=256):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bench = benchmark.get_benchmark_dict()[suite_name]()
    task = bench.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": bddl, "camera_heights": resolution, "camera_widths": resolution}
    if gripper and gripper != "default":
        env_args["gripper_types"] = gripper
    env = OffScreenRenderEnv(robots=[robot_name], **env_args)
    env.seed(seed)
    return env, task.language


def dump_structure(env, robot):
    g = robot.gripper
    sim = live_sim(env, robot)
    s = {
        "gripper_class": type(g).__name__,
        "gripper_dof": int(g.dof),
        "n_gripper_joints": len(robot._ref_gripper_joint_pos_indexes),
        "gripper_joints_attr": list(getattr(g, "joints", [])),
        "declared_init_qpos": np.asarray(g.init_qpos, dtype=float).tolist(),
        "declared_init_qpos_len": int(np.asarray(g.init_qpos).size),
        "current_action_len": int(np.asarray(g.current_action).size),
        "actuators": [],
        "joint_ranges": {},
        "sim_qpos_len": int(len(sim.data.qpos)),
    }
    for a in getattr(g, "actuators", []):
        try:
            aid = sim.model.actuator_name2id(a)
            s["actuators"].append({
                "name": a,
                "ctrlrange": np.asarray(sim.model.actuator_ctrlrange[aid], dtype=float).tolist(),
                "gear": np.asarray(sim.model.actuator_gear[aid], dtype=float).tolist(),
            })
        except Exception as e:  # noqa: BLE001
            s["actuators"].append({"name": a, "error": str(e)})
    for jn in JOINT_ORDER:
        adr = joint_qpos_index(sim, jn)
        rng = None
        try:
            for i in range(getattr(sim.model, "njnt", 0)):
                if (sim.model.joint_id2name(i) or "").endswith(jn):
                    rng = np.asarray(sim.model.jnt_range[i], dtype=float).tolist()
                    break
        except Exception:  # noqa: BLE001
            pass
        s["joint_ranges"][jn] = {"qpos_adr": (None if adr is None else int(adr)), "range": rng}
    return s


def snapshot_state(env, robot, tag, obs=None):
    sim = live_sim(env, robot)
    q = read_gripper_qpos(env, robot, obs)
    by_name = {jn: float(q[i]) for i, jn in enumerate(JOINT_ORDER) if i < len(q)}
    return {
        "tag": tag,
        "gripper_qpos": q.tolist(),
        "qpos_by_name": by_name,
        "joint_eq_residuals": joint_eq_residuals(by_name),
        "connect_residuals": connect_residuals(sim),
        "tip_separation_m": tip_separation(sim),
        "has_nan": bool(np.any(~np.isfinite(q))),
    }


def drive(env, robot, g, steps, csv_rows=None, phase="", speed=None):
    """连续下发 7 维动作（6 维 arm 保持 + 夹爪 g），返回逐步轨迹。"""
    traj = []
    for k in range(steps):
        a = np.zeros(7, dtype=float)
        a[6] = g
        obs, _, _, _ = env.step(a.tolist())
        sim = live_sim(env, robot)
        q = read_gripper_qpos(env, robot, obs)
        sep = tip_separation(sim)
        traj.append({"step": k + 1, "phase": phase, "cmd": g,
                     "tip_separation_m": sep, "qpos": q.tolist()})
        if csv_rows is not None:
            csv_rows.append({"phase": phase, "step": k + 1, "cmd": g,
                             "speed": speed, "tip_separation_m": sep,
                             **{f"q_{n}": (q[i] if i < len(q) else "")
                                for i, n in enumerate(JOINT_ORDER)}})
    return traj


def settle_time(traj, target, tol):
    for r in traj:
        if r["tip_separation_m"] is not None and abs(r["tip_separation_m"] - target) <= tol:
            return r["step"]
    return None


def set_init_qpos_override(gripper_cls, arr):
    """进程内开关：覆盖 init_qpos，不改任何文件。返回 restore 函数。"""
    original = gripper_cls.__dict__.get("init_qpos", None)

    def _get(self):
        return np.array(arr, dtype=float)

    gripper_cls.init_qpos = property(_get)

    def restore():
        if original is None:
            delattr(gripper_cls, "init_qpos")
        else:
            gripper_cls.init_qpos = original
    return restore


def run_timing(env, gripper_cls, speed, steps, csv_rows):
    """给定 speed 下测全开/全闭所需步数。"""
    os.environ["ZHIXING_GRIPPER_SPEED"] = str(speed)
    try:
        _, robot = reset_env(env)
        drive(env, robot, -1.0, steps, csv_rows, f"open_speed{speed}", speed)
        sim = live_sim(env, robot)
        sep_open = tip_separation(sim)
        t_close = drive(env, robot, +1.0, steps, csv_rows, f"close_speed{speed}", speed)
        sep_close = tip_separation(sim)
        tol = max(1e-4, 0.01 * abs((sep_open or 1.0) - (sep_close or 0.0)))
        return {
            "speed": speed,
            "full_open_separation_m": sep_open,
            "full_closed_separation_m": sep_close,
            "steps_to_full_close": settle_time(t_close, sep_close, tol),
            "close_traj_sep": [r["tip_separation_m"] for r in t_close],
        }
    finally:
        os.environ.pop("ZHIXING_GRIPPER_SPEED", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", default="ECO65")
    ap.add_argument("--gripper", default="ZhixingGripper")
    ap.add_argument("--suite", default="libero_spatial")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps-per-phase", type=int, default=60)
    ap.add_argument("--out", default="/app/agent-work-space/zhixing_adaptation_20260927/results")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = {
        "meta": {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "robot": args.robot, "gripper": args.gripper, "suite": args.suite,
            "task_id": args.task_id, "seed": args.seed,
            "steps_per_phase": args.steps_per_phase,
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "?"),
        },
        "checks": {}, "errors": [],
    }
    csv_rows = []

    try:
        env, task_language = make_env(args.robot, args.gripper, args.suite, args.task_id, args.seed)
        robot = get_robot(env)
        report["meta"]["task_language"] = task_language
        report["structure"] = dump_structure(env, robot)
        st = report["structure"]
        log(f"[结构] {st['gripper_class']} dof={st['gripper_dof']} n_joints={st['n_gripper_joints']} "
            f"declared_init_qpos={st['declared_init_qpos']} (len={st['declared_init_qpos_len']})")
        log(f"[结构] 关节索引 {list(robot._ref_gripper_joint_pos_indexes)} | sim qpos len={st['sim_qpos_len']}")
    except Exception as e:  # noqa: BLE001
        report["errors"].append(f"env 构建失败: {e}\n{traceback.format_exc()}")
        _write(report, csv_rows, args.out)
        log("FATAL: env 构建失败\n" + report["errors"][-1])
        return 1

    gripper_cls = type(robot.gripper)
    is_zhixing = "Zhixing" in gripper_cls.__name__

    # ---- A. 声明式 init_qpos（单值广播）下的 reset 残差 ----
    obs, robot_a = reset_env(env)
    robot_identity_changed = robot_a is not robot
    report["reset_declared"] = snapshot_state(env, robot_a, "declared_broadcast", obs)
    report["checks"]["robot_object_recreated_on_reset"] = bool(robot_identity_changed)
    log(f"\n[A] 声明式 init_qpos reset 后 gripper qpos = "
        f"{np.round(report['reset_declared']['gripper_qpos'], 5).tolist()}")
    log(f"[A] reset 是否重建 robot 对象: {robot_identity_changed}")

    # ---- B. 自洽 6 元素 init_qpos 对照（进程内开关）----
    if is_zhixing and report["structure"]["declared_init_qpos_len"] == 1:
        q0 = float(report["structure"]["declared_init_qpos"][0])
        by_name = {"Left_1_Joint": q0}
        for _, j1, j2, c0, c1 in JOINT_EQS:
            by_name[j1] = c0 + c1 * by_name[j2]
        consistent = [by_name[j] for j in JOINT_ORDER]
        report["consistent_init_qpos_derived"] = consistent
        log(f"\n[B] 由 5 条 equality 推出的自洽 init_qpos = {np.round(consistent, 5).tolist()}")
        restore = set_init_qpos_override(gripper_cls, consistent)
        try:
            obs_b, robot_b = reset_env(env)
            report["reset_consistent"] = snapshot_state(env, robot_b, "consistent_6elem", obs_b)
            log(f"[B] 自洽 init reset 后 qpos = "
                f"{np.round(report['reset_consistent']['gripper_qpos'], 5).tolist()}")
        finally:
            restore()

        d, c = report["reset_declared"], report["reset_consistent"]
        report["checks"]["declared_max_joint_eq_residual_rad"] = float(
            max(abs(r["residual_rad"]) for r in d["joint_eq_residuals"]))
        report["checks"]["consistent_max_joint_eq_residual_rad"] = float(
            max(abs(r["residual_rad"]) for r in c["joint_eq_residuals"]))
        report["checks"]["init_qpos_broadcast_is_inconsistent"] = bool(
            report["checks"]["declared_max_joint_eq_residual_rad"] > 1e-3)
    else:
        log("\n[B] 跳过：仅对 ZhiXing 单值 init_qpos 做自洽化对照")

    # ---- C. 开合符号 / 开口范围（用声明的 init_qpos 状态起测）----
    obs_c, robot_c = reset_env(env)
    sim = live_sim(env, robot_c)
    start_sep = tip_separation(sim)
    speed_tag = os.environ.get("ZHIXING_GRIPPER_SPEED", "default(0.2)")
    traj_open = drive(env, robot_c, -1.0, args.steps_per_phase, csv_rows, "open", speed_tag)
    sep_open = tip_separation(live_sim(env, robot_c))
    st_open = snapshot_state(env, robot_c, "full_open")
    traj_close = drive(env, robot_c, +1.0, args.steps_per_phase, csv_rows, "close", speed_tag)
    sep_close = tip_separation(live_sim(env, robot_c))
    st_close = snapshot_state(env, robot_c, "full_closed")
    pen = fingerpad_penetration(live_sim(env, robot_c))

    report["motion"] = {
        "start_separation_m": start_sep,
        "full_open_separation_m": sep_open,
        "full_closed_separation_m": sep_close,
        "opening_range_m": (None if (sep_open is None or sep_close is None)
                            else float(sep_open - sep_close)),
        "state_full_open": st_open,
        "state_full_closed": st_close,
        "fingerpad_penetration": pen,
    }
    report["checks"]["cmd_minus1_opens"] = bool(
        start_sep is not None and sep_open is not None and sep_open >= start_sep - 1e-4)
    report["checks"]["cmd_plus1_closes"] = bool(
        sep_open is not None and sep_close is not None and sep_close < sep_open - 1e-4)
    report["checks"]["sign_convention_ok"] = bool(
        report["checks"]["cmd_minus1_opens"] and report["checks"]["cmd_plus1_closes"])
    log(f"\n[C] 开口: start={start_sep:.5f} 全开={sep_open:.5f} 全闭={sep_close:.5f} "
        f"行程={report['motion']['opening_range_m']:.5f} m")
    log(f"[C] 符号约定 -1=开 / +1=合 : {report['checks']['sign_convention_ok']}")

    # ---- D. 开合时间（按秒标定）+ speed 消融 ----
    ctrl_hz = float(getattr(env.env, "control_freq", 20) or 20)
    report["meta"]["control_freq_hz"] = ctrl_hz
    t_close_steps = settle_time(traj_close, sep_close,
                                max(1e-4, 0.01 * abs((sep_open or 1.0) - (sep_close or 0.0))))
    report["timing"] = {
        "control_freq_hz": ctrl_hz,
        "default_speed": os.environ.get("ZHIXING_GRIPPER_SPEED", "0.2"),
        "steps_to_full_close": t_close_steps,
        "seconds_to_full_close": (None if t_close_steps is None else t_close_steps / ctrl_hz),
    }
    log(f"[D] 默认 speed 下全闭 {t_close_steps} 步 @ {ctrl_hz} Hz "
        f"→ {report['timing']['seconds_to_full_close']} s")

    if is_zhixing:
        abl = []
        for sp in (0.2, 0.01):
            r = run_timing(env, gripper_cls, sp, args.steps_per_phase, csv_rows)
            r["seconds_to_full_close"] = (None if r["steps_to_full_close"] is None
                                          else r["steps_to_full_close"] / ctrl_hz)
            abl.append(r)
            log(f"[D] speed={sp}: 全开={r['full_open_separation_m']:.5f} "
                f"全闭={r['full_closed_separation_m']:.5f} 全闭步数={r['steps_to_full_close']}")
        report["speed_ablation"] = abl

    # ---- E. 数值健康 ----
    report["checks"]["no_nan_inf"] = bool(
        not any(s["has_nan"] for s in (report["reset_declared"], st_open, st_close)))
    report["checks"]["max_connect_residual_m"] = float(
        max([r.get("distance_m", 0.0) for r in st_close["connect_residuals"]] or [0.0]))
    log(f"\n[E] 有 NaN/Inf: {not report['checks']['no_nan_inf']} | "
        f"四连杆闭环最大残差 = {report['checks']['max_connect_residual_m']*1000:.4f} mm | "
        f"闭合时指垫接触数 = {pen.get('n_fingerpad_contacts')}")

    _write(report, csv_rows, args.out)
    return 0


def _write(report, csv_rows, out):
    jp = os.path.join(out, "gripper_audit.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    cp = os.path.join(out, "gripper_audit_steps.csv")
    if csv_rows:
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
    print(f"\nWROTE {jp}")
    if csv_rows:
        print(f"WROTE {cp}")
    print("\n=== CHECKS ===")
    for k, v in report.get("checks", {}).items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
