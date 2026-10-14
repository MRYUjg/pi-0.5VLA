#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""措施 B3 复验：关闭额外关节回拉 PD 的 ECO65 候选配置（复核 §3.2）。

复核 §3.2 要求的做法：
  1. 把额外关节回拉项**配置化**，建立 ECO65 候选配置（关闭），Panda 保持原配置作对照，
     不加大增益、不放宽角度阈值；
  2. 其余设置不变先做平移/旋转复验，同时记录**力矩饱和、关节限位、雅可比条件数、
     碰撞、夹爪约束残差**；
  3. **修正收敛判断时序**：控制步执行后重新读位姿；保持窗口结束后**同时**检查位置与角度，
     不使用上一步误差、不只检查旋转。

本探针在同一进程内切换 profile（`extra_joint_pd_kp_for` 每次调用都读环境变量），
因此 legacy 与 candidate 跑的是**同一环境、同一 seed、同一动作序列**，只有该项不同。

用法：
  docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
    /bin/bash -c "source /.venv/bin/activate && export ZHIXING_GRIPPER_SPEED=0.01 && python \
      /app/agent-work-space/zhixing_adaptation_20260927/scripts/verify_b3_extra_pd.py"
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

OUT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927/results")
OUT.mkdir(parents=True, exist_ok=True)

PROFILES = ["legacy", "eco65_no_extra_pd"]
POS_CMD_M = 0.02
ROT_CMD_RAD = 0.15
SETTLE_STEPS = 90          # 伺服上限
HOLD_STEPS = 10            # 保持窗口（§3.2.3：误差在窗口**结束后**读）
TOL_POS_M = 0.005
TOL_ROT_RAD = 0.03         # 收敛容差，故意严于通过闸门
GATE_ROT_RAD = 0.05        # 通过闸门（复核既有阈值，不放宽）
GATE_POS_M = 0.005


def build():
    """ECO65 + ZhiXing，libero_object task 0（与复核探针同一环境/seed）。"""
    import n4_collect as C

    env, task = C.build("libero_object", 0, seed=1)
    return env, task


def constraint_residuals(m, d):
    """按编译模型里的等式约束逐条算残差：joint 等式比 qpos，connect 比 site 距离。"""
    import mujoco

    out = {"joint_eq_max": 0.0, "connect_max_m": 0.0, "n_joint_eq": 0, "n_connect": 0}
    for e in range(m.neq):
        t = int(m.eq_type[e])
        if t == int(mujoco.mjtEq.mjEQ_JOINT):
            a1, a2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            q1 = float(d.qpos[int(m.jnt_qposadr[a1])])
            q2 = float(d.qpos[int(m.jnt_qposadr[a2])])
            # MuJoCo 关节等式： q1 = polycoef(q2)，系数在 eq_data[0:5]
            c = [float(v) for v in m.eq_data[e][0:5]]
            rhs = c[0] + c[1] * q2 + c[2] * q2 ** 2 + c[3] * q2 ** 3 + c[4] * q2 ** 4
            out["joint_eq_max"] = max(out["joint_eq_max"], abs(q1 - rhs))
            out["n_joint_eq"] += 1
        elif t == int(mujoco.mjtEq.mjEQ_CONNECT):
            s1, s2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            out["connect_max_m"] = max(out["connect_max_m"],
                                       float(np.linalg.norm(d.site_xpos[s1] - d.site_xpos[s2])))
            out["n_connect"] += 1
    return out


def contact_breakdown(m, d):
    """[§3.2.2] 碰撞分解：机器人自碰 / 机器人与环境 / 其它。"""
    import mujoco

    def is_robot(g):
        b = int(m.geom_bodyid[g])
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        return nm.startswith("robot0_")

    self_c = env_c = 0
    for i in range(int(d.ncon)):
        g1, g2 = int(d.contact[i].geom1), int(d.contact[i].geom2)
        r1, r2 = is_robot(g1), is_robot(g2)
        if r1 and r2:
            self_c += 1
        elif r1 or r2:
            env_c += 1
    return {"robot_self_contacts": self_c, "robot_env_contacts": env_c,
            "ncon_total": int(d.ncon)}


def arm_jacobian_cond(m, d, site_id, arm_qpos_idx):
    """抓取点 6×6 雅可比的条件数（只取臂关节列）。"""
    import mujoco

    jp = np.zeros((3, m.nv))
    jr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jp, jr, site_id)
    # 臂关节对应的 dof 列
    dofs = [int(m.jnt_dofadr[j]) for j in range(m.njnt)
            if int(m.jnt_qposadr[j]) in arm_qpos_idx]
    J = np.vstack([jp[:, dofs], jr[:, dofs]])
    try:
        return float(np.linalg.cond(J))
    except Exception:                                     # noqa: BLE001
        return None


def probe_action(env, kind, axis, sign, profile):
    """执行一次跟踪探针；误差在保持窗口**结束后**读取（§3.2.3）。

    每个探针前 `seed(1)+reset()`：保证 legacy 与 candidate 从**同一初态**出发，
    同时避免累计步数撞上 episode horizon（1000）。
    """
    import mujoco

    os.environ["EXTRA_JOINT_PD_PROFILE"] = profile
    env.seed(1)
    env.reset()
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    p0, R0 = N.gripsite(sim, m, d, gid)
    arm_qpos_idx = set(int(i) for i in r._ref_joint_pos_indexes)

    if kind == "translation":
        target_p = p0 + sign * POS_CMD_M * np.eye(3)[axis]
        target_R = R0.copy()
    else:
        # 绕世界轴旋转 ROT_CMD_RAD
        ax = np.eye(3)[axis]
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        target_R = (np.eye(3) + np.sin(sign * ROT_CMD_RAD) * K
                    + (1 - np.cos(sign * ROT_CMD_RAD)) * (K @ K)) @ R0
        target_p = p0.copy()

    sat = 0
    lim_hits = 0
    ncon_max = 0
    conds = []
    for _ in range(SETTLE_STEPS):
        sim, m, d, gid = N.ids(env)
        p, R = N.gripsite(sim, m, d, gid)
        ap = np.clip((target_p - p) / 0.05, -1.0, 1.0)
        er = N.rot_err_vec(target_R, R)
        # 旋转限制落在**指令增量**上（0.5 rad 满量程），避免 0.4 rad/step 的饱和指令
        ar = np.clip(er / 0.5, -1.0, 1.0)
        env.step(np.concatenate([ap, ar, [-1.0]]))
        # 力矩饱和与限位
        tl = r.torque_limits
        tq = np.asarray(r.torques, float)
        if np.any(np.isclose(tq, tl[0], atol=1e-6)) or np.any(np.isclose(tq, tl[1], atol=1e-6)):
            sat += 1
        q = d.qpos[list(arm_qpos_idx)]
        rng = np.array([m.jnt_range[j] for j in range(m.njnt)
                        if int(m.jnt_qposadr[j]) in arm_qpos_idx])
        if np.any((q <= rng[:, 0] + 1e-4) | (q >= rng[:, 1] - 1e-4)):
            lim_hits += 1
        ncon_max = max(ncon_max, int(d.ncon))
        if _ % 10 == 0:
            c = arm_jacobian_cond(m, d, gid, arm_qpos_idx)
            if c is not None:
                conds.append(c)
        if (float(np.linalg.norm(target_p - p)) < 1e-4
                and float(np.linalg.norm(N.rot_err_vec(target_R, R))) < 1e-4):
            break

    # 保持窗口：零指令 10 步，**窗口结束后**再读误差
    for _ in range(HOLD_STEPS):
        env.step(np.concatenate([np.zeros(6), [-1.0]]))
    sim, m, d, gid = N.ids(env)
    p, R = N.gripsite(sim, m, d, gid)
    pos_err = float(np.linalg.norm(target_p - p))
    rot_err = float(np.linalg.norm(N.rot_err_vec(target_R, R)))
    res = constraint_residuals(m, d)
    con = contact_breakdown(m, d)

    return {
        "profile": profile,
        "kind": kind, "axis": int(axis), "sign": int(sign),
        "commanded": POS_CMD_M if kind == "translation" else ROT_CMD_RAD,
        "pos_err_m": pos_err, "rot_err_rad": rot_err,
        "pos_err_mm": pos_err * 1000.0, "rot_err_deg": float(np.degrees(rot_err)),
        "converged": bool(pos_err <= TOL_POS_M and rot_err <= TOL_ROT_RAD),
        "pass": bool(pos_err <= GATE_POS_M and rot_err <= GATE_ROT_RAD),
        "n_servo_steps": SETTLE_STEPS,
        "torque_saturated_steps": sat, "joint_limit_steps": lim_hits,
        "ncon_max": con["ncon_total"],
        "robot_self_contacts": con["robot_self_contacts"],
        "robot_env_contacts": con["robot_env_contacts"],
        "jacobian_cond_median": float(np.median(conds)) if conds else None,
        "jacobian_cond_max": float(np.max(conds)) if conds else None,
        "gripper_joint_eq_max_rad": res["joint_eq_max"],
        "gripper_connect_max_m": res["connect_max_m"],
        "n_gripper_joint_eq": res["n_joint_eq"],
        "kp_extra": float(os.environ.get("EXTRA_JOINT_PD_KP", "")) if os.environ.get(
            "EXTRA_JOINT_PD_KP", "").strip() else None,
    }


def main():
    from robosuite.models.robots.robot_model import (
        extra_joint_pd_kp_for, EXTRA_JOINT_PD_PROFILES)

    out = {"node": "B3", "profiles": PROFILES, "results": [], "checks": {}}

    # 0) kp 解析核对
    resolve = {}
    for prof in PROFILES + ["legacy"]:
        os.environ["EXTRA_JOINT_PD_PROFILE"] = prof
        resolve[prof] = {cls: extra_joint_pd_kp_for(cls) for cls in
                         ("ECO65", "MountedECO65", "OnTheGroundECO65", "Panda")}
    out["kp_resolution"] = resolve
    env, task = build()
    print("[env] %s | %s + %s" % (task.language, type(env.env.robots[0].robot_model).__name__,
                                  type(env.env.robots[0].gripper).__name__))

    actions = ([("translation", a, s) for a in range(3) for s in (1, -1)]
               + [("rotation", a, s) for a in range(3) for s in (1, -1)])
    for prof in PROFILES:
        for kind, axis, sign in actions:
            r = probe_action(env, kind, axis, sign, prof)
            out["results"].append(r)
            print("  %-18s %-11s axis=%d s=%+d  位置 %6.3f mm  姿态 %6.3f°  conv=%-5s pass=%-5s"
                  % (prof, kind, axis, sign, r["pos_err_mm"], r["rot_err_deg"],
                     r["converged"], r["pass"]))
    env.close()

    # 1) 判据
    def rows(prof, kind=None):
        return [r for r in out["results"] if r["profile"] == prof
                and (kind is None or r["kind"] == kind)]

    for prof in PROFILES:
        p = "%s_" % prof
        out["checks"][p + "all_probes_pass"] = bool(all(r["pass"] for r in rows(prof)))
        out["checks"][p + "translation_all_converged"] = bool(
            all(r["converged"] for r in rows(prof, "translation")))
        out["checks"][p + "rotation_all_converged"] = bool(
            all(r["converged"] for r in rows(prof, "rotation")))
        out["checks"][p + "no_joint_limit_violation"] = bool(
            all(r["joint_limit_steps"] == 0 for r in rows(prof)))
        # [§3.2.2] 约束残差阈值：joint 等式 < 0.02 rad(~1.1°)、connect < 1 mm。
        # 参考量级：本环境静态下 joint eq 约 5.9e-3 rad、connect 约 4e-6 m，
        # 都是软约束在位置执行器下的正常稳态残差，不是缺陷。
        out["checks"][p + "gripper_constraints_ok"] = bool(
            all(r["gripper_joint_eq_max_rad"] < 0.02 and r["gripper_connect_max_m"] < 1e-3
                for r in rows(prof)))
        out["checks"][p + "no_robot_self_collision"] = bool(
            all(r["robot_self_contacts"] == 0 for r in rows(prof)))

    # 候选不得让夹爪约束或自碰变差（这才是本项真正要证明的）
    def mean_of(prof, key):
        return float(np.mean([r[key] for r in rows(prof)]))

    out["constraint_comparison"] = {
        "legacy_joint_eq_mean": mean_of("legacy", "gripper_joint_eq_max_rad"),
        "candidate_joint_eq_mean": mean_of("eco65_no_extra_pd", "gripper_joint_eq_max_rad"),
        "legacy_self_contacts_max": int(max(r["robot_self_contacts"] for r in rows("legacy"))),
        "candidate_self_contacts_max": int(
            max(r["robot_self_contacts"] for r in rows("eco65_no_extra_pd"))),
        "legacy_self_contacts": [r["robot_self_contacts"] for r in rows("legacy")],
    }
    out["checks"]["candidate_not_worse_on_gripper_constraints"] = bool(
        out["constraint_comparison"]["candidate_joint_eq_mean"]
        <= out["constraint_comparison"]["legacy_joint_eq_mean"] * 1.1)
    out["checks"]["candidate_not_worse_on_self_collision"] = bool(
        out["constraint_comparison"]["candidate_self_contacts_max"]
        <= out["constraint_comparison"]["legacy_self_contacts_max"])

    rot_leg = float(np.mean([r["rot_err_deg"] for r in rows("legacy", "rotation")]))
    rot_cand = float(np.mean([r["rot_err_deg"] for r in rows("eco65_no_extra_pd", "rotation")]))
    out["rotation_mean_err_deg"] = {"legacy": rot_leg, "candidate": rot_cand}
    out["checks"]["candidate_improves_rotation"] = bool(rot_cand < rot_leg)
    out["checks"]["legacy_reproduces_known_failure"] = bool(
        any(not r["converged"] for r in rows("legacy", "rotation")))
    # 判据语义：`legacy` 是**对照组**，预期复现已知缺陷（+ry/±z 不收敛），
    # 因此它的 "all_probes_pass" 为 False 是**正确结果**，不作为失败条件。
    # 真正的通过条件 = 候选全部达标 + 对照组确实复现缺陷 + 候选未让约束/自碰变差。
    candidate_keys = [k for k in out["checks"]
                      if k.startswith("eco65_no_extra_pd_") or k.startswith("candidate_")]
    out["control_note"] = ("legacy(kp=3.0) 为对照组，预期不达标；"
                           "其 legacy_* 项仅作复现证据，不参与通过判定")
    candidate_ok = bool(all(out["checks"][k] for k in candidate_keys))
    control_ok = bool(out["checks"]["legacy_reproduces_known_failure"])
    out["candidate_requirements"] = {k: out["checks"][k] for k in candidate_keys}
    out["candidate_requirements"]["legacy_reproduces_known_failure"] = control_ok
    out["verdict"] = "PASS" if (candidate_ok and control_ok) else "FAIL"

    p = OUT / "b3_extra_pd_verify.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 78)
    print("kp 解析: %s" % json.dumps(resolve, ensure_ascii=False))
    print("旋转平均姿态误差: legacy %.3f°  →  candidate %.3f°" % (rot_leg, rot_cand))
    print("-" * 78)
    for k, v in out["checks"].items():
        print("  %-46s %s" % (k, "✅" if v else "❌"))
    print("NODE_VERDICT = %s" % out["verdict"])
    print("[out] %s" % p)
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
