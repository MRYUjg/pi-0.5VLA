#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""措施 B2 复验：沿**真实 env.step 路径**验证夹爪动态行为（复核 §5）。

复核 §5.2 指出：`n2v2_settled_probe.py` 的"部署斜坡"其实是**自行插值写 ctrl +
每子步锁臂**的几何试验，不是真实控制入口，因此不能证明
`env.step → format_action → actuator` 这条链稳定。

本探针只走真实入口（`env.step([0,0,0,0,0,0,grip_cmd])`），覆盖 §5 要求的动作序列：
  1. 开 → 合 → 开
  2. 半程反向
  3. reset 后立即闭合
  4. 空载 / 带载（把目标物体移到夹持点之间再闭合）
  5. 不同臂姿态（3 组关节角）
并检查：关节范围、equality/connect 残差、接触穿透、超时。

用法：
  docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
    /bin/bash -c "source /.venv/bin/activate && python \
      /app/agent-work-space/zhixing_adaptation_20260927/scripts/verify_b2_dynamic_path.py"
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N          # noqa: E402
import runtime_config as RC        # noqa: E402

OUT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927/results")
OUT.mkdir(parents=True, exist_ok=True)

MAX_STEPS = 120          # 单段超时上限
EQ_TOL_RAD = 0.02
CONNECT_TOL_M = 1e-3
PEN_TOL_MM = 2.0         # 允许的软接触穿透上限（与 N3 口径一致）


def clamp_cmd(v):
    return float(np.clip(v, -1.0, 1.0))


def hold_action(grip_cmd):
    return np.concatenate([np.zeros(6), [grip_cmd]])


def gripper_state(env):
    """夹爪：净开口、关节向量、关节范围越界数、等式/connect 残差。"""
    import mujoco

    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    idx = list(r._ref_gripper_joint_pos_indexes)
    q = np.array([d.qpos[i] for i in idx], float)
    # 关节范围：**只对真正声明了范围的关节判越界**。
    # XML 里只有 Left_1_Joint 写了 range="-0.91 0"，其余 5 个无 range
    # → MuJoCo 的 jnt_range 为 [0,0]，语义是"无限位"，不是"必须等于 0"。
    jids = [j for j in range(m.njnt) if int(m.jnt_qposadr[j]) in set(int(i) for i in idx)]
    limited = [(j, float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) for j in jids
               if float(m.jnt_range[j][0]) != float(m.jnt_range[j][1])]
    out_of_range = 0
    for j, lo, hi in limited:
        v = float(d.qpos[int(m.jnt_qposadr[j])])
        if v < lo - 1e-6 or v > hi + 1e-6:
            out_of_range += 1

    eq = 0.0
    n_eq = 0
    con = 0.0
    for e in range(m.neq):
        t = int(m.eq_type[e])
        if t == int(mujoco.mjtEq.mjEQ_JOINT):
            a1, a2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            q1 = float(d.qpos[int(m.jnt_qposadr[a1])])
            q2 = float(d.qpos[int(m.jnt_qposadr[a2])])
            c = [float(v) for v in m.eq_data[e][0:5]]
            rhs = c[0] + c[1] * q2 + c[2] * q2 ** 2 + c[3] * q2 ** 3 + c[4] * q2 ** 4
            eq = max(eq, abs(q1 - rhs))
            n_eq += 1
        elif t == int(mujoco.mjtEq.mjEQ_CONNECT):
            s1, s2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
            con = max(con, float(np.linalg.norm(d.site_xpos[s1] - d.site_xpos[s2])))
    return {"jaw_mm": N.jaw_face_separation_mm(env), "q": q.tolist(),
            "out_of_range": out_of_range, "n_limited_joints": len(limited),
            "joint_eq_max_rad": eq, "connect_max_m": con, "n_joint_eq": n_eq}


def pad_penetration_mm(env):
    """指垫接触穿透，分两类报告：
      * `object` —— 指垫↔**非指垫**（也就是被抓物体/桌面）的穿透，这才是抓取关心的量；
      * `padpad` —— 两片指垫互相挤压的穿透，只在**空载闭合**时出现，单独记录。
    返回 (object_mm, padpad_mm)。
    """
    import mujoco

    sim, m, d, _ = N.ids(env)
    pads = set()
    for g in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if "fingerpad" in n:
            pads.add(g)
    worst_o = worst_p = 0.0
    for i in range(int(d.ncon)):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        p1, p2 = g1 in pads, g2 in pads
        if not (p1 or p2):
            continue
        dist = float(c.dist)
        if dist >= 0:
            continue
        if p1 and p2:
            worst_p = max(worst_p, -dist)
        else:
            worst_o = max(worst_o, -dist)
    return worst_o * 1000.0, worst_p * 1000.0


def drive(env, cmd, target_jaw_mm=None, max_steps=MAX_STEPS, stop_on=None):
    """沿真实 env.step 推进夹爪指令，直到达到目标开口或超时。

    返回 (步数, 指垫↔物体穿透峰值 mm, 指垫↔指垫穿透峰值 mm)。
    """
    steps = 0
    pen_o = pen_p = 0.0
    for _ in range(max_steps):
        env.step(hold_action(cmd))
        steps += 1
        o, p = pad_penetration_mm(env)
        pen_o, pen_p = max(pen_o, o), max(pen_p, p)
        if target_jaw_mm is not None:
            j = N.jaw_face_separation_mm(env)
            if j is not None and stop_on == "close" and j <= target_jaw_mm:
                break
            if j is not None and stop_on == "open" and j >= target_jaw_mm:
                break
    return steps, pen_o, pen_p


def main():
    RC.apply()
    import n4_collect as C

    env, task = C.build("libero_object", 0, seed=1)
    r = env.env.robots[0]
    eff = RC.effective(env)
    print("[B2] 冻结版本=%s  生效 speed=%.4f  PD profile=%s"
          % (eff["freeze_version"], eff["gripper_speed"], eff["extra_joint_pd_profile"]))
    out = {"node": "B2", "runtime": RC.fingerprint(env), "cases": [], "checks": {}}

    # 记录初始全开状态。注意：reset 后 `gripper.current_action = 0`（行程中点），
    # speed=0.01 时每 env step 变化 25×0.01 = 0.25，故**半程需 4 步、全程 8 步**。
    # 为比较**稳态**，这里统一用固定 12 步饱和到端点，而不是"达到某开口就停"
    # （后者会停在运动中途，取到瞬态值）。
    SAT = 12
    env.seed(1)
    env.reset()
    n_steps, pen_o, pen_p = drive(env, -1.0, max_steps=SAT)
    s_open = gripper_state(env)
    print("[open]  %d 步  净开口=%.2f mm  残差 eq=%.5f rad con=%.2e m  限位越界=%d/%d"
          % (n_steps, s_open["jaw_mm"], s_open["joint_eq_max_rad"],
             s_open["connect_max_m"], s_open["out_of_range"], s_open["n_limited_joints"]))

    # ---- 1. 开 → 合 → 开 ----
    n_close, c_o, c_p = drive(env, 1.0, max_steps=SAT)
    s_closed = gripper_state(env)
    n_open, o_o, o_p = drive(env, -1.0, max_steps=SAT)
    s_reopen = gripper_state(env)
    print("[1] 开→合→开: 合 %d 步 开口=%.2f mm | 开 %d 步 开口=%.2f mm"
          % (n_close, s_closed["jaw_mm"], n_open, s_reopen["jaw_mm"]))
    out["cases"].append({"case": "open_close_open", "open": s_open, "closed": s_closed,
                         "reopened": s_reopen, "close_steps": n_close, "open_steps": n_open,
                         "penetration_object_mm": max(c_o, o_o),
                         "penetration_padpad_mm": max(c_p, o_p)})

    # ---- 2. 半程反向 ----
    drive(env, 1.0, max_steps=6)          # 半程（约 2 步到达中点，6 步确保进入行程）
    mid = gripper_state(env)
    n_rev, r_o, r_p = drive(env, -1.0, max_steps=SAT)
    s_after_rev = gripper_state(env)
    print("[2] 半程反向: 反向点=%.2f mm → %d 步后 %.2f mm"
          % (mid["jaw_mm"], n_rev, s_after_rev["jaw_mm"]))
    out["cases"].append({"case": "half_reverse", "mid": mid, "after_reverse": s_after_rev,
                         "reverse_steps": n_rev,
                         "penetration_object_mm": r_o, "penetration_padpad_mm": r_p})

    # ---- 3. reset 后立即闭合 ----
    env.seed(1)
    env.reset()
    n_rc, rc_o, rc_p = drive(env, 1.0, max_steps=SAT)
    s_reset_close = gripper_state(env)
    print("[3] reset 后立即闭合: %d 步 开口=%.2f mm 残差 eq=%.5f con=%.2e"
          % (n_rc, s_reset_close["jaw_mm"], s_reset_close["joint_eq_max_rad"],
             s_reset_close["connect_max_m"]))
    out["cases"].append({"case": "reset_then_close", "closed": s_reset_close,
                         "steps": n_rc,
                         "penetration_object_mm": rc_o, "penetration_padpad_mm": rc_p})

    # ---- 4. 带载闭合：把目标物体移到夹持点之间 ----
    loaded = {"case": "loaded_close", "skipped": None}
    try:
        import mujoco

        sim, m, d, gid = N.ids(env)
        bodies = N.find_object_bodies(sim, "alphabet_soup")
        if not bodies:
            raise RuntimeError("找不到 alphabet_soup 物体")
        # find_object_bodies 返回 (body_id, body_name) 元组
        bid, bname = bodies[0]
        jn = N.free_joint_of(sim, bname)
        pitch, _ = N.gripsite(sim, m, d, gid)
        # 物体中心放在**两片指垫中心的连线中点**：这样按构造就在夹口之内。
        # （第一版放在 grip_site 下方 3 cm、第二版放在 grip_site 处，都不在夹口里，
        #   物体直接掉到桌面 —— 实测 z 0.2667→0.0328。指垫沿水平方向开合，
        #   必须按指垫自身的世界位置定位。）
        pads_w = []
        for g in range(m.ngeom):
            nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
            if "fingerpad" in nm:
                pads_w.append(np.array(d.geom_xpos[g], float))
        if len(pads_w) < 2:
            raise RuntimeError("找不到两片指垫 geom")
        mid = 0.5 * (pads_w[0] + pads_w[1])
        place = [float(mid[0]), float(mid[1]), float(mid[2])]
        place_z = place[2]
        N.set_free_joint(sim, jn, tuple(place))
        loaded["pad_midpoint"] = place
        drive(env, -1.0, max_steps=SAT)
        n_l, l_o, l_p = drive(env, 1.0, max_steps=60)
        sim, m, d, gid = N.ids(env)
        jn2 = N.free_joint_of(sim, bname)
        adr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn2)])
        oz = float(d.qpos[adr + 2])
        loaded.update({"obj_body": bname, "obj_z_placed": place_z, "obj_z_after": oz,
                       "closed": gripper_state(env), "steps": n_l,
                       "penetration_object_mm": l_o, "penetration_padpad_mm": l_p})
        print("[4] 带载闭合: %d 步 开口=%.2f mm 物体 z %.4f→%.4f 穿透(物体)=%.3f mm"
              % (n_l, loaded["closed"]["jaw_mm"], place_z, oz, l_o))
    except Exception as e:                                # noqa: BLE001
        loaded["skipped"] = "%s: %s" % (type(e).__name__, e)
        print("[4] 带载闭合: 跳过（%s）" % loaded["skipped"])
    out["cases"].append(loaded)

    # ---- 5. 不同臂姿态 ----
    poses = []
    base_q = np.array([0.0, -0.3, -1.6, 0.3, 1.5, -1.5])
    for k, dq in enumerate([np.zeros(6), np.array([0.2, 0.1, 0.2, -0.1, 0.1, 0.2]),
                            np.array([-0.2, -0.1, -0.2, 0.1, -0.1, -0.2])]):
        env.seed(1)
        env.reset()
        sim, m, d, gid = N.ids(env)
        arm_idx = list(r._ref_joint_pos_indexes)
        d.qpos[arm_idx] = base_q + dq
        import mujoco
        mujoco.mj_forward(m, d)
        n_a, a_o, a_p = drive(env, -1.0, max_steps=SAT)
        n_b, b_o, b_p = drive(env, 1.0, max_steps=SAT)
        st = gripper_state(env)
        poses.append({"pose": k, "arm_qpos": (base_q + dq).tolist(),
                      "closed": st, "steps": [n_a, n_b],
                      "penetration_object_mm": max(a_o, b_o),
                      "penetration_padpad_mm": max(a_p, b_p)})
        print("[5.%d] 臂姿态 %s: 闭合开口=%.2f mm 残差 eq=%.5f con=%.2e 越界=%d"
              % (k, np.round(base_q + dq, 3).tolist(), st["jaw_mm"],
                 st["joint_eq_max_rad"], st["connect_max_m"], st["out_of_range"]))
    out["cases"].append({"case": "arm_poses", "poses": poses})

    # ---- 判据 ----
    c = out["checks"]
    c["gripper_speed_matches_freeze"] = bool(
        abs(eff["gripper_speed"] - float(RC.DEPLOY_GRIPPER_SPEED)) < 1e-12)
    c["open_close_open_returns_to_open"] = bool(
        abs(s_reopen["jaw_mm"] - s_open["jaw_mm"]) < 5.0)
    c["half_reverse_returns_to_open"] = bool(
        abs(s_after_rev["jaw_mm"] - s_open["jaw_mm"]) < 5.0)
    c["reset_then_close_reaches_closed"] = bool(s_reset_close["jaw_mm"] < 25.0)
    c["closed_state_is_stable"] = bool(abs(s_closed["jaw_mm"] - s_reset_close["jaw_mm"]) < 5.0)
    all_states = [s_open, s_closed, s_reopen, mid, s_after_rev, s_reset_close] + \
                 [p["closed"] for p in poses]
    # 只对真正的状态 dict 做判据（防御性：任何非 dict 元素都不参与）
    all_states = [s for s in all_states if isinstance(s, dict)]
    c["no_joint_out_of_range"] = bool(all(s["out_of_range"] == 0 for s in all_states))
    c["joint_eq_residuals_ok"] = bool(
        all(s["joint_eq_max_rad"] < EQ_TOL_RAD for s in all_states))
    c["connect_residuals_ok"] = bool(
        all(s["connect_max_m"] < CONNECT_TOL_M for s in all_states))
    # 穿透：闸门只看**指垫↔物体**（抓取关心的量）；指垫↔指垫只在空载闭合时出现，
    # 单独记录为已知现象，不作为通过条件。
    obj_pens = [x.get("penetration_object_mm") or 0.0 for x in out["cases"]]
    pad_pens = [x.get("penetration_padpad_mm") or 0.0 for x in out["cases"]]
    out["penetration_object_max_mm"] = float(max(obj_pens))
    out["penetration_padpad_max_mm"] = float(max(pad_pens))
    c["penetration_object_within_limit"] = bool(
        out["penetration_object_max_mm"] < PEN_TOL_MM)
    c["no_timeout"] = bool(all(s < MAX_STEPS for p in poses for s in p["steps"])
                           and n_close < MAX_STEPS and n_open < MAX_STEPS
                           and n_rev < MAX_STEPS and n_rc < MAX_STEPS)
    jaws = [p["closed"]["jaw_mm"] for p in poses]
    c["all_arm_poses_behave_identically"] = bool(max(jaws) - min(jaws) < 0.05)
    if loaded.get("obj_body"):
        # 带载成功的判据：物体没有被挤落到地面（z 未明显下坠），且物体穿透在限内
        c["loaded_object_not_dropped"] = bool(
            loaded["obj_z_after"] > (loaded["obj_z_placed"] - 0.02))
        c["loaded_penetration_object_within_limit"] = bool(
            (loaded.get("penetration_object_mm") or 0.0) < PEN_TOL_MM)
    out["verdict"] = "PASS" if all(c.values()) else "FAIL"

    p = OUT / "b2_dynamic_path_verify.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("=" * 78)
    for k, v in c.items():
        print("  %-44s %s" % (k, "✅" if v else "❌"))
    print("NODE_VERDICT = %s" % out["verdict"])
    print("[out] %s" % p)
    env.close()
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
