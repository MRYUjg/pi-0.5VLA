#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2-v2 诊断：当前资产下 ZhiXing 夹爪「闭合端」在**动力学驱动**下是否收敛。

背景：在本轮 N2-v2 重测中，probe_pad_face.py 与 n2v2_settled_probe.py 都发现
ECO65+ZhixingGripper 的 `closed` 态（ctrl=-0.91）没有收敛：
Left_1 = -1.353 rad（越出关节范围 [-0.91, 0]），joint equality 残差 1.43 rad，
connect 残差 18 mm。而 N2 历史产物（同一脚本、旧资产/旧臂位姿）是收敛的。

本脚本回答：
  A. 加大步数（800/2000/5000/10000）是否能收敛？给出 Left_1 轨迹。
  B. 从 reset 直接闭（不先开）是否收敛？
  C. 换成 R1 之前的旧臂位姿是否收敛？（区分“资产问题”vs“臂位姿问题”）
  D. **运动学装配**（按 XML joint equality 给出自洽构型，不做动力学）在
     Left_1 = 0 / -0.91 两个端点的残差、夹持点偏移、开口是多少？
  E. 闭合过程中夹爪 geom 是否与桌面/物体发生接触？

只做仿真内状态修改，不写任何生产资产。
"""
import json
import math
import pathlib
import sys

import numpy as np

RES = 256
TASK_SUITE = "libero_spatial"
OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results/n2v2_20260928"

# R1 之前的 ZhixingGripper init_qpos（旧契约 initial_pose.landed_init_qpos）
OLD_ZX_ARM_QPOS = [-0.12822699637501933, -0.08018875108093924, -1.4820718667163302,
                   -0.06491839613575459, 1.5635290046770178, -1.6993079970320994]
# 当前落地（gripper-dispatched）
CUR_ZX_ARM_QPOS = [-0.12639, -0.222035, -0.841572, -0.563545, 1.563635, -1.697476]
# ECO65 + PandaGripper 分派解
CUR_PANDA_ARM_QPOS = [-0.128227, -0.080175, -1.482052, -0.064915, 1.563529, -1.699308]

GJ = ["gripper0_Left_1_Joint", "gripper0_Left_Support_Joint", "gripper0_Left_2_Joint",
      "gripper0_Right_2_Joint", "gripper0_Right_1_Joint", "gripper0_Right_Support_Joint"]


def build(robot="ECO65", gripper="ZhixingGripper", task_id=0, seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kw = {"bddl_file_name": bddl, "camera_heights": RES, "camera_widths": RES, "robots": [robot]}
    if gripper and gripper != "default":
        kw["gripper_types"] = gripper
    env = OffScreenRenderEnv(**kw)
    env.seed(seed)
    return env


def arm_qadr(sim):
    import mujoco

    m = sim.model._model
    js = [i for i in range(m.njnt)
          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("robot0_joint")]
    js.sort(key=lambda j: int("".join(
        c for c in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "") if c.isdigit())))
    return js


def set_arm(sim, q):
    import mujoco

    m, d = sim.model._model, sim.data._data
    for j, v in zip(arm_qadr(sim), q):
        d.qpos[int(m.jnt_qposadr[j])] = float(v)
    mujoco.mj_forward(m, d)


def set_jaw_consistent(sim, left1):
    """按 XML joint equality 写自洽构型：Left_1=v, Left_2=v, Support=-v, Right_1=-v, Right_2=-v, Right_Support=-v。"""
    import mujoco

    m, d = sim.model._model, sim.data._data
    vals = {"gripper0_Left_1_Joint": left1, "gripper0_Left_2_Joint": left1,
            "gripper0_Left_Support_Joint": -left1, "gripper0_Right_1_Joint": -left1,
            "gripper0_Right_2_Joint": -left1, "gripper0_Right_Support_Joint": -left1}
    for n, v in vals.items():
        d.qpos[int(m.jnt_qposadr[sim.model.joint_name2id(n)])] = float(v)
    mujoco.mj_forward(m, d)


def residuals(sim):
    import mujoco

    m, d = sim.model._model, sim.data._data
    mujoco.mj_forward(m, d)
    jr, cr = [], []
    for i in range(m.neq):
        if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
            j1, j2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            p = m.eq_data[i]
            jr.append(abs(d.qpos[int(m.jnt_qposadr[j1])]
                          - (p[0] + p[1] * d.qpos[int(m.jnt_qposadr[j2])])))
        elif m.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT:
            s1, s2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            cr.append(float(np.linalg.norm(d.site_xpos[s1] - d.site_xpos[s2])))
    return {"joint_eq_max": float(max(jr)) if jr else None,
            "connect_max_m": float(max(cr)) if cr else None}


def verts(sim, gid):
    import mujoco

    m, d = sim.model._model, sim.data._data
    pos = np.array(d.geom_xpos[gid], dtype=float)
    mat = np.array(d.geom_xmat[gid], dtype=float).reshape(3, 3)
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gid]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = np.array(m.mesh_vert[a:a + n], dtype=float)
    else:
        s = np.array(m.geom_size[gid], dtype=float)
        v = s * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (mat @ v.T).T + pos


def find_geom(sim, *subs):
    import mujoco

    m = sim.model._model
    for i in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
        if all(s.lower() in n for s in subs):
            return i
    return None


def geometry(sim):
    import mujoco

    m, d = sim.model._model, sim.data._data
    gid = None
    for i in range(m.nsite):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or "").endswith("grip_site"):
            gid = i
    gp = np.array(d.site_xpos[gid], dtype=float)
    gm = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)
    gl = find_geom(sim, "left_fingerpad_col")
    gr = find_geom(sim, "right_fingerpad_col")
    ret = {}
    if gl is not None and gr is not None:
        Vl, Vr = verts(sim, gl), verts(sim, gr)
        cl, cr = Vl.mean(axis=0), Vr.mean(axis=0)
        u = (cr - cl) / max(np.linalg.norm(cr - cl), 1e-12)
        fl = Vl[(Vl @ u) >= (Vl @ u).max() - 0.005].mean(axis=0)
        fr = Vr[(Vr @ -u) >= (Vr @ -u).max() - 0.005].mean(axis=0)
        pinch = (fl + fr) / 2.0
        loc = gm.T @ (pinch - gp)
        ret["pad_center_separation_mm"] = float(np.linalg.norm(cl - cr) * 1000)
        ret["grasp_along_mm"] = float(loc[2] * 1000)
        ret["pad_face_separation_mm"] = float(np.linalg.norm(fl - fr) * 1000)
    ret["grip_site_z"] = float(gp[2])
    # 夹爪与外部 geom 的接触数
    contacts = 0
    names = []
    gg = set()
    for g_ in range(m.ngeom):
        bid = int(m.geom_bodyid[g_])
        chain, guard = [], 0
        while bid != 0 and guard < 64:
            chain.append(bid)
            bid = int(m.body_parentid[bid])
            guard += 1
        if any("right_gripper" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
               for b in chain):
            gg.add(g_)
    for c in range(d.ncon):
        con = d.contact[c]
        g1, g2 = int(con.geom1), int(con.geom2)
        if g1 in gg or g2 in gg:
            contacts += 1
            names.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1),
                          mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2)))
    ret["gripper_contact_count"] = contacts
    ret["gripper_contacts"] = names[:10]
    ret["left_1_qpos"] = float(d.qpos[int(m.jnt_qposadr[sim.model.joint_name2id(
        "gripper0_Left_1_Joint")])])
    return ret


def drive_closed(sim, q_arm, steps, from_open=False, trace_every=0):
    import mujoco

    m, d = sim.model._model, sim.data._data
    set_arm(sim, q_arm)
    if from_open:
        set_jaw_consistent(sim, 0.0)
    else:
        set_jaw_consistent(sim, -0.91)
    mujoco.mj_forward(m, d)

    jadr = [int(m.jnt_qposadr[j]) for j in arm_qadr(sim)]
    jdof = [int(m.jnt_dofadr[j]) for j in arm_qadr(sim)]
    gripper_act = [i for i in range(m.nu)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
                   .startswith("gripper0_")]
    q0 = np.array([d.qpos[a] for a in jadr], dtype=float)
    d.qvel[:] = 0.0
    trace = []
    l1adr = int(m.jnt_qposadr[sim.model.joint_name2id("gripper0_Left_1_Joint")])
    for s in range(steps):
        for a, v in zip(jadr, q0):
            d.qpos[a] = v
        for a in jdof:
            d.qvel[a] = 0.0
        for i in gripper_act:
            d.ctrl[i] = -0.91
        sim.step()
        if trace_every and (s + 1) % trace_every == 0:
            trace.append({"step": s + 1, "left_1": float(d.qpos[l1adr]),
                          "connect": residuals(sim)["connect_max_m"],
                          "eq": residuals(sim)["joint_eq_max"]})
    mujoco.mj_forward(m, d)
    out = {"steps": steps, "from_open": from_open}
    out.update(geometry(sim))
    out["residuals"] = residuals(sim)
    if trace:
        out["trace"] = trace
    return out


def main():
    out = {"note": "只读诊断；不改生产资产", "cases": {}}

    # ---- D. 运动学装配（不做动力学）----
    env = build()
    env.reset()
    sim = env.env.sim
    kin = {}
    for name, q_arm in (("cur_zhixing_qpos", CUR_ZX_ARM_QPOS),
                        ("old_pre_r1_zhixing_qpos", OLD_ZX_ARM_QPOS)):
        set_arm(sim, q_arm)
        for l1 in (0.0, -0.91, -0.5):
            set_jaw_consistent(sim, l1)
            rec = {"left_1_target": l1}
            rec.update(geometry(sim))
            rec["residuals"] = residuals(sim)
            kin[f"{name}__left1_{l1}"] = rec
            print(f"[kin] {name:<26s} left1={l1:+6.3f} eq={rec['residuals']['joint_eq_max']} "
                  f"conn={rec['residuals']['connect_max_m']} sep={rec.get('pad_center_separation_mm')} "
                  f"grasp_along={rec.get('grasp_along_mm')}")
    out["cases"]["kinematic_assembly"] = kin
    env.close()

    # ---- A/B/C. 动力学驱动 ----
    for tag, q_arm in (("current_landed", CUR_ZX_ARM_QPOS),
                       ("pre_r1_landed", OLD_ZX_ARM_QPOS),
                       ("panda_dispatched", CUR_PANDA_ARM_QPOS)):
        for from_open in (False, True):
            key = f"drive__{tag}__from_{'open' if from_open else 'closed'}"
            env = build()
            env.reset()
            if tag == "panda_dispatched":
                # 该臂位姿属于 ECO65+PandaGripper 分派解；仅在 ZhiXing 上做对照
                pass
            sim = env.env.sim
            try:
                r = drive_closed(sim, q_arm, 10000, from_open=from_open, trace_every=1000)
            except Exception as e:
                import traceback
                traceback.print_exc()
                r = {"error": f"{type(e).__name__}: {e}"}
            out["cases"][key] = r
            print(f"[dyn] {key}: left1={r.get('left_1_qpos')} eq={dig(r,'residuals','joint_eq_max')} "
                  f"conn={dig(r,'residuals','connect_max_m')} sep={r.get('pad_center_separation_mm')} "
                  f"grasp_along={r.get('grasp_along_mm')} contacts={r.get('gripper_contact_count')}")
            if r.get("trace"):
                print("      trace:", [(t["step"], round(t["left_1"], 4)) for t in r["trace"]])
            env.close()

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT) / "n2v2_close_diag.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {OUT}/n2v2_close_diag.json")
    return 0


def dig(o, *ks):
    cur = o
    for k in ks:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


if __name__ == "__main__":
    sys.exit(main())
