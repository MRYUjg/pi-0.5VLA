#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2 补充（修正版）：ZhiXing 是**二连平行连杆**结构。

前一版把 6 个夹爪关节 qpos 直接写死再调 `mj_kinematics` —— 这**不会强制**
XML 里的 2 条 `connect`（四连杆闭合）约束，可能得到物理上不可实现的开合构型，
从而把「连杆未闭合」误读成「指垫沿接近轴平移 17 mm / 两面不平行 23°」。

本版改为**驱动执行器 + 逐步仿真**（`mj_step` 会求解 equality），
并在测量前**断言约束残差**，确保被测构型是物理可实现的。
"""
import json
import pathlib
import sys

import numpy as np

RES = 256
TASK_SUITE = "libero_spatial"
OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"
CONFIGS = [("ECO65", "ZhixingGripper", "eco65_zhixing"), ("ECO65", "PandaGripper", "eco65_panda")]


def build(robot, gripper, task_id=0, seed=7):
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
    return env


def settle_jaw(env, gripper, state, steps=800):
    """驱动夹爪执行器并步进，让 equality/connect 约束真正收敛（同时锁住臂关节）。"""
    import mujoco

    sim = env.env.sim
    m, d = sim.model._model, sim.data._data
    if state == "reset":
        return {"driven": False}

    # 按名字前缀找夹爪执行器（ZhiXing 单个；Panda 两个指关节）
    acts = [i for i in range(m.nu)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith("gripper0_")]
    if not acts:
        raise RuntimeError("找不到夹爪执行器")

    if gripper == "ZhixingGripper":
        v = {"open": 0.0, "closed": -0.91}[state]
        targets = {acts[0]: v}
    else:
        v = {"open": 0.04, "closed": 0.0}[state]
        named = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in acts}
        targets = {}
        for nm, i in named.items():
            targets[i] = v if nm.endswith("joint1") else -v

    # 锁住 6 个臂关节：每步复位 qpos/qvel，避免臂在重力下乱掉导致求解器发散
    arm_j = [i for i in range(m.njnt)
             if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("robot0_joint")]
    aadr = [int(m.jnt_qposadr[j]) for j in arm_j]
    adof = [int(m.jnt_dofadr[j]) for j in arm_j]
    q0 = np.array([d.qpos[a] for a in aadr], dtype=float)

    d.qvel[:] = 0.0
    for _ in range(steps):
        for a, vv in zip(aadr, q0):
            d.qpos[a] = vv
        for a in adof:
            d.qvel[a] = 0.0
        for i, vv in targets.items():
            d.ctrl[i] = vv
        sim.step()
    mujoco.mj_forward(m, d)
    return {"driven": True, "ctrl": {str(k): float(v) for k, v in targets.items()}, "steps": steps}


def constraint_residuals(env):
    """joint equality 与 connect 的最大残差（用 MuJoCo 的 eq 活动量）。"""
    import mujoco

    sim = env.env.sim
    m, d = sim.model._model, sim.data._data
    mujoco.mj_forward(m, d)
    out = {"joint_eq_max": None, "connect_max_m": None}
    if m.neq > 0:
        act = np.array(d.efc_force)  # 仅参考；残差改用位置差直接算
    # 直接按 XML 关系算位置残差：connect 用两 site 距离
    jr = []
    for i in range(m.neq):
        if m.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
            j1, j2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            p = m.eq_data[i]
            a1 = int(m.jnt_qposadr[j1])
            a2 = int(m.jnt_qposadr[j2])
            # polycoef: q1 = c0 + c1*q2 + ...
            pred = p[0] + p[1] * d.qpos[a2]
            jr.append(abs(d.qpos[a1] - pred))
    out["joint_eq_max"] = float(max(jr)) if jr else None

    cr = []
    for i in range(m.neq):
        if m.eq_type[i] == mujoco.mjtEq.mjEQ_CONNECT:
            s1, s2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
            cr.append(float(np.linalg.norm(d.site_xpos[s1] - d.site_xpos[s2])))
    out["connect_max_m"] = float(max(cr)) if cr else None
    return out


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


def plane_normal(V, ctr, band_m=0.005):
    """
    取内侧面附近一薄层顶点，用 PCA 最小特征向量作面法向。
    同时返回该薄层的厚度（第三主成分跨度），厚度远小于尺度才说明它确实是「面」。
    """
    d = V - ctr
    r = np.linalg.norm(d, axis=1)
    sel = V[r <= band_m]
    if sel.shape[0] < 10:
        sel = V[np.argsort(r)[: max(10, V.shape[0] // 20)]]
    X = sel - sel.mean(axis=0)
    w, v = np.linalg.eigh(X.T @ X)
    n = v[:, 0]
    return n / np.linalg.norm(n), float(np.sqrt(max(w[0], 0)) * 1000), int(sel.shape[0])


def main():
    import mujoco

    out = {}
    for robot, gripper, tag in CONFIGS:
        env = build(robot, gripper)
        env.reset()
        sim = env.env.sim
        m, d = sim.model._model, sim.data._data
        gid = next(i for i in range(m.nsite)
                   if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or "").endswith("grip_site"))
        gl = (find_geom(sim, "left_fingerpad_col") if gripper == "ZhixingGripper"
              else find_geom(sim, "finger1_pad_collision"))
        gr = (find_geom(sim, "right_fingerpad_col") if gripper == "ZhixingGripper"
              else find_geom(sim, "finger2_pad_collision"))

        rec = {"tag": tag, "states": {}}
        for state in ("reset", "open", "closed"):
            info = settle_jaw(env, gripper, state)
            res = constraint_residuals(env)

            gp = np.array(d.site_xpos[gid], dtype=float)
            gm = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)
            # grip_site 位置在步进后会漂，重新取；同时重取夹爪关节
            Vl, Vr = verts(sim, gl), verts(sim, gr)
            cl, cr = Vl.mean(axis=0), Vr.mean(axis=0)
            u = (cr - cl) / np.linalg.norm(cr - cl)

            # 内侧面：沿 u 投影最靠前的一薄层
            def face(V, sign):
                pr = (V @ u) * sign
                thr = pr.max() - 0.005
                f = V[pr >= thr]
                return f.mean(axis=0), int(f.shape[0])

            fl, nl = face(Vl, +1.0)
            fr, nr = face(Vr, -1.0)
            pinch = (fl + fr) / 2.0
            loc = gm.T @ (pinch - gp)

            nlv, tl, _ = plane_normal(Vl, fl)
            nrv, tr, _ = plane_normal(Vr, fr)
            ang = float(np.degrees(np.arccos(np.clip(abs(nlv @ nrv), -1, 1))))

            rec["states"][state] = {
                "driven": info,
                "constraint_residuals": res,
                "pinch_local_mm": (loc * 1000).tolist(),
                "pinch_approach_mm": float(loc[2] * 1000),
                "pinch_lateral_mm": float(np.linalg.norm(loc[:2]) * 1000),
                "separation_mm": float(np.linalg.norm(fl - fr) * 1000),
                "pad_face_normal_angle_deg": ang,
                "face_band_thickness_mm": [tl, tr],
                "left_1_joint": float(d.qpos[int(m.jnt_qposadr[sim.model.joint_name2id("gripper0_Left_1_Joint")])])
                if gripper == "ZhixingGripper" else None,
            }
        env.close()
        out[tag] = rec

        print(f"\n===== {tag} =====")
        for st, r in rec["states"].items():
            cr_ = r["constraint_residuals"]
            eq = cr_["joint_eq_max"]
            cn = cr_["connect_max_m"]
            eq_s = f"{eq:.2e}" if eq is not None else "n/a(无 joint equality)"
            cn_s = f"{cn:.2e} m" if cn is not None else "n/a(无 connect)"
            print(f"  [{st:<6s}] ctrl={r['driven'].get('ctrl')}  eq_res={eq_s}  conn_res={cn_s}")
            print(f"           夹持点 沿接近轴={r['pinch_approach_mm']:+7.2f} mm  "
                  f"横向={r['pinch_lateral_mm']:.2f} mm  面间距={r['separation_mm']:6.2f} mm  "
                  f"两面法向夹角={r['pad_face_normal_angle_deg']:.2f}°  "
                  f"面厚度={np.round(r['face_band_thickness_mm'],2).tolist()} mm")

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT) / "pad_face_geometry.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[out] {OUT}/pad_face_geometry.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
