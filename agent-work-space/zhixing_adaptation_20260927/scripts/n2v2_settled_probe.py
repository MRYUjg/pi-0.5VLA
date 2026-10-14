#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2-v2：在当前资产（eef=0.20254 / R1 夹爪分派 init_qpos / 极性修正 / 速度开关）下，
对三个配置重新测量 TCP、相机变换、开合状态、初始位姿与腕部视觉占比。

两个口径（都落盘，避免把不可实现的构型当成真值）：
  A. `geometry`            —— **约束自洽装配口径**（primary）：
       按 XML joint equality 写出自洽构型（ZhiXing: Left_1=v, 其余按 equality 取符号；
       Panda: finger1=v, finger2=-v），再报告 joint equality 与两条 connect 的残差。
  B. `geometry_driven`     —— **执行器驱动 + mj_step 口径**（cross-check）：
       与 probe_pad_face.py 完全同口径（锁臂 + 800 步），并给出是否收敛的判据。

两者都记录，是因为本轮实测发现：在当前臂位姿下，ZhiXing 的 closed 态在动力学驱动下
**不收敛**（Left_1 越过关节范围到 -1.35 rad，connect 残差 18 mm），
而自洽装配口径残差为 0。详见 n2v2_close_diag.json。

产物：results/n2v2_20260928/n2v2_settled_probe.json
"""
import argparse
import json
import math
import os
import pathlib
import re
import sys

import numpy as np

RES = 256
TASK_SUITE = "libero_spatial"
DEFAULT_OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results/n2v2_20260928"

CONFIGS = [
    ("ECO65", "ZhixingGripper", "eco65_zhixing"),
    ("ECO65", "PandaGripper", "eco65_panda"),
    ("Panda", "PandaGripper", "panda_panda"),
]
STATES = ["reset", "open", "closed"]

GRIP_CTRL = {
    "ZhixingGripper": {"open": 0.0, "closed": -0.91},
    "PandaGripper": {"open": 0.04, "closed": 0.0},
}
# 自洽装配：ZhiXing 关节名后缀 -> Left_1 的系数（见 zhixing_gripper.xml 的 5 条 joint equality）
ZX_EQ_SIGNS = [("Left_1", +1.0), ("Left_2", +1.0), ("Left_Support", -1.0),
               ("Right_1", -1.0), ("Right_2", -1.0), ("Right_Support", -1.0)]
PD_SIGNS = [("finger_joint1", +1.0), ("finger_joint2", -1.0)]

SETTLE_STEPS = 800
CONVERGED_TOL = {"joint_eq_max": 1e-3, "connect_max_m": 5e-4}


def T_of(pos, mat):
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = pos
    return T


def build_env(robot, gripper, task_id=0, seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kw = {"bddl_file_name": bddl, "camera_heights": RES, "camera_widths": RES,
          "robots": [robot]}
    if gripper and gripper != "default":
        kw["gripper_types"] = gripper
    env = OffScreenRenderEnv(**kw)
    env.seed(seed)
    return env, suite, task


def arm_joint_ids(sim):
    import mujoco

    m = sim.model._model
    js = [i for i in range(m.njnt)
          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("robot0_joint")]
    js.sort(key=lambda j: int(re.sub(r"\D", "",
                                     mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "0")))
    return js


def gripper_joint_names(sim):
    import mujoco

    m = sim.model._model
    out = []
    for i in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if n.startswith("gripper0_"):
            out.append(n)
    return out


def geom_vertices_world(sim, gid):
    import mujoco

    m, d = sim.model._model, sim.data._data
    pos = np.array(d.geom_xpos[gid], dtype=float)
    mat = np.array(d.geom_xmat[gid], dtype=float).reshape(3, 3)
    gt = m.geom_type[gid]
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gid]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = np.array(m.mesh_vert[a:a + n], dtype=float)
    elif gt == mujoco.mjtGeom.mjGEOM_BOX:
        s = np.array(m.geom_size[gid], dtype=float)
        v = s * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    else:
        s = np.array(m.geom_size[gid], dtype=float)
        v = np.array([[0, 0, -s[1]], [0, 0, s[1]]], dtype=float)
    return (mat @ v.T).T + pos


def find_geom(sim, *subs):
    import mujoco

    m = sim.model._model
    for i in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
        if all(s.lower() in n for s in subs):
            return i
    return None


def pad_geom_ids(sim, gripper):
    if gripper == "ZhixingGripper":
        return find_geom(sim, "left_fingerpad_col"), find_geom(sim, "right_fingerpad_col")
    return find_geom(sim, "finger1_pad_collision"), find_geom(sim, "finger2_pad_collision")


def plane_normal(V, ctr, band_m=0.005):
    d = V - ctr
    r = np.linalg.norm(d, axis=1)
    sel = V[r <= band_m]
    if sel.shape[0] < 10:
        sel = V[np.argsort(r)[: max(10, V.shape[0] // 20)]]
    X = sel - sel.mean(axis=0)
    w, v = np.linalg.eigh(X.T @ X)
    n = v[:, 0]
    return n / np.linalg.norm(n)


def gripper_geom_ids(sim):
    import mujoco

    m = sim.model._model
    out = []
    for gid in range(m.ngeom):
        bid = int(m.geom_bodyid[gid])
        chain, guard = [], 0
        while bid != 0 and guard < 64:
            chain.append(bid)
            bid = int(m.body_parentid[bid])
            guard += 1
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in chain]
        if any("right_gripper" in n for n in names):
            out.append(gid)
    return out


def find_suffix(sim, objtype, suffix):
    import mujoco

    raw = sim.model._model
    n = {mujoco.mjtObj.mjOBJ_SITE: raw.nsite, mujoco.mjtObj.mjOBJ_CAMERA: raw.ncam}[objtype]
    for i in range(n):
        nm = mujoco.mj_id2name(raw, objtype, i) or ""
        if nm.endswith(suffix):
            return i, nm
    raise RuntimeError(f"找不到以 {suffix} 结尾的 {objtype}")


def constraint_residuals(sim):
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
    return {"n_joint_eq": len(jr), "n_connect": len(cr),
            "joint_eq_max": float(max(jr)) if jr else None,
            "connect_max_m": float(max(cr)) if cr else None}


def set_assembly(sim, gripper, state):
    """按约束自洽构型写夹爪关节角（不做动力学）。"""
    import mujoco

    m, d = sim.model._model, sim.data._data
    if state == "reset":
        return {"applied": False}
    v = GRIP_CTRL[gripper][state]
    signs = ZX_EQ_SIGNS if gripper == "ZhixingGripper" else PD_SIGNS
    names = gripper_joint_names(sim)
    applied = {}
    for suffix, sgn in signs:
        hit = next((n for n in names if suffix in n), None)
        if hit is None:
            raise RuntimeError(f"找不到夹爪关节 *{suffix} in {names}")
        d.qpos[int(m.jnt_qposadr[sim.model.joint_name2id(hit)])] = float(sgn * v)
        applied[hit] = float(sgn * v)
    mujoco.mj_forward(m, d)
    return {"applied": True, "target": v, "joint_qpos": applied}


def settle_jaw_ramped(sim, gripper, state, steps=SETTLE_STEPS, ramp_steps=100):
    """执行器驱动，但 ctrl 按部署路径**斜坡**变化（对应 format_action 在 speed=0.01 下的累积）。

    与 settle_jaw 的阶跃输入相对照：阶跃到 ctrl=-0.91 会让四连杆卡死（见 n2v2_close_diag.json），
    斜坡输入则是部署时真实发生的情况。
    """
    import mujoco

    m, d = sim.model._model, sim.data._data
    if state == "reset":
        return {"driven": False}
    acts = [i for i in range(m.nu)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
            .startswith("gripper0_")]
    v = GRIP_CTRL[gripper][state]
    aj = arm_joint_ids(sim)
    aadr = [int(m.jnt_qposadr[j]) for j in aj]
    adof = [int(m.jnt_dofadr[j]) for j in aj]
    q0 = np.array([d.qpos[a] for a in aadr], dtype=float)
    d.qvel[:] = 0.0
    for s in range(steps):
        frac = min(1.0, float(s + 1) / ramp_steps)
        for a, vv in zip(aadr, q0):
            d.qpos[a] = vv
        for a in adof:
            d.qvel[a] = 0.0
        for i in acts:
            if gripper == "ZhixingGripper":
                # ctrl 目标 v 对应的归一化命令 u = (v - bias)/weight, bias=-0.455, weight=0.455
                u_target = (v + 0.455) / 0.455
                d.ctrl[i] = -0.455 + 0.455 * (frac * u_target)
            else:
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
                u = frac * (v / 0.04)
                d.ctrl[i] = u if nm.endswith("joint1") else -u
        sim.step()
    mujoco.mj_forward(m, d)
    return {"driven": True, "ramped": True, "target": v, "steps": steps,
            "ramp_steps": ramp_steps}


def settle_jaw(sim, gripper, state, steps=SETTLE_STEPS):
    """执行器驱动 + mj_step（与 probe_pad_face.py 同口径）；臂关节锁死。"""
    import mujoco

    m, d = sim.model._model, sim.data._data
    if state == "reset":
        return {"driven": False}
    acts = [i for i in range(m.nu)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "")
            .startswith("gripper0_")]
    if not acts:
        raise RuntimeError("找不到夹爪执行器")
    v = GRIP_CTRL[gripper][state]
    if gripper == "ZhixingGripper":
        targets = {acts[0]: v}
    else:
        named = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i for i in acts}
        targets = {i: (v if nm.endswith("joint1") else -v) for nm, i in named.items()}

    aj = arm_joint_ids(sim)
    aadr = [int(m.jnt_qposadr[j]) for j in aj]
    adof = [int(m.jnt_dofadr[j]) for j in aj]
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
    return {"driven": True, "ctrl": {str(k): float(v) for k, v in targets.items()},
            "steps": steps}


def view_stats(sim, cam_name, gripper_gids, res=RES):
    import mujoco

    seg = sim.render(camera_name=cam_name, width=res, height=res, segmentation=True)
    rgb = sim.render(camera_name=cam_name, width=res, height=res)
    if isinstance(rgb, tuple):
        rgb = rgb[0]
    rgb = np.asarray(rgb, dtype=np.uint8)[::-1]
    seg = np.asarray(seg)[::-1]
    is_geom = seg[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    gid_map = np.where(is_geom, seg[:, :, 1], -1)
    total = gid_map.size
    mask = np.isin(gid_map, list(int(g) for g in gripper_gids))
    n_grip = int(mask.sum())
    h = gid_map.shape[0]
    bottom = slice(int(h * 0.6), h)
    n_bottom_total = gid_map[bottom, :].size
    n_grip_bottom = int(mask[bottom, :].sum())
    mean_color = rgb[mask].mean(axis=0).tolist() if n_grip > 0 else None
    return {"camera": cam_name, "resolution": res, "gripper_pixels": n_grip,
            "total_pixels": int(total),
            "gripper_pixel_fraction": float(n_grip / total),
            "gripper_pixel_fraction_bottom40": float(n_grip_bottom / n_bottom_total),
            "mean_color_gripper_rgb": mean_color}


# --------------------------------------------------------------------------- #
def measure_config(robot, gripper, tag, poses, out_dir):
    import mujoco

    env, suite, task = build_env(robot, gripper)
    env.reset()
    rb = env.env.robots[0]           # N1 技术坑：reset() 会重建 robot
    sim = env.env.sim

    applied = 0
    for n, v in poses.items():
        try:
            sim.data.set_joint_qpos(n, np.array(v, dtype=float))
            applied += 1
        except Exception:
            pass
    sim.forward()

    gid, gname = find_suffix(sim, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    cid, cname = find_suffix(sim, mujoco.mjtObj.mjOBJ_CAMERA, "eye_in_hand")
    aid, aname = find_suffix(sim, mujoco.mjtObj.mjOBJ_CAMERA, "agentview")
    ggids = gripper_geom_ids(sim)
    gjoints = gripper_joint_names(sim)

    aj = arm_joint_ids(sim)
    landed_qpos = [float(sim.data._data.qpos[int(sim.model._model.jnt_qposadr[j])]) for j in aj]
    landed_names = [mujoco.mj_id2name(sim.model._model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in aj]

    eef_info = None
    eid = ename = rgid = None
    for i in range(sim.model._model.nbody):
        n = mujoco.mj_id2name(sim.model._model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if n.endswith("eef"):
            eid, ename = i, n
        if n.endswith("right_gripper"):
            rgid = i
    if eid is not None and rgid is not None:
        bp = np.array(sim.data._data.xpos[rgid], dtype=float)
        bm = np.array(sim.data._data.xmat[rgid], dtype=float).reshape(3, 3)
        ep = np.array(sim.data._data.xpos[eid], dtype=float)
        eef_info = {"eef_body_name": ename,
                    "right_gripper_body_name": mujoco.mj_id2name(
                        sim.model._model, mujoco.mjtObj.mjOBJ_BODY, rgid),
                    "eef_pos_in_right_gripper_m": (bm.T @ (ep - bp)).tolist(),
                    "eef_world_pos_at_reset": ep.tolist()}

    gl, gr = pad_geom_ids(sim, gripper)

    def snap():
        gpos = np.array(sim.data._data.site_xpos[gid], dtype=float)
        gmat = np.array(sim.data._data.site_xmat[gid], dtype=float).reshape(3, 3)
        cpos = np.array(sim.data._data.cam_xpos[cid], dtype=float)
        cmat = np.array(sim.data._data.cam_xmat[cid], dtype=float).reshape(3, 3)
        T_grip, T_cam = T_of(gpos, gmat), T_of(cpos, cmat)

        Vl, Vr = geom_vertices_world(sim, gl), geom_vertices_world(sim, gr)
        cl, cr = Vl.mean(axis=0), Vr.mean(axis=0)
        u = (cr - cl) / max(np.linalg.norm(cr - cl), 1e-12)
        fl = Vl[(Vl @ u) >= (Vl @ u).max() - 0.005].mean(axis=0)
        fr = Vr[(Vr @ -u) >= (Vr @ -u).max() - 0.005].mean(axis=0)
        pinch = (fl + fr) / 2.0
        loc = gmat.T @ (pinch - gpos)
        nlv, nrv = plane_normal(Vl, fl), plane_normal(Vr, fr)
        ang = float(np.degrees(np.arccos(np.clip(abs(nlv @ nrv), -1, 1))))
        d2 = ((Vl[:, None, :] - Vr[None, :, :]) ** 2).sum(-1)
        i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
        nip_loc = gmat.T @ (((Vl[i] + Vr[j]) / 2.0) - gpos)
        allv = np.vstack([geom_vertices_world(sim, g_) for g_ in ggids]) if ggids \
            else np.zeros((1, 3))
        proj = (allv - gpos) @ gmat[:, 2]

        rec = {
            "grip_site_world_pos": gpos.tolist(),
            "grip_site_world_mat": gmat.tolist(),
            "grasp_point_world_pos": pinch.tolist(),
            "grasp_point_local_m": loc.tolist(),
            "grasp_point_offset_along_approach_mm": float(loc[2] * 1000.0),
            "grasp_point_offset_lateral_mm": float(np.linalg.norm(loc[:2]) * 1000.0),
            "grasp_point_offset_norm_mm": float(np.linalg.norm(loc) * 1000.0),
            "pad_face_separation_mm": float(np.linalg.norm(fl - fr) * 1000.0),
            "pad_face_normal_angle_deg": ang,
            "jaw_opening_pad_center_m": float(np.linalg.norm(cl - cr)),
            "nip_clearance_m": float(np.linalg.norm(Vl[i] - Vr[j])),
            "nip_offset_approach_mm_legacy": float(nip_loc[2] * 1000.0),
            "gripper_extent_plus_z_mm": float(proj.max() * 1000.0),
            "gripper_extent_minus_z_mm": float(proj.min() * 1000.0),
            "T_cam2grip": (np.linalg.inv(T_cam) @ T_grip).tolist(),
            "T_grip2cam": (np.linalg.inv(T_grip) @ T_cam).tolist(),
            "cam_world_pos": cpos.tolist(),
            "dist_cam_to_gripsite_mm": float(np.linalg.norm(cpos - gpos) * 1000.0),
            "constraint_residuals": constraint_residuals(sim),
            "gripper_joint_qpos": {
                n: float(sim.data._data.qpos[int(sim.model._model.jnt_qposadr[
                    sim.model.joint_name2id(n)])]) for n in gjoints},
            "gripper_ctrl": {
                (mujoco.mj_id2name(sim.model._model, mujoco.mjtObj.mjOBJ_ACTUATOR, k) or ""):
                    float(sim.data._data.ctrl[k])
                for k in range(sim.model._model.nu)
                if (mujoco.mj_id2name(sim.model._model, mujoco.mjtObj.mjOBJ_ACTUATOR, k) or "")
                .startswith("gripper0_")},
        }
        prefix = getattr(rb.robot_model, "naming_prefix", "") or ""
        for cand in (prefix + "right_hand", "robot0_right_hand", "right_hand"):
            try:
                hpos = sim.model.body_name2id(cand)
            except Exception:
                continue
            hp = np.array(sim.data._data.xpos[hpos], dtype=float)
            hm = np.array(sim.data._data.xmat[hpos], dtype=float).reshape(3, 3)
            rec["right_hand_body_name"] = mujoco.mj_id2name(
                sim.model._model, mujoco.mjtObj.mjOBJ_BODY, int(hpos))
            rec["T_hand2grip"] = (np.linalg.inv(T_of(hp, hm)) @ T_grip).tolist()
            break
        return rec

    res = {"tag": tag, "robot": robot, "gripper": gripper, "task": task.language,
           "objects_frozen": applied, "gripper_geoms": len(ggids),
           "grip_site_name": gname, "camera_name": cname,
           "gripper_joint_names": gjoints, "gripper_joint_count": len(gjoints),
           "gripper_dof": int(rb.gripper.dof),
           "gripper_speed_asset_default": float(rb.gripper.speed),
           "landed_init_qpos": landed_qpos, "landed_arm_joint_names": landed_names,
           "eef_mount": eef_info,
           "controller": {"type": type(rb.controller).__name__,
                          "output_max": [float(v) for v in np.atleast_1d(rb.controller.output_max)],
                          "input_min": [float(v) for v in np.atleast_1d(rb.controller.input_min)],
                          "input_max": [float(v) for v in np.atleast_1d(rb.controller.input_max)],
                          "control_timestep": float(env.env.control_timestep)},
           "geometry": {}, "geometry_driven": {}, "views": {},
           "assembly_applied": {}, "driven_applied": {}}

    # --- A. 自洽装配口径（primary）：渲染腕部/外部视图 + 几何 ---
    for state in STATES:
        info = set_assembly(sim, gripper, state)
        res["assembly_applied"][state] = info
        res["geometry"][state] = snap()
        res["views"][f"wrist_{state}"] = view_stats(sim, cname, ggids)
        res["views"][f"agentview_{state}"] = view_stats(sim, aname, ggids)
        g = res["geometry"][state]
        print(f"  [{tag}/{state:<6s}] ASSEMBLY grip={np.round(g['grip_site_world_pos'],5).tolist()} "
              f"along={g['grasp_point_offset_along_approach_mm']:+8.3f}mm "
              f"padsep={g['pad_face_separation_mm']:7.3f}mm "
              f"eq={g['constraint_residuals']['joint_eq_max']} "
              f"conn={g['constraint_residuals']['connect_max_m']} "
              f"wrist={res['views'][f'wrist_{state}']['gripper_pixel_fraction']:.5f}")

    # --- B. 执行器驱动口径（cross-check）：不渲染，只测几何 + 收敛性 ---
    env2, _, _ = build_env(robot, gripper)
    env2.reset()
    sim2 = env2.env.sim
    for n, v in poses.items():
        try:
            sim2.data.set_joint_qpos(n, np.array(v, dtype=float))
        except Exception:
            pass
    sim2.forward()
    # 与 probe_pad_face 一致：先 reset 态，再 open，再 closed
    for state in STATES:
        info = settle_jaw(sim2, gripper, state)
        res["driven_applied"][state] = info
        res["geometry_driven"][state] = _snap_on(sim2, gripper, gid, cid, ggids, gjoints, rb)
        gg = res["geometry_driven"][state]
        print(f"  [{tag}/{state:<6s}] DRIVEN   "
              f"along={gg['grasp_point_offset_along_approach_mm']:+8.3f}mm "
              f"padsep={gg['pad_face_separation_mm']:7.3f}mm "
              f"eq={gg['constraint_residuals']['joint_eq_max']} "
              f"conn={gg['constraint_residuals']['connect_max_m']}")
    env2.close()

    # --- C. 部署路径口径（斜坡 ctrl）：cross-check，只测几何 ---
    env3, _, _ = build_env(robot, gripper)
    env3.reset()
    sim3 = env3.env.sim
    for n, v in poses.items():
        try:
            sim3.data.set_joint_qpos(n, np.array(v, dtype=float))
        except Exception:
            pass
    sim3.forward()
    res["geometry_ramped"] = {}
    res["ramped_applied"] = {}
    for state in STATES:
        info = settle_jaw_ramped(sim3, gripper, state)
        res["ramped_applied"][state] = info
        res["geometry_ramped"][state] = _snap_on(sim3, gripper, gid, cid, ggids, gjoints, rb)
        gg = res["geometry_ramped"][state]
        print(f"  [{tag}/{state:<6s}] RAMPED   "
              f"along={gg['grasp_point_offset_along_approach_mm']:+8.3f}mm "
              f"padsep={gg['pad_face_separation_mm']:7.3f}mm "
              f"eq={gg['constraint_residuals']['joint_eq_max']} "
              f"conn={gg['constraint_residuals']['connect_max_m']}")
    env3.close()

    # 收敛判据
    conv = {}
    for state in STATES:
        d = res["geometry_driven"][state]["constraint_residuals"]
        eq, cn = d.get("joint_eq_max"), d.get("connect_max_m")
        ok = True
        if eq is not None and eq > CONVERGED_TOL["joint_eq_max"]:
            ok = False
        if cn is not None and cn > CONVERGED_TOL["connect_max_m"]:
            ok = False
        dr = res["geometry_ramped"][state]["constraint_residuals"]
        eqr, cnr = dr.get("joint_eq_max"), dr.get("connect_max_m")
        okr = True
        if eqr is not None and eqr > CONVERGED_TOL["joint_eq_max"]:
            okr = False
        if cnr is not None and cnr > CONVERGED_TOL["connect_max_m"]:
            okr = False
        a = res["geometry"][state]["grasp_point_offset_along_approach_mm"]
        b = res["geometry_driven"][state]["grasp_point_offset_along_approach_mm"]
        c = res["geometry_ramped"][state]["grasp_point_offset_along_approach_mm"]
        conv[state] = {"driven_step_input_converged": bool(ok),
                       "driven_ramp_input_converged": bool(okr),
                       "tolerance": CONVERGED_TOL,
                       "assembly_minus_driven_grasp_along_mm": float(a - b),
                       "assembly_minus_ramped_grasp_along_mm": float(a - c),
                       "driven_pad_face_separation_mm":
                           res["geometry_driven"][state]["pad_face_separation_mm"],
                       "ramped_pad_face_separation_mm":
                           res["geometry_ramped"][state]["pad_face_separation_mm"],
                       "assembly_pad_face_separation_mm":
                           res["geometry"][state]["pad_face_separation_mm"]}
    res["convergence"] = conv

    res["camera"] = {
        "T_cam2grip_reset": res["geometry"]["reset"]["T_cam2grip"],
        "T_cam2grip_translation_m": np.array(res["geometry"]["reset"]["T_cam2grip"])[:3, 3].tolist(),
        "T_cam2grip_invariant_across_states": bool(all(
            np.allclose(res["geometry"]["reset"]["T_cam2grip"],
                        res["geometry"][s]["T_cam2grip"], atol=1e-12) for s in STATES)),
        "dist_cam_to_gripsite_mm": res["geometry"]["reset"]["dist_cam_to_gripsite_mm"],
    }
    env.close()
    return res


def _snap_on(sim, gripper, gid, cid, ggids, gjoints, rb):
    """在给定 sim 上做一次几何快照（与 measure_config.snap 同口径）。"""
    import mujoco

    gpos = np.array(sim.data._data.site_xpos[gid], dtype=float)
    gmat = np.array(sim.data._data.site_xmat[gid], dtype=float).reshape(3, 3)
    cpos = np.array(sim.data._data.cam_xpos[cid], dtype=float)
    cmat = np.array(sim.data._data.cam_xmat[cid], dtype=float).reshape(3, 3)
    T_grip, T_cam = T_of(gpos, gmat), T_of(cpos, cmat)
    gl, gr = pad_geom_ids(sim, gripper)
    Vl, Vr = geom_vertices_world(sim, gl), geom_vertices_world(sim, gr)
    cl, cr = Vl.mean(axis=0), Vr.mean(axis=0)
    u = (cr - cl) / max(np.linalg.norm(cr - cl), 1e-12)
    fl = Vl[(Vl @ u) >= (Vl @ u).max() - 0.005].mean(axis=0)
    fr = Vr[(Vr @ -u) >= (Vr @ -u).max() - 0.005].mean(axis=0)
    pinch = (fl + fr) / 2.0
    loc = gmat.T @ (pinch - gpos)
    nlv, nrv = plane_normal(Vl, fl), plane_normal(Vr, fr)
    ang = float(np.degrees(np.arccos(np.clip(abs(nlv @ nrv), -1, 1))))
    d2 = ((Vl[:, None, :] - Vr[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    allv = np.vstack([geom_vertices_world(sim, g_) for g_ in ggids]) if ggids else np.zeros((1, 3))
    proj = (allv - gpos) @ gmat[:, 2]
    return {
        "grip_site_world_pos": gpos.tolist(),
        "grasp_point_world_pos": pinch.tolist(),
        "grasp_point_local_m": loc.tolist(),
        "grasp_point_offset_along_approach_mm": float(loc[2] * 1000.0),
        "grasp_point_offset_lateral_mm": float(np.linalg.norm(loc[:2]) * 1000.0),
        "pad_face_separation_mm": float(np.linalg.norm(fl - fr) * 1000.0),
        "pad_face_normal_angle_deg": ang,
        "jaw_opening_pad_center_m": float(np.linalg.norm(cl - cr)),
        "nip_clearance_m": float(np.linalg.norm(Vl[i] - Vr[j])),
        "gripper_extent_plus_z_mm": float(proj.max() * 1000.0),
        "gripper_extent_minus_z_mm": float(proj.min() * 1000.0),
        "T_cam2grip": (np.linalg.inv(T_cam) @ T_grip).tolist(),
        "dist_cam_to_gripsite_mm": float(np.linalg.norm(cpos - gpos) * 1000.0),
        "constraint_residuals": constraint_residuals(sim),
        "gripper_joint_qpos": {
            n: float(sim.data._data.qpos[int(sim.model._model.jnt_qposadr[
                sim.model.joint_name2id(n)])]) for n in gjoints},
    }


# --------------------------------------------------------------------------- #
def polarity_probe():
    """走真实 env.step 路径：-1 开 → +1 合 → -1 再开，验证极性与部署速度。"""
    out = {"speed_asset_default": None, "speed_with_env_0p01": None, "sequence": {}}
    env, _, _ = build_env("ECO65", "ZhixingGripper")
    for speed in (None, 0.01):
        if speed is None:
            os.environ.pop("ZHIXING_GRIPPER_SPEED", None)
        else:
            os.environ["ZHIXING_GRIPPER_SPEED"] = str(speed)
        from robosuite.models.grippers.zhixing_gripper import ZhixingGripper
        out["speed_asset_default" if speed is None else "speed_with_env_0p01"] = \
            float(ZhixingGripper().speed)

        env.reset()
        rb = env.env.robots[0]
        sim = env.env.sim
        gl, gr = pad_geom_ids(sim, "ZhixingGripper")

        def opening():
            return float(np.linalg.norm(geom_vertices_world(sim, gl).mean(axis=0)
                                        - geom_vertices_world(sim, gr).mean(axis=0)))

        seq = {"opening_at_reset_m": opening(), "steps_per_phase": 250, "phases": []}
        for u in (-1.0, +1.0, -1.0):
            for _ in range(250):
                env.step(np.concatenate([np.zeros(6), np.array([u])]))
            seq["phases"].append({
                "action": u, "opening_after_m": opening(),
                "gripper_current_action": np.array(rb.gripper.current_action).tolist(),
                "gripper_ctrl": float(sim.data._data.ctrl[
                    [i for i in range(sim.model._model.nu)
                     if (mujoco_name(sim, i) or "").startswith("gripper0_")][0]]),
            })
        out["sequence"]["speed_%s" % ("default_0.2" if speed is None else "0.01")] = seq
    os.environ.pop("ZHIXING_GRIPPER_SPEED", None)
    env.close()

    d = out["sequence"]["speed_0.01"]["phases"]
    out["verdict"] = {
        "minus1_opens": bool(d[0]["opening_after_m"] > out["sequence"]["speed_0.01"]
                             ["opening_at_reset_m"]),
        "plus1_closes_from_open": bool(d[1]["opening_after_m"] < d[0]["opening_after_m"]),
        "minus1_reopens": bool(d[2]["opening_after_m"] > d[1]["opening_after_m"]),
        "convention": "-1 = open, +1 = close",
        "open_stroke_span_m": float(d[0]["opening_after_m"] - d[1]["opening_after_m"]),
    }
    return out


def mujoco_name(sim, i):
    import mujoco

    return mujoco.mj_id2name(sim.model._model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--skip-polarity", action="store_true")
    a = ap.parse_args()

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import mujoco

    env0, suite0, _ = build_env("Panda", "PandaGripper", a.task_id, a.seed)
    env0.reset()
    env0.set_init_state(suite0.get_task_init_states(a.task_id)[a.init_idx])
    sim0 = env0.env.sim
    poses = {}
    for i in range(sim0.model._model.njnt):
        n = mujoco.mj_id2name(sim0.model._model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if n.startswith("robot0_") or n.startswith("gripper0_"):
            continue
        poses[n] = np.array(sim0.data.get_joint_qpos(n), dtype=float).tolist()
    env0.close()
    print(f"[setup] 冻结物体关节 {len(poses)} 个 (task_id={a.task_id}, init_idx={a.init_idx})")

    configs, errors = [], {}
    for robot, gripper, tag in CONFIGS:
        print(f"[measure] {tag}: {robot} + {gripper}")
        try:
            configs.append(measure_config(robot, gripper, tag, poses, out_dir))
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors[tag] = f"{type(e).__name__}: {e}"

    pol = None
    if not a.skip_polarity:
        try:
            pol = polarity_probe()
            print(f"[polarity] {json.dumps(pol['verdict'], ensure_ascii=False)}")
            print(f"[speed] asset_default={pol['speed_asset_default']} "
                  f"env_0.01={pol['speed_with_env_0p01']}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors["polarity"] = f"{type(e).__name__}: {e}"

    by = {c["tag"]: c for c in configs}
    cmp_out = {}
    if "panda_panda" in by and "eco65_zhixing" in by:
        z, p = by["eco65_zhixing"], by["panda_panda"]
        for s in STATES:
            a_ = np.array(z["geometry"][s]["T_cam2grip"])
            b_ = np.array(p["geometry"][s]["T_cam2grip"])
            R = a_[:3, :3].T @ b_[:3, :3]
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
            cmp_out[f"zhixing_vs_pandagripper_{s}"] = {
                "T_cam2grip_trans_delta_mm": float(np.linalg.norm(a_[:3, 3] - b_[:3, 3]) * 1000),
                "T_cam2grip_rot_delta_deg": float(ang),
                "delta_vector_mm": (a_[:3, 3] - b_[:3, 3]).tolist(),
                "grasp_along_delta_mm": float(
                    z["geometry"][s]["grasp_point_offset_along_approach_mm"]
                    - p["geometry"][s]["grasp_point_offset_along_approach_mm"])}
        for s in STATES:
            fz = z["views"][f"wrist_{s}"]["gripper_pixel_fraction"]
            fp = p["views"][f"wrist_{s}"]["gripper_pixel_fraction"]
            cmp_out[f"wrist_frac_{s}"] = {"zhixing": fz, "panda": fp,
                                          "ratio": (fz / fp) if fp else None}
    if "eco65_panda" in by and "panda_panda" in by:
        a_ = np.array(by["eco65_panda"]["camera"]["T_cam2grip_reset"])
        b_ = np.array(by["panda_panda"]["camera"]["T_cam2grip_reset"])
        cmp_out["eco65_panda_vs_panda_panda_T_cam2grip_trans_delta_mm"] = float(
            np.linalg.norm(a_[:3, 3] - b_[:3, 3]) * 1000)

    blob = {
        "generated_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "purpose": "N2-v2 当前资产下的 TCP/相机/开合/初始位姿重测（自洽装配 primary + 动力学驱动 cross-check）",
        "task_suite": TASK_SUITE, "task_id": a.task_id, "init_idx": a.init_idx,
        "seed": a.seed, "resolution": RES, "settle_steps": SETTLE_STEPS,
        "converged_tolerance": CONVERGED_TOL,
        "gripper_ctrl_targets": GRIP_CTRL,
        "configs": configs, "comparison": cmp_out, "polarity": pol, "errors": errors,
    }
    (out_dir / "n2v2_settled_probe.json").write_text(
        json.dumps(blob, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {out_dir / 'n2v2_settled_probe.json'}")
    print(json.dumps(cmp_out, indent=2, ensure_ascii=False))
    if errors:
        print(json.dumps(errors, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
