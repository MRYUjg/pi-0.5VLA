#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3-redo 诊断：量化 ZhiXing 指垫**碰撞体**的缺陷，并给出与 Panda 同构的专用 pad box 参数。

根因线索（2026-09-27 核对 XML）
-------------------------------
Panda 的夹爪每个手指有**两个**碰撞 geom：
  - `fingerN_collision`  : type=mesh, contype=0, conaffinity=1, solref="0.02 1",
                           friction="1 0.005 0.0001"   ← 手指本体
  - `fingerN_pad_collision`: type=**box** size="0.008 0.004 0.008",
                           solref="**0.01 0.5**", friction="**2 0.05 0.0001**",
                           contype=**1**                  ← 专用橡胶垫（硬、10× 抗扭摩擦）

ZhiXing **只有** `left/right_fingerpad_col`：type=mesh（整条 Support_Link 凸包），
且沿用**手指本体**的软参数 solref="0.02 1"、friction="1 0.005 0.0001"、contype=0。
→ 既没有独立橡胶垫，接触也软 10 倍。实测后果：闭合到 50 mm 试块时穿透 **9.26 mm**、
试块被推 6.6 mm、抬升仅 1.8 mm。

本脚本只做**只读测量 + 参数建议**，不改任何资产。
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"


def pad_geom_ids(sim):
    import mujoco

    m = sim.model._model
    out = {}
    for g in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if "left_fingerpad_col" in n:
            out["left"] = g
        elif "right_fingerpad_col" in n:
            out["right"] = g
    return out


def geom_attr(m, g):
    import mujoco

    return {
        "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
        "type": int(m.geom_type[g]),
        "contype": int(m.geom_contype[g]),
        "conaffinity": int(m.geom_conaffinity[g]),
        "condim": int(m.geom_condim[g]),
        "solref": [float(v) for v in m.geom_solref[g]],
        "solimp": [float(v) for v in m.geom_solimp[g]],
        "friction": [float(v) for v in m.geom_friction[g]],
        "size": [float(v) for v in m.geom_size[g]],
    }


def mesh_pts_body_local(sim, g, bid):
    """该 geom 的 mesh 顶点，换算到其所属 body 的局部坐标系。"""
    import mujoco

    m, d = sim.model._model, sim.data._data
    gt = m.geom_type[g]
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = np.array(m.mesh_vert[a:a + n], dtype=float)
    else:
        s = np.array(m.geom_size[g], dtype=float)
        v = s * np.array([[sx, sy, sz] for sx in (-1, 1)
                          for sy in (-1, 1) for sz in (-1, 1)])
    Rg = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
    Pw = (Rg @ v.T).T + np.array(d.geom_xpos[g], dtype=float)
    Rb = np.array(d.xmat[bid], dtype=float).reshape(3, 3)
    return (Rb.T @ (Pw - np.array(d.xpos[bid])).T).T


def main():
    import mujoco

    env, task = N.build(task_id=0)
    env.reset()
    sim, m, d, gid = N.ids(env)

    # 开到全开
    from n3_grasp_test import step_hold, _arm_idx
    sm, mm, dd, gg, aj, aadr, adof = _arm_idx(env)
    q = np.array([dd.qpos[x] for x in aadr], dtype=float)
    step_hold(env, q, 0.0, n=800)
    sim, m, d, gid = N.ids(env)

    pads = pad_geom_ids(sim)
    rep = {"task": task.language, "pads": {}}

    for side, g in pads.items():
        bid = int(m.geom_bodyid[g])
        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
        rep["pads"][side] = {
            "geom": geom_attr(m, g),
            "body": bname,
            "geom_xpos_world": [float(v) for v in d.geom_xpos[g]],
            "body_xpos_world": [float(v) for v in d.xpos[bid]],
        }

    # 闭合轴（世界系）：左垫中心 -> 右垫中心
    pl = np.array(d.geom_xpos[pads["left"]], dtype=float)
    pr = np.array(d.geom_xpos[pads["right"]], dtype=float)
    axis_w = pr - pl
    axis_w /= np.linalg.norm(axis_w)

    for side, g in pads.items():
        bid = int(m.geom_bodyid[g])
        V = mesh_pts_body_local(sim, g, bid)
        sgn = 1.0 if side == "left" else -1.0
        Rb = np.array(d.xmat[bid], dtype=float).reshape(3, 3)
        u = sgn * (Rb.T @ axis_w)          # body 局部：指向对方（内侧）
        u /= np.linalg.norm(u)
        proj = V @ u
        pmax = float(proj.max())
        rep["pads"][side].update({
            "n_verts": int(V.shape[0]),
            "proj_min_mm": float(proj.min()) * 1000,
            "proj_max_mm": pmax * 1000,
            "extent_along_closing_axis_mm": float(proj.max() - proj.min()) * 1000,
            "inner_dir_body_local": [float(v) for v in u],
        })

        # 内侧薄层 = 真正与物体接触的候选面
        for band in (0.002, 0.003, 0.005):
            sel = V[proj >= pmax - band]
            if sel.shape[0] >= 3:
                break
        c = sel.mean(axis=0)
        # 面内两个正交基
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(u @ tmp)) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(u, tmp)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(u, e1)
        a1 = (sel - c) @ e1
        a2 = (sel - c) @ e2
        rep["pads"][side].update({
            "face_band_used_mm": band * 1000,
            "face_n_verts": int(sel.shape[0]),
            "face_center_body_local_m": [float(v) for v in c],
            "face_half_extent_e1_mm": float(max(abs(a1.min()), abs(a1.max()))) * 1000,
            "face_half_extent_e2_mm": float(max(abs(a2.min()), abs(a2.max()))) * 1000,
            "face_center_proj_mm": float(c @ u) * 1000,
            "face_plane_offset_from_max_mm": float((c @ u - pmax)) * 1000,
            "e1_body_local": [float(v) for v in e1],
            "e2_body_local": [float(v) for v in e2],
        })

    # 名义开口 vs 碰撞凸包内侧净距（同一状态）
    sep = N.jaw_face_separation_mm(env)
    rep["opening_measured_mm"] = sep
    rep["nominal_opening_mm"] = 89.96

    print(json.dumps(rep, indent=2, ensure_ascii=False))
    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT) / "n3r_pad_collision.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {OUT}/n3r_pad_collision.json")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
