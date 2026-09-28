#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3-redo 关键核对：`GRASP_OFFSET_CLOSED = 82.54 mm` 是否真的对应「两指垫内侧面中点」？

背景
----
`n3_grasp_test.ik_goto()` 的全部瞄准都依赖
    grip_site = pinch_target - R @ GRASP_OFFSET_CLOSED
若该偏移与实际指垫内面不符，专家就会**在物体的错误高度闭合**——
所有抓取测试（含既有 N3）都会因此系统性失败，而现象酷似"夹爪抓不住"。

本脚本在**同一编译模型**上直接测量：指垫内面中点位于 grip_site 局部系的哪个位置，
并与 82.54 / 65.11 mm 对照。
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"


def find_pads(sim):
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


def mesh_pts_world(sim, g):
    import mujoco

    m, d = sim.model._model, sim.data._data
    mid = m.geom_dataid[g]
    a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
    v = np.array(m.mesh_vert[a:a + n], dtype=float)
    R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
    return (R @ v.T).T + np.array(d.geom_xpos[g], dtype=float)


def measure(env, gsel, label):
    sim, m, d, gid = N.ids(env)
    aadr = N._arm_idx(env)[5]
    q_now = np.array([d.qpos[x] for x in aadr], dtype=float)
    N.step_hold(env, q_now, gsel, n=800)
    sim, m, d, gid = N.ids(env)
    p_gs = np.array(d.site_xpos[gid], dtype=float)
    R_gs = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)

    pads = find_pads(sim)
    centroid = {}
    facemid = {}
    for side, g in pads.items():
        P = mesh_pts_world(sim, g)
        centroid[side] = P.mean(axis=0)
        # 内侧面：朝对方最靠前的一薄层
        other = "right" if side == "left" else "left"
        u = (mesh_pts_world(sim, pads[other]).mean(axis=0) - P.mean(axis=0))
        u /= np.linalg.norm(u)
        proj = P @ u
        sel = P[proj >= proj.max() - 0.002]
        facemid[side] = sel.mean(axis=0)

    mid_w = 0.5 * (facemid["left"] + facemid["right"])
    sep = float(np.linalg.norm(facemid["left"] - facemid["right"])) * 1000
    loc = R_gs.T @ (mid_w - p_gs)

    # 直接对内外侧面做「相对 grip_site 的局部 z」统计
    res = {
        "label": label, "ctrl": gsel,
        "grip_site_world": [float(v) for v in p_gs],
        "pad_face_midpoint_world": [float(v) for v in mid_w],
        "face_midpoint_in_gripsite_local_mm": [float(v) * 1000 for v in loc],
        "approach_offset_mm": float(loc[2]) * 1000,
        "inner_face_separation_mm": sep,
        "reference_GRASP_OFFSET_CLOSED_mm": 82.54,
        "reference_GRASP_OFFSET_OPEN_mm": 65.11,
    }
    print(f"\n[{label}] ctrl={gsel}")
    print(f"   grip_site world          = {np.round(p_gs,4).tolist()}")
    print(f"   两指垫内面中点 world      = {np.round(mid_w,4).tolist()}")
    print(f"   → 在 grip_site 局部系(mm) = {np.round(loc*1000,2).tolist()}")
    print(f"   → 沿接近轴偏移 = {loc[2]*1000:+.2f} mm "
          f"(参照值: 闭合 82.54 / 全开 65.11)")
    print(f"   内面净距 = {sep:.2f} mm")
    return res


def main():
    env, task = N.build(task_id=0)
    env.reset()
    out = {"task": task.language, "measurements": []}
    for ctrl, label in ((0.0, "open"), (-0.91, "closed")):
        out["measurements"].append(measure(env, ctrl, label))
    env.close()

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT) / "n3r_pinch_offset.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[out] {OUT}/n3r_pinch_offset.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
