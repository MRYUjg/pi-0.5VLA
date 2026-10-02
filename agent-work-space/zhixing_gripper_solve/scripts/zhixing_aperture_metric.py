#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZhiXing 开口口径核对（只读）：把**所有**手指碰撞 geom 的几何量都打出来，
用真实 mesh 顶点算最近点对，回答「合到底时指垫到底还剩多宽」。

为什么需要：`jaw_gap_mm` 之前对 mesh 类型的指垫退化成"几何中心距"，
而 `left_fingerpad_col` 是 **Support_Link 的 mesh**（不是指尖夹持面），
所以 18 mm 那个数字的口径需要重核 —— 本脚本把每一种口径都算出来对照。
"""
import argparse
import datetime
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N              # noqa: E402
import n4_collect as N4                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_object")
ap.add_argument("--task-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--out", default="/app/agent-work-space/outputs/diag/zhixing_aperture_metric.json")
a = ap.parse_args()
os.environ.setdefault("ZHIXING_GRIPPER_SPEED", "0.01")
RES = {"node": "ZHIXING-APERTURE-METRIC",
       "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
       "note": "只读：未修改任何代码/参数"}


def log(m):
    print(m, flush=True)


def step_raw(env, cmd):
    try:
        env.env.done = False
    except Exception:                                          # noqa: BLE001
        pass
    return env.step(np.concatenate([np.zeros(6), [cmd]]))


def geoms(env):
    """列出所有与手指/夹爪相关的碰撞 geom（名字、类型、尺寸、所属 body）。"""
    import mujoco
    sim, m, d, _ = N.ids(env)
    out = []
    for g in range(m.ngeom):
        nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        bn = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                int(m.geom_bodyid[g])) or "").lower()
        if not ("gripper0" in bn or "gripper0" in nm):
            continue
        out.append({
            "id": g, "geom": nm, "body": bn,
            "type": int(m.geom_type[g]),
            "contype": int(m.geom_contype[g]), "conaffinity": int(m.geom_conaffinity[g]),
            "size": [float(v) for v in m.geom_size[g][:3]],
            "pos_world": [float(v) for v in d.geom_xpos[g]],
            "is_collision": bool(int(m.geom_contype[g]) > 0 or int(m.geom_conaffinity[g]) > 0),
        })
    return out


def mesh_points(m, d, g):
    """geom 表面点：mesh → 真实顶点；sphere/box/capsule → 采样。"""
    import mujoco
    ctr = np.array(d.geom_xpos[g], dtype=float)
    R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
    t = int(m.geom_type[g])
    sz = [float(v) for v in m.geom_size[g][:3]]
    if t == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(m.geom_dataid[g])
        if mid < 0:
            return ctr.reshape(1, 3)
        vadr, vnum = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        v = np.array(m.mesh_vert[vadr:vadr + vnum], dtype=float)
        return ctr + v @ R.T
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        u = np.linspace(0, np.pi, 7)
        w = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        loc = np.array([[np.sin(p) * np.cos(q), np.sin(p) * np.sin(q), np.cos(p)]
                        for p in u for q in w]) * sz[0]
        return ctr + loc @ R.T
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        g1 = np.linspace(-1, 1, 9)
        loc = np.array([[x, y, z] for x in g1 for y in g1 for z in g1])
        return ctr + (loc * np.array(sz[:3])) @ R.T
    return ctr.reshape(1, 3)


def pair_distance_mm(env, ids_a, ids_b):
    """两组 geom 之间的最小表面距离（mm）+ 最近点对。"""
    import mujoco
    sim, m, d, _ = N.ids(env)
    best, pair = None, None
    for ga in ids_a:
        pa = mesh_points(m, d, ga)
        for gb in ids_b:
            pb = mesh_points(m, d, gb)
            dd = ((pa[:, None, :] - pb[None, :, :]) ** 2).sum(axis=2)
            k = int(np.argmin(dd))
            dist = float(np.sqrt(dd.flat[k]))
            if best is None or dist < best:
                best = dist
                ia, ib = divmod(k, pb.shape[0])
                pair = {"a": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, ga) or str(ga),
                        "b": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gb) or str(gb),
                        "point_a": [float(v) for v in pa[ia]], "point_b": [float(v) for v in pb[ib]],
                        "n_pts_a": int(pa.shape[0]), "n_pts_b": int(pb.shape[0])}
    return (None if best is None else round(best * 1000, 4)), pair


def measure(env, tag):
    import mujoco
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    idx = [int(i) for i in r._ref_gripper_joint_pos_indexes]
    joints = [str(j) for j in r.gripper.joints]
    q = [float(d.qpos[i]) for i in idx]
    gl = geoms(env)
    # 按 body 名分组（左/右）
    left = [g["id"] for g in gl if g["is_collision"] and
            ("left" in g["body"] or "finger1" in g["geom"] or "_1_" in g["geom"])]
    right = [g["id"] for g in gl if g["is_collision"] and
             ("right" in g["body"] or "finger2" in g["geom"] or "_2_" in g["geom"])]
    d_all, pair_all = pair_distance_mm(env, left, right) if (left and right) else (None, None)
    # 只取指尖小 box（碰撞、且 size 有 3 个分量的 box）
    def tip_boxes(ids):
        out = []
        for g in ids:
            if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                out.append(g)
        return out
    lt, rt = tip_boxes(left), tip_boxes(right)
    d_tip, pair_tip = pair_distance_mm(env, lt, rt) if (lt and rt) else (None, None)
    named_pads = [g["id"] for g in gl if "pad" in g["geom"]]
    return {
        "tag": tag,
        "Left_1": next((q[j] for j, nm in enumerate(joints) if nm.endswith("Left_1_Joint")), None),
        "gripper_qpos": [round(v, 6) for v in q],
        "aperture_all_finger_collision_mm": d_all,
        "closest_pair_all": pair_all,
        "aperture_tip_boxes_mm": d_tip,
        "closest_pair_tip_boxes": pair_tip,
        "n_left_collision_geoms": len(left), "n_right_collision_geoms": len(right),
        "n_tip_boxes_left": len(lt), "n_tip_boxes_right": len(rt),
        "geoms_named_pad": [g["geom"] for g in gl if g["id"] in named_pads],
        "all_gripper_geoms": gl,
    }


def main():
    env, task = N4.build(a.suite, a.task_id, robot="ECO65", gripper="ZhixingGripper", seed=a.seed)
    import mujoco
    sim, m, d, gid = N.ids(env)
    RES["ngeom_total"] = int(m.ngeom)
    gl = geoms(env)
    RES["geom_inventory"] = gl
    log("夹爪相关 geom 共 %d 个：" % len(gl))
    for g in gl:
        log("  id=%-3d %-32s body=%-26s type=%d contype=%d size=%s"
            % (g["id"], g["geom"], g["body"], g["type"], g["contype"],
               [round(v, 5) for v in g["size"]]))

    for tag, act in (("reset", None), ("open_cmd_-1", -1.0), ("close_cmd_+1", 1.0)):
        env.reset()
        if act is not None:
            for _ in range(500):
                step_raw(env, act)
        RES[tag] = measure(env, tag)
        r = RES[tag]
        log("\n[%s] Left_1=%+.4f  q=%s" % (tag, r["Left_1"], r["gripper_qpos"]))
        log("   全手指碰撞 geom 最近点对 = %s mm  pair=%s"
            % (r["aperture_all_finger_collision_mm"],
               None if not r["closest_pair_all"] else (r["closest_pair_all"]["a"], r["closest_pair_all"]["b"])))
        log("   指尖 box 最近点对        = %s mm  pair=%s"
            % (r["aperture_tip_boxes_mm"],
               None if not r["closest_pair_tip_boxes"] else (r["closest_pair_tip_boxes"]["a"],
                                                             r["closest_pair_tip_boxes"]["b"])))

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RES, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log("\n[out] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
