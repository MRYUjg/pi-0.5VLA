#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2 交叉验证：用 **XML 声明的 site 与 body**（而非 mesh 顶点）独立复核
ZhiXing / Panda 的夹爪几何相对 `grip_site` 的偏移。

动机：`left_fingerpad_col` 用的是整条 `Left_Support_Link` 网格，其「最近点对」
可能落在四连杆铰接处而不是指垫接触面。因此必须用无歧义的 site 独立复核，
避免把「连杆最近点」误当成「夹持点」。

输出：每个配置的 grip_site 局部系下，所有夹爪 site / body 的偏移表。
"""
import json
import pathlib
import sys

import numpy as np

RES = 256
TASK_SUITE = "libero_spatial"
DEFAULT_OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"

CONFIGS = [
    ("ECO65", "ZhixingGripper", "eco65_zhixing"),
    ("ECO65", "PandaGripper", "eco65_panda"),
    ("Panda", "PandaGripper", "panda_panda"),
]


def build_env(robot, gripper, task_id=0, seed=7):
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
    return env, suite, task


def main():
    import mujoco

    out = {}
    for robot, gripper, tag in CONFIGS:
        env, suite, _ = build_env(robot, gripper)
        env.reset()
        sim = env.env.sim
        m = sim.model._model
        d = sim.data._data

        gid = None
        for i in range(m.nsite):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or ""
            if n.endswith("grip_site"):
                gid = i
                break
        gpos = np.array(d.site_xpos[gid], dtype=float)
        gmat = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)

        def loc(p):
            return (gmat.T @ (np.asarray(p, dtype=float) - gpos)) * 1000.0  # mm

        rec = {"tag": tag, "robot": robot, "gripper": gripper,
               "grip_site_world_pos": gpos.tolist(), "sites": [], "bodies": []}

        # 全部 site（只保留夹爪子树内的）
        for i in range(m.nsite):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or ""
            bid = int(m.site_bodyid[i])
            chain, guard = [], 0
            b = bid
            while b != 0 and guard < 64:
                chain.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                b = int(m.body_parentid[b])
                guard += 1
            if not any("right_gripper" in c or "finger" in c or "Link" in c for c in chain):
                continue
            v = loc(d.site_xpos[i])
            rec["sites"].append({"name": n, "body": chain[0] if chain else "",
                                 "local_mm": v.tolist(), "norm_mm": float(np.linalg.norm(v))})

        # 全部 body（夹爪子树）
        for i in range(m.nbody):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            chain, guard, b = [], 0, i
            while b != 0 and guard < 64:
                chain.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
                b = int(m.body_parentid[b])
                guard += 1
            if not any("right_gripper" in c for c in chain):
                continue
            v = loc(d.xpos[i])
            rec["bodies"].append({"name": n, "local_mm": v.tolist(),
                                  "norm_mm": float(np.linalg.norm(v))})

        # 指垫碰撞 geom 的最近点对（保留下方，用于和 site 对照）
        pads = []
        for i in range(m.ngeom):
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if "fingerpad" in n or "pad_collision" in n:
                pads.append((n, i))
        if len(pads) == 2:
            def verts(g_):
                pos = np.array(d.geom_xpos[g_], dtype=float)
                mat = np.array(d.geom_xmat[g_], dtype=float).reshape(3, 3)
                if m.geom_type[g_] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = m.geom_dataid[g_]
                    a, nn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                    vv = np.array(m.mesh_vert[a:a + nn], dtype=float)
                else:
                    s = np.array(m.geom_size[g_], dtype=float)
                    vv = s * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
                return (mat @ vv.T).T + pos

            vl, vr = verts(pads[0][1]), verts(pads[1][1])
            d2 = ((vl[:, None, :] - vr[None, :, :]) ** 2).sum(-1)
            i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
            rec["pad_closest_pair_local_mm"] = {
                "left": loc(vl[i]).tolist(), "right": loc(vr[j]).tolist(),
                "mid": loc((vl[i] + vr[j]) / 2).tolist(),
                "clearance_mm": float(np.linalg.norm(vl[i] - vr[j]) * 1000),
            }

        out[tag] = rec
        env.close()

        print(f"\n===== {tag} ({robot} + {gripper}) =====")
        print("  sites (local offset from grip_site, mm):")
        for s in sorted(rec["sites"], key=lambda x: -x["local_mm"][2]):
            print(f"    {s['name']:<34s} x={s['local_mm'][0]:8.2f} y={s['local_mm'][1]:8.2f} z={s['local_mm'][2]:8.2f}  |.|={s['norm_mm']:8.2f}")
        print("  bodies (local offset from grip_site, mm):")
        for s in sorted(rec["bodies"], key=lambda x: -x["local_mm"][2]):
            print(f"    {s['name']:<34s} x={s['local_mm'][0]:8.2f} y={s['local_mm'][1]:8.2f} z={s['local_mm'][2]:8.2f}  |.|={s['norm_mm']:8.2f}")
        if "pad_closest_pair_local_mm" in rec:
            p = rec["pad_closest_pair_local_mm"]
            print(f"  pad closest pair: mid_local={np.round(p['mid'],2).tolist()} clearance={p['clearance_mm']:.3f} mm")

    out_dir = pathlib.Path(DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tcp_geometry_sites.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[out] {out_dir / 'tcp_geometry_sites.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
