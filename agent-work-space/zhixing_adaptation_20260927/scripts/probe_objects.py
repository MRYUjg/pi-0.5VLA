#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3 前置探测：列出 LIBERO 任务里可抓取物体的位姿与尺寸，
以便选出与 ZhiXing 开口（内面净距 ~90 mm）匹配的抓取目标。
"""
import pathlib
import sys

import numpy as np

RES = 256
TASK_SUITE = "libero_spatial"


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
    return env, suite, task


def main():
    import mujoco

    env, suite, task = build("ECO65", "ZhixingGripper", 0)
    # 注意：不能对 ZhiXing 用 set_init_state —— LIBERO 预计算的 init state 是
    # Panda 时代的 40 维 qvel，装上 ZhiXing（6 关节夹爪）后是 46 维，会广播失败。
    # 这与 `夹爪对照实验结论.md` 记录的方法学陷阱是同一个。
    env.reset()
    sim = env.env.sim
    m, d = sim.model._model, sim.data._data

    print(f"task language: {task.language}\n")
    print(f"{'body':<34s} {'ngeom':>5s} {'pos(m)':<28s} {'bbox size(mm)':<24s}")
    print("-" * 100)

    skip = ("robot0", "gripper0", "world", "table", "mount", "fixed")
    rows = []
    for b in range(m.nbody):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not name or any(s in name for s in skip):
            continue
        # 该 body 下所有 geom 的世界包围盒
        gids = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == b]
        if not gids:
            continue
        pts = []
        for g in gids:
            lo, hi = [], []
            gt = m.geom_type[g]
            if gt == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                v = np.array(m.mesh_vert[a:a + n], dtype=float)
            else:
                s = np.array(m.geom_size[g], dtype=float)
                v = s * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
            pts.append((R @ v.T).T + np.array(d.geom_xpos[g], dtype=float))
        P = np.vstack(pts)
        size = (P.max(axis=0) - P.min(axis=0)) * 1000
        ctr = P.mean(axis=0)
        rows.append((name, len(gids), ctr, size))

    for name, ng, ctr, size in sorted(rows, key=lambda r: -max(r[3])):
        print(f"{name:<34s} {ng:>5d} {np.round(ctr,4).tolist()!s:<28s} "
              f"{np.round(size,1).tolist()!s:<24s}")

    print("\n=== free joints（可摆放的物体关节）===")
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if n.startswith("robot0"):
                continue
            a = int(m.jnt_qposadr[j])
            print(f"  {n:<34s} qpos={np.round(d.qpos[a:a+7],4).tolist()}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
