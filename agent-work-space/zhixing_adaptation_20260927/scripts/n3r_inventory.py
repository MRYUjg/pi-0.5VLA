#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3-redo 前置：遍历 libero_spatial 全部 10 个任务，列出**碰撞几何**尺寸，
寻找「闭合轴宽度 < 90 mm 开口、高度 >= 45 mm」的可抓取标准目标。

动机
----
已有 N3 用任务物体（薄壁碗 111.8 mm）测夹爪，把「物体模型缺陷 / 摆放不稳 /
薄壁力学 / 夹爪接触模型」四个变量混在一起，无法归因。
按 N3 §5.1：应先用**已知几何的标准件**标定夹爪。

真机约束（已核实）：ChangingTek CTAG2F90「90 Collaborative Two Finger Hands」，
Maximum travel distance = **90 mm**——与仿真模型的 89.96 mm 内面净距一致，
故 111.8 mm 的碗在真机上同样抓不了，属硬件行程限制而非仿真 bug。

本脚本只做**只读探测**，不改任何资产。
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

SUITE = "libero_spatial"
OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"
OPENING_MM = 89.96          # N2 权威口径：全开内面净距
FINGERTIP_BEYOND_PINCH_MM = 19.9

SKIP = ("robot0", "gripper0", "mount", "table", "main_table", "world")


def build(task_id, robot="ECO65", gripper="ZhixingGripper", seed=7, res=128):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=res, camera_widths=res,
        robots=[robot], gripper_types=gripper)
    env.seed(seed)
    return env, task


def body_dims(sim, bid):
    """该 body 下**碰撞 geom** 的世界包围盒；无碰撞 geom 返回 None。"""
    pts = N.body_world_pts(sim, bid, collision_only=True)
    if pts is None or pts.shape[0] == 0:
        return None
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return lo, hi, (hi - lo) * 1000.0


def main():
    import mujoco

    all_tasks = []
    candidates = []

    for tid in range(10):
        env, task = build(tid)
        env.reset()
        sim, m, d, gid = N.ids(env)

        rows = []
        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if not name or any(s in name for s in SKIP):
                continue
            r = body_dims(sim, b)
            if r is None:
                continue
            lo, hi, size = r
            # 该 body 是否有 free joint（可被抓走）
            has_free = False
            for j in range(m.njnt):
                if int(m.jnt_bodyid[j]) == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                    has_free = True
            rows.append({
                "body": name, "ngeom_col": int(np.sum(
                    [1 for g in range(m.ngeom)
                     if int(m.geom_bodyid[g]) == b and m.geom_contype[g] != 0])),
                "free": bool(has_free),
                "size_mm": [round(float(v), 1) for v in size],
                "z_lo": round(float(lo[2]), 4), "z_hi": round(float(hi[2]), 4),
                "xy": [round(float(0.5 * (lo[0] + hi[0])), 4),
                       round(float(0.5 * (lo[1] + hi[1])), 4)],
            })

        # 可抓候选：可动(free) + 最小水平尺寸 < 开口 + 高度够指尖净空
        for r in rows:
            if not r["free"]:
                continue
            sx, sy, sz = r["size_mm"]
            min_h = min(sx, sy)
            # 抓取时抓持点需高于桌面 >=19.9 mm，故物体高度需 > ~45 mm
            ok_w = min_h < OPENING_MM - 5.0
            ok_h = sz >= 45.0
            r["candidate"] = bool(ok_w and ok_h)
            if r["candidate"]:
                candidates.append({"task_id": tid, "language": task.language, **r})

        all_tasks.append({"task_id": tid, "language": task.language,
                          "objects": sorted(rows, key=lambda x: -max(x["size_mm"]))})
        print(f"[task {tid}] {task.language}")
        for r in sorted(rows, key=lambda x: -max(x["size_mm"])):
            mark = "  <== 候选" if r.get("candidate") else ""
            print(f"    {r['body']:<38s} free={int(r['free'])} "
                  f"size={r['size_mm']} z=[{r['z_lo']},{r['z_hi']}] xy={r['xy']}{mark}")
        env.close()

    print("\n" + "=" * 78)
    print(f"可抓候选（最小水平尺寸 < {OPENING_MM - 5:.1f} mm 且高度 >= 45 mm）：")
    for c in candidates:
        print(f"  task{c['task_id']:>2d}  {c['body']:<36s} size={c['size_mm']}  {c['language'][:40]}")
    if not candidates:
        print("  （无）—— 需要注入标准试块或改用其他目标")

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    (pathlib.Path(OUT) / "n3r_inventory.json").write_text(
        json.dumps({"opening_mm": OPENING_MM, "tasks": all_tasks,
                    "candidates": candidates}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\n[out] {OUT}/n3r_inventory.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
