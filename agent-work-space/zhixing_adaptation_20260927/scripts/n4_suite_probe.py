#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4 前置探测：哪些 LIBERO suite 能在 ECO65 + ZhiXing 下**构建成功**，
以及各 suite 的抓取目标物体是否落在 ZhiXing 的 90 mm 行程 / ≥45 mm 高度包络内。

动机
----
用户选定用 `libero_object` 替代 `libero_spatial`（后者 10 个任务的抓取目标
都是 107 mm 黑碗，超出真机 90 mm 行程）。但实测发现 suite 的 arena 类型不同，
机器人挂载方式也不同：

| suite | problem 类 | 机器人前缀 | 现有类 |
|---|---|---|---|
| libero_spatial | Tabletop | `Mounted`   | MountedECO65 ✓ |
| libero_goal    | Tabletop | `Mounted`   | MountedECO65 ✓ |
| libero_10      | Kitchen Tabletop | `Mounted` | MountedECO65（含 kitchen_table 键）✓ |
| libero_object  | **Floor** | **`OnTheGround`** | **缺失** ✗ |

`Libero_Floor_Manipulation` 会把机器人名加前缀 `OnTheGround`，而仓库只有
`OnTheGroundPanda`——故 `libero_object` 当前**无法构建**。

本脚本对**能构建**的 suite 做只读探测：列出可动目标的碰撞几何尺寸与
逐任务抓取可行性（复用 `n3_grasp_test.grasp_plan` 的口径）。
"""
import json
import pathlib
import sys
import traceback

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"
OPENING_MM = 89.96
MIN_HEIGHT_MM = 45.0
SKIP = ("robot0", "gripper0", "mount", "table", "main_table", "world",
        "bin", "basket", "floor", "wall")

SUITES = ["libero_object", "libero_goal", "libero_10", "libero_spatial"]


def build(suite, task_id, robot="ECO65", gripper="ZhixingGripper", seed=7, res=128):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    s = benchmark.get_benchmark_dict()[suite]()
    task = s.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=res,
                             camera_widths=res, robots=[robot],
                             gripper_types=gripper)
    env.seed(seed)
    return env, task


def movable_objects(sim):
    """返回 [(bid, name)]：带 free joint（可被抓走）的物体 body。"""
    import mujoco

    m = sim.model._model
    out = []
    for b in range(m.nbody):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not n or any(s in n for s in SKIP):
            continue
        for j in range(m.njnt):
            if int(m.jnt_bodyid[j]) == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                out.append((b, n))
                break
    return out


def main():
    summary = {"opening_mm": OPENING_MM, "min_height_mm": MIN_HEIGHT_MM,
               "suites": {}}

    for suite in SUITES:
        print("\n" + "=" * 78)
        print("[suite] %s" % suite)
        entry = {"build_ok": False}
        try:
            env, task = build(suite, 0)
        except Exception as e:                              # noqa: BLE001
            entry["build_error"] = "%s: %s" % (type(e).__name__, e)
            print("  ✗ 构建失败：%s" % entry["build_error"])
            print("    %s" % traceback.format_exc().strip().splitlines()[-1])
            summary["suites"][suite] = entry
            continue

        entry["build_ok"] = True
        entry["task0_language"] = task.language
        env.reset()
        sim, m, d, gid = N.ids(env)
        entry["n_movable"] = len(movable_objects(sim))
        print("  ✓ 构建成功  task0: %s" % task.language)
        print("  可动物体数 = %d" % entry["n_movable"])

        tasks = []
        for tid in range(10):
            try:
                if tid > 0:
                    env.close()
                    env, task = build(suite, tid)
                    env.reset()
                    sim, m, d, gid = N.ids(env)
                objs = movable_objects(sim)
                rows = []
                for bid, name in objs:
                    pts = N.body_world_pts(sim, bid, collision_only=True)
                    if pts is None or pts.shape[0] == 0:
                        continue
                    lo, hi = pts.min(axis=0), pts.max(axis=0)
                    size = (hi - lo) * 1000.0
                    lo_xy = float(min(size[0], size[1]))
                    rows.append({
                        "body": name,
                        "size_mm": [round(float(v), 1) for v in size],
                        "min_horizontal_mm": round(lo_xy, 1),
                        "height_mm": round(float(size[2]), 1),
                        "graspable": bool(lo_xy <= OPENING_MM - 5.0
                                          and size[2] >= MIN_HEIGHT_MM),
                    })
                # 该任务的抓取目标 = 语言里提到的可动物体（粗略：全部列出）
                t = {"task_id": tid, "language": task.language, "objects": rows}
                tasks.append(t)
                g = [r for r in rows if r["graspable"]]
                print("  [task %d] %s" % (tid, task.language[:62]))
                for r in sorted(rows, key=lambda x: -x["min_horizontal_mm"]):
                    mark = " <== 可抓" if r["graspable"] else ""
                    print("      %-34s %s%s" % (
                        r["body"][:34], r["size_mm"], mark))
                if not g:
                    print("      → 该任务无可抓物体")
            except Exception as e:                          # noqa: BLE001
                tasks.append({"task_id": tid, "error": str(e)})
                print("  [task %d] ✗ %s" % (tid, e))
        entry["tasks"] = tasks
        entry["n_tasks_with_graspable"] = sum(
            1 for t in tasks if any(o.get("graspable") for o in t.get("objects", [])))
        print("\n  → %s：%d/10 个任务存在可抓物体"
              % (suite, entry["n_tasks_with_graspable"]))
        summary["suites"][suite] = entry
        try:
            env.close()
        except Exception:                                   # noqa: BLE001
            pass

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(OUT) / "n4_suite_probe.json"
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 78)
    print("[summary]")
    for k, v in summary["suites"].items():
        if v["build_ok"]:
            print("  %-15s 构建 ✓   可抓物体任务 %d/10" % (k, v["n_tasks_with_graspable"]))
        else:
            print("  %-15s 构建 ✗   %s" % (k, v.get("build_error", "")[:60]))
    print("[out] %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
