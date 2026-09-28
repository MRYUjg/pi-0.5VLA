#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""给抓取可视化挑一个好机位：同一状态下渲染 agentview + 多个自由相机方位角。"""
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N      # noqa: E402
import n3r_block_test as B     # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results/grasp_viz/camtest"


def main():
    import mujoco
    import PIL.Image

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    B.install_block_injection((0.030, 0.025, 0.035), (-0.22, 0.10, 1.30))
    env, _ = N.build(task_id=0)
    env.reset()
    for _ in range(400):
        env.step(np.concatenate([np.zeros(6), [0.0]]))
    sim, m, d, gid = N.ids(env)
    bid = B.block_id(sim)
    pts = N.body_world_pts(sim, bid, collision_only=True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = 0.5 * (lo + hi)
    print("[block] center=%s  (nearby scene objects may occlude)" % np.round(c, 3).tolist())

    # 闭合到停住，让特写看到真实抓取姿态
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    aadr = N._arm_idx(env)[5]
    q_home = np.array([d.qpos[x] for x in aadr], dtype=float)
    pin = np.array([float(c[0]), float(c[1]), float(lo[2]) + 0.035])
    q_pre, _ = N.ik_goto(env, q_home, pin + np.array([0, 0, 0.06]), R_ref, 0.0, settle=15)
    q_g, _ = N.ik_goto(env, q_pre, pin, R_ref, 0.0, settle=15)
    N.step_hold(env, q_g, -0.91, n=3300)
    print("[close] jaw=%.2f mm" % N.jaw_face_separation_mm(env))

    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), 640)
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), 480)
    rnd = mujoco.Renderer(m, height=480, width=640)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "block" in nm and nm.endswith("_geom"):
            m.geom_rgba[g] = [0.85, 0.22, 0.18, 1.0]
        elif int(m.geom_group[g]) == 0:
            m.geom_rgba[g] = [0.70, 0.71, 0.73, 1.0]

    # 1) 内置 agentview
    for camname in ("agentview", "frontview", "birdview"):
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camname)
        if cid < 0:
            print("  (无内置相机 %s)" % camname)
            continue
        rnd.update_scene(d, camera=camname, scene_option=opt)
        PIL.Image.fromarray(rnd.render()).save(pathlib.Path(OUT) / ("builtin_%s.png" % camname))
        print("[img] builtin_%s.png" % camname)

    # 2) 自由相机：绕一圈
    for azim in (0, 45, 90, 135, 180, 225, 270, 315):  # 8 个方位挑一个能看清夹爪开合的
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = c
        cam.distance = 0.30
        cam.azimuth = azim
        cam.elevation = 12
        rnd.update_scene(d, camera=cam, scene_option=opt)
        PIL.Image.fromarray(rnd.render()).save(pathlib.Path(OUT) / ("free_az%03d.png" % azim))
        print("[img] free_az%03d.png" % azim)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
