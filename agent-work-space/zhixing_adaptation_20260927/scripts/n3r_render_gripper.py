#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
渲染 ZhiXing 夹爪本体（隔离），用于**目视确认结构**：
  * 哪两个 link 是接触体（带碰撞 geom 的才是）；
  * 指垫面在哪个 link 上、朝向哪里、行程内如何运动；
  * 视觉 geom（group 1）与碰撞 geom（group 0）的差别。

不依赖 env / 不依赖策略，直接加载 `zhixing_gripper.xml`。
"""
import argparse
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

XML = "/app/robosuite/models/assets/grippers/zhixing_gripper.xml"
OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results/gripper_render"
W = H = 480


def look_at(cam, target, dist, azim_deg, elev_deg):
    a, e = np.radians(azim_deg), np.radians(elev_deg)
    cam.lookat[:] = target
    cam.distance = dist
    cam.azimuth = azim_deg
    cam.elevation = elev_deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-collision", action="store_true")
    a = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "Left_1_Joint")

    # 打印结构清单
    print("=== bodies / geoms ===")
    for b in range(m.nbody):
        bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        rows = []
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) != b:
                continue
            gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            gt = m.geom_type[g]
            collides = (m.geom_contype[g] != 0) or (m.geom_conaffinity[g] != 0)
            rows.append("%s(type=%d,contype=%d,conaff=%d%s)"
                        % (gn, gt, m.geom_contype[g], m.geom_conaffinity[g],
                           ",**碰撞**" if collides else ",仅视觉"))
        kind = ""
        for j in range(m.njnt):
            if int(m.jnt_bodyid[j]) == b:
                kind = " joint=%s range=%s" % (
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                    np.round(m.jnt_range[j], 3).tolist())
        if rows or kind:
            print("  body[%2d] %-28s%s" % (b, bn, kind))
            for r in rows:
                print("        %s" % r)
    print("=== actuators ===")
    for i in range(m.nu):
        print("  [%d] %s joint=%s ctrlrange=%s kp=%s forcelimited=%d forcerange=%s"
              % (i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i),
                 mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(m.actuator_trnid[i][0])),
                 np.round(m.actuator_ctrlrange[i], 3).tolist(), m.actuator_gainprm[i][0],
                 int(m.actuator_forcelimited[i]), np.round(m.actuator_forcerange[i], 3).tolist()))

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)

    # 碰撞 geom 高亮成红色；视觉 geom 灰色
    for g in range(m.ngeom):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if n.endswith("_col") or n.endswith("pad_collision"):
            m.geom_rgba[g] = [1.0, 0.15, 0.1, 1.0]
        else:
            m.geom_rgba[g] = [0.62, 0.64, 0.68, 1.0]

    def body_pos(name):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        return np.array(d.xpos[b], dtype=float)

    rnd = mujoco.Renderer(m, height=480, width=480)
    opt = mujoco.MjvOption()
    import PIL.Image

    for label, jval in (("open", 0.0), ("closed", -0.91)):
        d.qpos[int(m.jnt_qposadr[jid])] = jval
        mujoco.mj_forward(m, d)
        L, R = body_pos("Left_Support_Link"), body_pos("Right_Support_Link")
        mid = 0.5 * (L + R)
        axis = (R - L) / max(np.linalg.norm(R - L), 1e-9)
        print("[%s] L=%s R=%s |LR|=%.1fmm axis=%s"
              % (label, np.round(L, 4).tolist(), np.round(R, 4).tolist(),
                 np.linalg.norm(R - L) * 1000, np.round(axis, 3).tolist()))
        # 视图：垂直于闭合轴看（0），沿闭合轴看（90）
        for vname, azim, elev, dist in (("perp", 0.0, 0.0, 0.20),
                                        ("along", 90.0, 0.0, 0.22),
                                        ("iso", 35.0, 25.0, 0.22)):
            cam = mujoco.MjvCamera()
            cam.lookat[:] = mid
            cam.distance = dist
            cam.azimuth = azim
            cam.elevation = elev
            opt.geomgroup[:] = 1
            rnd.update_scene(d, camera=cam, scene_option=opt)
            pth = pathlib.Path(OUT) / ("%s_%s.png" % (label, vname))
            PIL.Image.fromarray(rnd.render()).save(pth)
            print("[img]", pth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
