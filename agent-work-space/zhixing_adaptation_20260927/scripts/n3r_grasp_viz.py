#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ZhiXing 夹爪抓持能力 —— MuJoCo 可视化（双画面）。

在已落地资产（forcerange="-2 2" + 指垫橡胶参数）上跑一次完整抓取：
接近 -> 下降 -> 闭合到停住 -> 力驱动抬升 5 cm -> 保持 2 s -> 放回 -> 张开。

输出：
  grasp_<W>mm.mp4        完整时序（左 = agentview 全局，右 = 特写）
  stage_<W>mm_*.png      关键阶段大图
  frames_<W>mm/          逐帧（--save-frames）

两个「实测踩过」的坑，已在代码里规避：
1. 不要开 mjVIS_CONTACTFORCE：力矢量 glyph 体积巨大，会把画面糊成橙色。
2. 不要直接画 robosuite 的碰撞 geom（group 0）：它们被染成橙色，叠上去会把
   场景变成橙色块。这里改为渲染前重新着色（group-0 -> 中性灰，试块 -> 红）。
   也不能把试块 geom 改成 group="1" —— robosuite 的 arena merge 会报
   "mass ... larger than mjMINVAL"（已实测）。

特写机位：默认 --cam-follow agentview，沿用 agentview 的观察方向（已知无遮挡）
但拉近到试块上，避免在杂物堆里瞎试机位。
"""
import argparse
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N      # noqa: E402
import n3r_block_test as B     # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results/grasp_viz"


def prepare_scene_colors(m, mujoco):
    """渲染前重新着色：group-0 碰撞体 -> 中性灰；试块 -> 红。"""
    n = 0
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if "block" in nm and nm.endswith("_geom"):
            m.geom_rgba[g] = [0.85, 0.22, 0.18, 1.0]
            n += 1
        elif int(m.geom_group[g]) == 0:
            m.geom_rgba[g] = [0.70, 0.71, 0.73, 1.0]
            n += 1
    return n


def camera_from_builtin(m, mujoco, camname, look, dist):
    """沿用内置相机的观察方向，但把机位拉近到 look 上。返回 (azim, elev)。"""
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, camname)
    if cid < 0:
        return None
    p = np.array(m.cam_pos[cid], dtype=float)
    d = p - np.asarray(look, dtype=float)
    r = float(np.linalg.norm(d))
    if r < 1e-9:
        return None
    elev = float(np.degrees(np.arcsin(np.clip(d[2] / r, -1.0, 1.0))))
    azim = float(np.degrees(np.arctan2(d[1], d[0])))
    print("[cam] 由 %s 推导特写机位: azim=%.1f elev=%.1f 距离 %.2f m"
          % (camname, azim, elev, dist))
    return azim, elev


class Viz:
    def __init__(self, env, out_dir, res, azim, elev, dist, builtin, crop=None):
        import mujoco

        self.env = env
        self.out = pathlib.Path(out_dir)
        (self.out / "frames").mkdir(parents=True, exist_ok=True)
        self.frames, self.stage, self.n = [], {}, 0
        sim, m, d, _ = N.ids(env)
        w, h = res
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), w * 2)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), h)
        n = prepare_scene_colors(m, mujoco)
        print("[viz] 重新着色 %d 个 geom（试块=红，碰撞体=灰）" % n)
        self.rnd = mujoco.Renderer(m, height=h, width=w)
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 1
        for f in ("mjVIS_CONTACTFORCE", "mjVIS_CONTACTPOINT", "mjVIS_TRANSPARENT"):
            self.opt.flags[getattr(mujoco.mjtVisFlag, f)] = False
        self.builtin = None if builtin in (None, "", "none") else builtin
        self.crop = crop            # (x0, y0, x1, y1) 于 agentview 画面内
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.distance, self.cam.azimuth, self.cam.elevation = dist, azim, elev

    def snap(self, lookat, tag=None, save_png=False, label=None, sub=None):
        sim, m, d, _ = N.ids(self.env)
        self.cam.lookat[:] = lookat
        panes = []
        if self.builtin:
            self.rnd.update_scene(d, camera=self.builtin, scene_option=self.opt)
            panes.append(self.rnd.render().copy())
        self.rnd.update_scene(d, camera=self.cam, scene_option=self.opt)
        panes.append(self.rnd.render().copy())
        if self.crop is not None and len(panes) == 2:
            # 用 agentview 的 2D 放大裁剪做特写：比自由相机可靠得多
            # （自由相机的 elevation 符号很容易把机位放到桌面以下，实测踩到）
            x0, y0, x1, y1 = self.crop
            patch = panes[0][y0:y1, x0:x1]     # 注意别用 sub：会遮蔽标注参数
            import cv2
            panes[1] = cv2.resize(patch, (panes[0].shape[1], panes[0].shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
        img = panes[0] if len(panes) == 1 else np.hstack(panes)
        if label:
            import cv2
            cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(img, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2, cv2.LINE_AA)
        if sub:
            import cv2
            cv2.putText(img, sub, (12, img.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 4, cv2.LINE_AA)
            cv2.putText(img, sub, (12, img.shape[0] - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 80), 2, cv2.LINE_AA)
        self.frames.append(img)
        self.n += 1
        if tag:
            self.stage[tag] = img.copy()
        if save_png:
            import PIL.Image
            PIL.Image.fromarray(img).save(
                self.out / "frames" / ("%04d_%s.png" % (self.n, tag or "f")))
        return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width-mm", type=float, default=50.0)
    ap.add_argument("--height-mm", type=float, default=70.0)
    ap.add_argument("--depth-mm", type=float, default=60.0)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--res-w", type=int, default=640)
    ap.add_argument("--res-h", type=int, default=480)
    ap.add_argument("--builtin", default="agentview")
    ap.add_argument("--cam-follow", default="agentview",
                    help="沿用哪个内置相机的方向做特写；none = 用手调角度")
    ap.add_argument("--cam-azim", type=float, default=0.0)
    ap.add_argument("--cam-elev", type=float, default=12.0)
    ap.add_argument("--cam-dist", type=float, default=0.30)
    ap.add_argument("--crop", default="300,120,530,290",
                    help="右画面 = agentview 的该矩形放大（x0,y0,x1,y1）；空串则用自由相机")
    ap.add_argument("--place-x", type=float, default=-0.22)
    ap.add_argument("--place-y", type=float, default=0.10)
    ap.add_argument("--drop-z", type=float, default=1.30)
    ap.add_argument("--settle-steps", type=int, default=400)
    ap.add_argument("--close-max-steps", type=int, default=8000)
    ap.add_argument("--close-chunk", type=int, default=60)
    ap.add_argument("--lift-m", type=float, default=0.05)
    ap.add_argument("--grasp-above-table", type=float, default=0.035)
    ap.add_argument("--snap-every", type=int, default=4)
    ap.add_argument("--save-frames", action="store_true")
    a = ap.parse_args()

    hy, hz = a.width_mm / 2000.0, a.height_mm / 2000.0
    hx = a.depth_mm / 2000.0
    B.install_block_injection((hx, hy, hz), (a.place_x, a.place_y, a.drop_z))
    env, _task = N.build(task_id=0)
    env.reset()
    for _ in range(a.settle_steps):
        env.step(np.concatenate([np.zeros(6), [0.0]]))

    import mujoco
    sim, m, d, gid = N.ids(env)
    aid = N.gripper_actuator(env)
    bid = B.block_id(sim)
    pts = N.body_world_pts(sim, bid, collision_only=True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1]))
    z_g = float(lo[2]) + a.grasp_above_table
    look = np.array([cx, cy, 0.5 * (float(lo[2]) + float(hi[2]))])
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    aadr = N._arm_idx(env)[5]
    q_home = np.array([d.qpos[x] for x in aadr], dtype=float)
    obj0 = N.body_world_pts(sim, bid).mean(axis=0)
    pin = np.array([cx, cy, z_g])

    print("[cfg] forcelimited=%d forcerange=%s"
          % (int(m.actuator_forcelimited[aid]),
             np.round(m.actuator_forcerange[aid], 3).tolist()))
    print("[scene] 试块 %.0fx%.0fx%.0f mm  xy=(%.3f,%.3f)  z=[%.4f,%.4f]"
          % (a.depth_mm, a.width_mm, a.height_mm, cx, cy, lo[2], hi[2]))

    azim, elev = a.cam_azim, a.cam_elev
    if a.cam_follow and a.cam_follow != "none":
        got = camera_from_builtin(m, mujoco, a.cam_follow, look, a.cam_dist)
        if got:
            azim, elev = got

    crop = None
    if a.crop:
        crop = tuple(int(v) for v in a.crop.split(","))
        print("[viz] 右画面 = agentview 裁剪 %s 放大" % (crop,))
    v = Viz(env, a.out_dir, (a.res_w, a.res_h), azim, elev, a.cam_dist,
            a.builtin, crop=crop)

    def h_mm():
        return (N.body_world_pts(N.ids(env)[0], bid).mean(axis=0)[2] - obj0[2]) * 1000

    def jaw_mm():
        j = N.jaw_face_separation_mm(env)
        return j if j is not None else float("nan")

    def snap(tag=None, png=False, label=None):
        # cv2.putText 用 Hershey 字体，**不支持中文**（会渲染成 ?????）。
        # 需要中文得改用 PIL + CJK 字体；这里用 ASCII 保证可读。
        sub = "block lift %+6.1f mm    jaw gap %6.2f mm" % (h_mm(), jaw_mm())
        v.snap(look, tag=tag, save_png=png or a.save_frames,
               label=label or tag, sub=sub)

    snap("00_initial", True, "0. initial (block on table)")

    q_pre, _ = N.ik_goto(env, q_home, pin + np.array([0, 0, 0.06]), R_ref, 0.0, settle=15)
    snap("01_approach", True, "1. approach")
    q_g, _ = N.ik_goto(env, q_pre, pin, R_ref, 0.0, settle=15)
    snap("02_descended", True, "2. descend to grasp height")

    l1_prev, stable, total = None, 0, 0
    while total < a.close_max_steps:
        N.step_hold(env, q_g, -0.91, n=a.close_chunk)
        total += a.close_chunk
        l1 = B.left1_qpos(env)
        snap("03_closing", total == a.close_chunk)
        if l1_prev is not None and abs(l1 - l1_prev) < 1e-4:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
        l1_prev = l1
    print("[close] %d 子步  Left_1=%.4f  夹爪净距=%.2f mm（试块 %.0f mm）"
          % (total, B.left1_qpos(env), N.jaw_face_separation_mm(env), a.width_mm))
    snap("04_closed", True, "4. closed -> grasped")

    sim, m, d, gid = N.ids(env)
    gs0 = np.array(d.site_xpos[gid], dtype=float)
    target = gs0 + np.array([0, 0, a.lift_m])
    R0 = N.gripsite(sim, m, d, gid)[1].copy()
    for k in range(1, 301):
        sim, m, d, gid = N.ids(env)
        p, R = N.gripsite(sim, m, d, gid)
        act_p = np.clip((target - p) / N.OSC_POS_SCALE, -1, 1)
        act_r = np.clip(N.rot_err_vec(R0, R) / N.OSC_ROT_SCALE, -1, 1)
        env.step(np.concatenate([act_p, act_r, [+1.0]]))
        if k % a.snap_every == 0:
            snap("05_lifting", False, "5. lifting...")
        sim, m, d, gid = N.ids(env)
        if np.linalg.norm(target - N.gripsite(sim, m, d, gid)[0]) < 0.003:
            break
    obj_lift = N.body_world_pts(N.ids(env)[0], bid).mean(axis=0)
    print("[lift ] 命令升 %.0f mm -> 试块实际抬升 %+.1f mm"
          % (a.lift_m * 1000, (obj_lift[2] - obj0[2]) * 1000))
    snap("06_lifted", True, "5. lift 5 cm (OSC force)")

    for k in range(40):
        N.hold(env, +1.0, 1)
        if k % a.snap_every == 0:
            snap("07_holding", False, "6. holding...")
    obj_hold = N.body_world_pts(N.ids(env)[0], bid).mean(axis=0)
    print("[hold ] 保持 2 s -> 抬升 %+.1f mm" % ((obj_hold[2] - obj0[2]) * 1000))
    snap("08_held", True, "6. hold 2 s")

    q_now = np.array([N.ids(env)[2].qpos[x] for x in aadr], dtype=float)
    N.ik_goto(env, q_now, pin, R_ref, -0.91, settle=15)
    snap("09_placed", True, "7. place back")
    N.step_hold(env, q_g, 0.0, n=600)
    snap("10_released", True, "8. release")
    obj_end = N.body_world_pts(N.ids(env)[0], bid).mean(axis=0)
    print("[place] 放回后 z 偏差 %+.1f mm" % ((obj_end[2] - obj0[2]) * 1000))
    print("[result] 成功 = %s（判据：抬升 >30 mm 且保持 >30 mm）"
          % bool((obj_hold[2] - obj0[2]) > 0.03))

    import imageio
    mp4 = pathlib.Path(a.out_dir) / ("grasp_%.0fmm.mp4" % a.width_mm)
    imageio.mimsave(str(mp4), v.frames, fps=a.fps, macro_block_size=None)
    print("[video] %s  (%d 帧, %.1f s @ %d fps)"
          % (mp4, len(v.frames), len(v.frames) / a.fps, a.fps))
    import PIL.Image
    for tag, img in v.stage.items():
        PIL.Image.fromarray(img).save(
            pathlib.Path(a.out_dir) / ("stage_%.0fmm_%s.png" % (a.width_mm, tag)))
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
