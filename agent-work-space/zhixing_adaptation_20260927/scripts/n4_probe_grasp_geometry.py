#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4 诊断：**为什么闭合时把物体打飞** —— 直接测量指垫与物体在抓取位形的相对几何。

背景
----
`n4_collect.py` 首次试采：descend 后 `grip_site` 与物体中心只差 1~2 mm，
但闭合 300 步后物体从 (-0.119,-0.244) 飞到 (-0.524,-0.059)（位移 0.44 m），
夹爪净距收到 0.6 mm（空合）。说明**指垫没夹住物体，而是把它拨飞了**。

TCP 修复后的关键几何事实（N2）
-----------------------------
`grip_site` = **闭合位**夹持点。夹持点随开合沿接近轴移动 17.44 mm：
    · 闭合位：夹持点 = `grip_site`（偏移 0）
    · 全开位：夹持点 = `grip_site` 沿接近轴**后退** 17.44 mm
所以「把 grip_site 放到物体中心」并不等于「张开的指垫正对物体中心」——
张开的指垫其实在 grip_site 上方 17.4 mm 处。

本脚本对**开/闭**两个位形，在 `grip_site` 局部系里量出：
  * 两指垫内侧面中点（夹持点）
  * 指垫接触顶点的**局部包围盒**（横向 x/y 范围、沿接近轴 z 范围）
  * 指尖最深点
并据此算出「要让指垫在竖直方向上覆盖物体中段，`grip_site` 该放在哪」。
另附**抓取高度扫描**：直接在若干 grip_site 高度上试闭合，看哪个不把物体打飞。

只读探测，不改任何资产。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N          # noqa: E402
import n4_collect as C             # noqa: E402
import verify_tcp_fix as V         # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"


def pad_local(env, ctrl, settle=90):
    """驱动夹爪到指定位形，返回指垫在 grip_site 局部系里的几何。"""
    import mujoco

    N.hold(env, ctrl, settle)
    sim, m, d, gid = N.ids(env)
    p = np.array(d.site_xpos[gid], dtype=float)
    R = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)
    pair = V.find_pad_geoms(m)
    if pair is None:
        return None

    def verts(g):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = np.array(m.mesh_vert[a:a + n], dtype=float)
        else:
            s = np.array(m.geom_size[g], dtype=float)
            v = s * np.array([[sx, sy, sz] for sx in (-1, 1)
                              for sy in (-1, 1) for sz in (-1, 1)])
        Rw = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
        return (Rw @ v.T).T + np.array(d.geom_xpos[g], dtype=float)

    out = {"grip_site_world": p.tolist()}
    faces = []
    for tag, g in zip(("left", "right"), pair):
        Vw = verts(g)
        Vl = (Vw - p) @ R                      # 转到 grip_site 局部系
        # 内侧面 = 朝对方最近的一薄层
        other = verts(pair[1] if tag == "left" else pair[0])
        u = other.mean(axis=0) - Vw.mean(axis=0)
        u /= np.linalg.norm(u)
        sel = (Vw @ u) >= (Vw @ u).max() - 0.005
        face_w = Vw[sel]
        face_l = (face_w - p) @ R
        faces.append(face_w.mean(axis=0))
        out[tag] = {
            "face_mid_local_mm": (face_l.mean(axis=0) * 1000).round(3).tolist(),
            # 局部系：x/y 横向，z 沿接近轴（+z = 更深/更低）
            "face_bbox_local_mm": {
                "x": [round(float(face_l[:, 0].min() * 1000), 2),
                      round(float(face_l[:, 0].max() * 1000), 2)],
                "y": [round(float(face_l[:, 1].min() * 1000), 2),
                      round(float(face_l[:, 1].max() * 1000), 2)],
                "z": [round(float(face_l[:, 2].min() * 1000), 2),
                      round(float(face_l[:, 2].max() * 1000), 2)],
            },
            "all_verts_z_range_local_mm": [
                round(float(Vl[:, 2].min() * 1000), 2),
                round(float(Vl[:, 2].max() * 1000), 2)],
        }
    out["pinch_mid_local_mm"] = (np.mean(faces, axis=0) - p) @ R * 1000
    out["jaw_separation_mm"] = float(np.linalg.norm(faces[0] - faces[1]) * 1000)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--heights-mm", default="20,28,36,44,52")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    import mujoco

    res = {"node": "N4-diag", "check": "grasp_geometry", "suite": args.suite,
           "task_id": args.task_id, "seed": args.seed}
    env, task = C.build(args.suite, args.task_id, seed=args.seed)
    env.reset()
    print("=" * 78)
    print("[task] %s" % task.language)

    # ---------- 1. 指垫几何（开 / 闭） ----------
    for label, ctrl in (("OPEN", 0.0), ("CLOSED", -0.91)):
        g = pad_local(env, ctrl)
        res[label] = g
        print("\n[%s] grip_site 局部系（x/y 横向，z 沿接近轴，+z 更深/更低）" % label)
        print("  夹持点(内面中点) = %s mm   净距 %.2f mm"
              % (np.round(g["pinch_mid_local_mm"], 2).tolist(),
                 g["jaw_separation_mm"]))
        for tag in ("left", "right"):
            f = g[tag]
            print("  %-5s 内面中点=%s mm  内面 bbox: x=%s y=%s z=%s"
                  % (tag, f["face_mid_local_mm"], f["face_bbox_local_mm"]["x"],
                     f["face_bbox_local_mm"]["y"], f["face_bbox_local_mm"]["z"]))

    sim, m, d, gid = N.ids(env)
    b = C.goal_objects(env)
    bid, bname = C.find_body(sim, b[0])
    rb, rh = C.bbox(sim, C.find_body(sim, b[1])[0])
    N.hold(env, -1.0, 90)
    sim, m, d, gid = N.ids(env)
    ob, oh = C.bbox(sim, bid)
    print("\n[object] %s  z范围=[%.4f, %.4f]  高 %.1f mm  xy中心=(%.4f, %.4f)"
          % (bname, ob[2], oh[2], (oh[2] - ob[2]) * 1000,
             (ob[0] + oh[0]) / 2, (ob[1] + oh[1]) / 2))
    res["object"] = {"name": bname, "z": [float(ob[2]), float(oh[2])],
                     "h_mm": float((oh[2] - ob[2]) * 1000)}

    # ---------- 2. 抓取高度扫描：哪个 grip_site 高度能把物体夹住 ----------
    print("\n" + "=" * 78)
    print("[sweep] 固定 xy=物体中心，扫 grip_site 高度，看闭合是否抓住（不打飞）")
    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    cx, cy = float((ob[0] + oh[0]) / 2), float((ob[1] + oh[1]) / 2)
    rows = []
    for zmm in [float(x) for x in args.heights_mm.split(",") if x.strip()]:
        env.seed(args.seed)
        env.reset()
        for _ in range(30):
            env.step(np.concatenate([np.zeros(6), [-1.0]]))
        sim, m, d, gid = N.ids(env)
        ob2, oh2 = C.bbox(sim, bid)
        c0 = 0.5 * (ob2 + oh2)
        R_ref = N.gripsite(sim, m, d, gid)[1].copy()
        pin = np.array([cx, cy, zmm / 1000.0])

        # 下降到目标
        for _ in range(80):
            p, R = N.gripsite(sim, m, d, gid)
            a = np.concatenate([
                np.clip((pin - p) / C.OSC_POS_SCALE, -1, 1),
                np.clip(N.rot_err_vec(R_ref, R) / C.OSC_ROT_SCALE, -1, 1),
                [-1.0]])
            env.step(a)
            sim, m, d, gid = N.ids(env)
        err = float(np.linalg.norm(np.array(d.site_xpos[gid]) - pin) * 1000)
        sep_before = N.jaw_face_separation_mm(env)
        pos_before = 0.5 * (lambda bb: bb[0] + bb[1])(C.bbox(sim, bid))

        # 闭合到停住
        l1p, stable, ncl = None, 0, 0
        for _ in range(220):
            p, R = N.gripsite(sim, m, d, gid)
            a = np.concatenate([
                np.clip((pin - p) / C.OSC_POS_SCALE, -1, 1),
                np.clip(N.rot_err_vec(R_ref, R) / C.OSC_ROT_SCALE, -1, 1),
                [1.0]])
            env.step(a)
            ncl += 1
            sim, m, d, gid = N.ids(env)
            l1 = C.left1(env)
            if l1p is not None and abs(l1 - l1p) < 1e-4:
                stable += 1
                if stable >= 3:
                    break
            else:
                stable = 0
            l1p = l1
        sep_after = N.jaw_face_separation_mm(env)
        obj_after = 0.5 * (lambda bb: bb[0] + bb[1])(C.bbox(sim, bid))
        shift = float(np.linalg.norm(obj_after - pos_before) * 1000)

        # 抬升 5 cm（力驱动）
        for _ in range(90):
            p, R = N.gripsite(sim, m, d, gid)
            tgt = pin + np.array([0, 0, 0.05])
            a = np.concatenate([
                np.clip((tgt - p) / C.OSC_POS_SCALE, -1, 1),
                np.clip(N.rot_err_vec(R_ref, R) / C.OSC_ROT_SCALE, -1, 1),
                [1.0]])
            env.step(a)
            sim, m, d, gid = N.ids(env)
            if float(np.linalg.norm(np.array(d.site_xpos[gid]) - tgt)) < 0.005:
                break
        obj_lift = 0.5 * (lambda bb: bb[0] + bb[1])(C.bbox(sim, bid))
        dz = float((obj_lift[2] - c0[2]) * 1000)
        caught = bool(sep_after > 20.0 and shift < 30.0 and dz > 30.0)
        rows.append({"grip_site_z_mm": zmm, "descend_err_mm": round(err, 2),
                     "sep_before_mm": round(sep_before or -1, 1),
                     "sep_after_mm": round(sep_after or -1, 1),
                     "close_steps": ncl, "obj_shift_mm": round(shift, 1),
                     "lift_mm": round(dz, 1), "caught": caught})
        print("  z=%5.1f mm  descend误差=%5.2f  闭合%3d步  净距 %5.1f→%5.1f mm"
              "  物体位移=%7.1f mm  抬升=%+7.1f mm  %s"
              % (zmm, err, ncl, sep_before or -1, sep_after or -1, shift, dz,
                 "✅抓住" if caught else "❌未抓住"))
    res["sweep"] = rows
    good = [r for r in rows if r["caught"]]
    res["verdict"] = "PASS" if good else "NO_HEIGHT_WORKS"
    res["working_heights_mm"] = [r["grip_site_z_mm"] for r in good]

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(args.out) / "n4_grasp_geometry.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 78)
    print("[verdict] %s  可用高度=%s" % (res["verdict"], res["working_heights_mm"]))
    print("[out] %s" % p)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
