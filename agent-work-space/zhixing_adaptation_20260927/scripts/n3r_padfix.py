#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3-redo A/B：给 ZhiXing 补上与 Panda 同构的**专用指垫碰撞体**，量化其贡献。

依据（n3r_pad_collision.py 实测 + XML 对照）
--------------------------------------------
Panda 每个手指有两个碰撞 geom：手指本体 mesh（软参数）+ **专用橡胶垫 box**
（solref="0.01 0.5"、friction="2 0.05 0.0001"、contype=1）。
ZhiXing **缺**专用垫，只有整条 Support_Link 的 mesh 凸包，沿用本体软参数。

本脚本在**测试进程内**生成一份改过的夹爪 XML 副本（写入 /tmp），
注册成 `ZhixingPadFix` 并显式作为 `gripper_types` 传入——**磁盘资产零改动**。

用法
----
  --mode add      保留原 mesh 碰撞，额外加专用垫 box（与 Panda 同构，默认）
  --mode replace  关掉原 mesh 垫的碰撞，只用专用垫 box（更干净的隔离）
参数来源：results/n3r_pad_collision.json（由 n3r_pad_collision.py 生成）。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N          # noqa: E402
import n3r_block_test as B         # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"
SRC_XML = "/app/robosuite/models/assets/grippers/zhixing_gripper.xml"
DST_XML = "/tmp/zhixing_gripper_padfix.xml"

# 与 Panda 专用垫一致的接触参数
PAD_SOLREF = "0.01 0.5"
PAD_FRICTION = "2 0.05 0.0001"
PAD_CONDIM = "4"


def quat_from_frame(e1, e2, u):
    """由三个正交基（列）构造 [w,x,y,z] 四元数。"""
    import mujoco

    R = np.column_stack([e1, e2, u])
    # 正交化，防止数值漂移
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return [float(v) for v in q]


def build_variant_xml(mode, pad_json, thin_mm=3.0, shrink=0.75, name_fixed="ZhixingPadFix"):
    """把专用垫 box 写进夹爪 XML 的副本。返回 (path, 说明)。

    副本落在 /tmp，故必须把 `<asset>` 里的**相对** mesh 路径改写成绝对路径，
    否则 MuJoCo 会去 /tmp/meshes/... 找 STL（实测踩到过）。
    """
    import os
    import xml.etree.ElementTree as ET

    tree = ET.parse(SRC_XML)
    root = tree.getroot()

    # 相对资产路径 -> 绝对（等价于 robosuite MujocoXML.resolve_asset_dependency）
    folder = os.path.dirname(SRC_XML)
    asset = root.find("asset")
    if asset is not None:
        for node in asset.findall("./*[@file]"):
            f = node.get("file")
            if f and not os.path.isabs(f):
                node.set("file", os.path.join(folder, f))

    wb = root.find("worldbody")

    notes = []
    body_of = {"left": "Left_Support_Link", "right": "Right_Support_Link"}
    geom_of = {"left": "left_fingerpad_col", "right": "right_fingerpad_col"}
    padname = {"left": "left_pad_collision", "right": "right_pad_collision"}

    for side in ("left", "right"):
        target = None
        for b in wb.iter("body"):
            if b.get("name") == body_of[side]:
                target = b
                break
        if target is None:
            raise RuntimeError(f"找不到 body {body_of[side]}")

        # replace 模式：关掉原 mesh 垫的碰撞（contype/conaffinity 置 0）
        for g in target.findall("geom"):
            if g.get("name") == geom_of[side]:
                if mode == "replace":
                    g.set("contype", "0")
                    g.set("conaffinity", "0")
                    notes.append(f"{geom_of[side]}: collision disabled")

        p = pad_json["pads"][side]
        c = np.array(p["face_center_body_local_m"], dtype=float)
        u = np.array(p["inner_dir_body_local"], dtype=float)
        e1 = np.array(p["e1_body_local"], dtype=float)
        e2 = np.array(p["e2_body_local"], dtype=float)
        pmax = p["proj_max_mm"] / 1000.0
        thin = thin_mm / 1000.0
        hx = max(p["face_half_extent_e1_mm"] / 1000.0 * shrink, 0.004)
        hy = max(p["face_half_extent_e2_mm"] / 1000.0 * shrink, 0.004)
        # 内侧面(max along u) 与原 mesh 内侧面齐平
        c_box = c + u * ((pmax - thin) - float(c @ u))
        quat = quat_from_frame(e1, e2, u)

        geom = ET.SubElement(target, "geom")
        geom.set("type", "box")
        geom.set("name", padname[side])
        geom.set("size", f"{hx:.6f} {hy:.6f} {thin:.6f}")
        geom.set("pos", f"{c_box[0]:.6f} {c_box[1]:.6f} {c_box[2]:.6f}")
        geom.set("quat", " ".join(f"{v:.8f}" for v in quat))
        geom.set("group", "0")
        geom.set("contype", "1")
        geom.set("conaffinity", "1")
        geom.set("condim", PAD_CONDIM)
        geom.set("solref", PAD_SOLREF)
        geom.set("friction", PAD_FRICTION)
        geom.set("rgba", "0 1 0 0.3")
        notes.append(f"{padname[side]}: box size=({hx*1000:.1f},{hy*1000:.1f},{thin*1000:.1f})mm "
                     f"quat_set=1 solref='{PAD_SOLREF}' friction='{PAD_FRICTION}'")

    tree.write(DST_XML, encoding="utf-8", xml_declaration=True)
    return DST_XML, notes


def register_variant(path, name="ZhixingPadFix"):
    """用改过的 XML 副本注册一个测试专用夹爪类（继承 ZhixingGripper 的动作语义）。"""
    from robosuite.models.grippers import GRIPPER_MAPPING
    from robosuite.models.grippers.gripper_model import GripperModel
    import robosuite.models.grippers.zhixing_gripper as ZG

    class ZhixingPadFixFull(ZG.ZhixingGripper):
        def __init__(self, idn=0):
            GripperModel.__init__(self, path, idn=idn)

    GRIPPER_MAPPING[name] = ZhixingPadFixFull
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["add", "replace"], default="add")
    ap.add_argument("--widths-mm", default="30,50,70")
    ap.add_argument("--height-mm", type=float, default=70.0)
    ap.add_argument("--depth-mm", type=float, default=60.0)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--place-x", type=float, default=-0.22)
    ap.add_argument("--place-y", type=float, default=0.10)
    ap.add_argument("--drop-z", type=float, default=1.30)
    ap.add_argument("--settle-steps", type=int, default=400)
    ap.add_argument("--close-steps", type=int, default=600)
    ap.add_argument("--hold-steps", type=int, default=40)
    ap.add_argument("--lift-m", type=float, default=0.05)
    ap.add_argument("--lift-ramp", type=int, default=25)
    ap.add_argument("--ramp-settle", type=int, default=8)
    ap.add_argument("--torque-cap", type=float, default=None)
    ap.add_argument("--close-max-steps", type=int, default=8000)
    ap.add_argument("--close-chunk", type=int, default=50)
    ap.add_argument("--grasp-above-table", type=float, default=0.035)
    ap.add_argument("--jitter-mm", type=float, default=0.0)
    ap.add_argument("--thin-mm", type=float, default=3.0)
    ap.add_argument("--shrink", type=float, default=0.75)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    pad_json = json.loads(
        (pathlib.Path(OUT) / "n3r_pad_collision.json").read_text(encoding="utf-8"))
    path, notes = build_variant_xml(a.mode, pad_json, thin_mm=a.thin_mm, shrink=a.shrink)
    gname = register_variant(path)

    print(f"[padfix] mode={a.mode}  xml={path}")
    for n in notes:
        print(f"    {n}")

    widths = [float(v) for v in a.widths_mm.split(",") if v.strip()]
    tag = a.tag or f"block_padfix_{a.mode}"
    hz, hx = a.height_mm / 2000.0, a.depth_mm / 2000.0

    summary = {"node": "N3-redo", "tag": tag, "method": "standard_solid_block",
               "gripper": gname, "pad_mode": a.mode,
               "pad_notes": notes, "opening_mm": 89.96,
               "threshold_per_width": 9, "results": []}

    for w in widths:
        hy = w / 2000.0
        size_m = (hx, hy, hz)
        B.install_block_injection(size_m, (a.place_x, a.place_y, a.drop_z))
        env, task = N.build(task_id=0, gripper=gname)
        print(f"\n{'='*76}\n[block] y-width={w:.0f} mm  gripper={gname}  mode={a.mode}")

        trials, n_ok = [], 0
        for t in range(a.trials):
            try:
                r = B.run_one(env, size_m, a, trial_seed=1000 + t)
            except Exception as e:                       # noqa: BLE001
                r = {"success": False, "reason": f"exception:{e}"}
            r["trial"] = t + 1
            trials.append(r)
            n_ok += int(r.get("success", False))
            cc = r.get("contact_close", {}) or {}
            print(f"  [{t+1:2d}/{a.trials}] IK={r.get('ik_res_mm',{}).get('grasp', float('nan')):5.2f}mm "
                  f"闭合净距={r.get('jaw_close_mm', float('nan')):6.2f}mm "
                  f"推挤={r.get('obj_shift_on_close_mm', float('nan')):6.1f}mm "
                  f"接触={cc.get('n_contacts','-')} 穿透={cc.get('min_dist_mm','-')} "
                  f"抬升={r.get('dz_lift_mm', float('nan')):+7.1f} 保持={r.get('dz_hold_mm', float('nan')):+7.1f}mm"
                  f"  {'OK 成功' if r.get('success') else 'XX ' + str(r.get('reason'))}")

        passed = (n_ok / a.trials) >= 0.9
        summary["results"].append({"width_mm": w, "n_success": n_ok, "n_total": a.trials,
                                   "success_rate": n_ok / a.trials, "pass": bool(passed),
                                   "trials": trials})
        print(f"  → 宽度 {w:.0f} mm：成功 {n_ok}/{a.trials} {'PASS' if passed else 'FAIL'}")
        env.close()

    best = max(summary["results"], key=lambda r: r["n_success"])
    summary["best_width_mm"] = best["width_mm"]
    summary["best_success"] = f"{best['n_success']}/{best['n_total']}"
    summary["any_width_pass"] = bool(any(r["pass"] for r in summary["results"]))
    summary["verdict"] = "PASS" if summary["any_width_pass"] else "NOT_PASSED"

    p = pathlib.Path(OUT) / f"n3r_{tag}.json"
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*76}\n[verdict] {summary['verdict']}  best={best['width_mm']:.0f}mm "
          f"{best['n_success']}/{best['n_total']}\n[out] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
