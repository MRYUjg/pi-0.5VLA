#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2：汇总生成 **ZhiXing TCP / 相机契约**，并给出相机对齐的量化建议。

输入（均为 N2 实测产物）：
  results/tcp_camera_audit.json     # 三配置几何 + 相机变换 + 视觉占比
  results/tcp_geometry_sites.json   # site 级交叉验证
  results/zhixing_init_pose.json    # 初始位姿 IK 重解

输出：
  results/tcp_camera_contract.json  # 机器可读契约
  results/camera_alignment_plan.json# 相机对齐量化选项
"""
import json
import math
import pathlib
import sys

import numpy as np

RESULTS = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927/results")
FOVY0_DEG = 75.0  # ECO65-B/robot.xml 的 eye_in_hand fovy
APPROACH_AXIS = "+z（grip_site 局部系，指向工作空间/桌面下方）"


def load(name):
    p = RESULTS / name
    if not p.exists():
        raise SystemExit(f"缺少输入 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    audit = load("tcp_camera_audit.json")
    sites = load("tcp_geometry_sites.json")
    initp = load("zhixing_init_pose.json")
    padf = load("pad_face_geometry.json")   # ★ 权威口径：约束收敛后的内侧面中点

    by = {c["tag"]: c for c in audit["configs"]}
    zx, ec_p, pd = by["eco65_zhixing"], by["eco65_panda"], by["panda_panda"]

    pf_zx = padf["eco65_zhixing"]["states"]
    pf_pd = padf["eco65_panda"]["states"]

    def g(cfg, state="reset"):
        return cfg["geometry"][state]

    # ---------- 契约 ----------
    contract = {
        "node": "N2",
        "date": "2026-09-27",
        "approach_axis_convention": APPROACH_AXIS,
        "definitions": {
            "grip_site": "robosuite/OSC 使用的 eef site；策略动作即作用于此点",
            "grasp_point": "两指垫碰撞 geom **内侧面中点**；在四连杆约束收敛的构型下测量",
            "beyond_gripsite_mm": "沿接近轴相对 grip_site 的前伸量，正值=更深入工作空间",
        },
        "panda_reference": {
            "config": "Panda + PandaGripper",
            "grip_site_world_pos": g(pd)["grip_site_world_pos"],
            "grasp_point_offset_along_approach_mm": {
                "open": pf_pd["open"]["pinch_approach_mm"],
                "closed": pf_pd["closed"]["pinch_approach_mm"],
                "note": "PandaGripper 为平动指，夹持点位置与开合无关（实测恒定）",
            },
            "pad_face_normal_angle_deg": pf_pd["closed"]["pad_face_normal_angle_deg"],
            "fingertip_deepest_vertex_mm": g(pd)["gripper_extent_plus_z_m"] * 1000.0,
            "flange_offset_mm": -g(pd)["gripper_extent_minus_z_m"] * 1000.0,
            "clip": "+9.3 mm（全开/闭合不变；手指为平动指）",
        },
        "zhixing": {
            "config": "ECO65 + ZhiXing",
            "grip_site_world_pos_current_init": g(zx)["grip_site_world_pos"],
            "grasp_point_offset_along_approach_mm": {
                "reset_near_closed": pf_zx["reset"]["pinch_approach_mm"],
                "open": pf_zx["open"]["pinch_approach_mm"],
                "closed": pf_zx["closed"]["pinch_approach_mm"],
                "travel_over_stroke_mm": (
                    pf_zx["closed"]["pinch_approach_mm"] - pf_zx["open"]["pinch_approach_mm"]
                ),
                "note": "二连平行连杆：两面保持平行，但夹持点随闭合并沿接近轴前移约 17.4 mm",
            },
            "pad_face_normal_angle_deg": {
                "reset": pf_zx["reset"]["pad_face_normal_angle_deg"],
                "open": pf_zx["open"]["pad_face_normal_angle_deg"],
                "closed": pf_zx["closed"]["pad_face_normal_angle_deg"],
                "note": "≈3°，确认两指垫在整个行程内保持平行（平行连杆特征）",
            },
            "face_separation_mm": {
                "reset": pf_zx["reset"]["separation_mm"],
                "open": pf_zx["open"]["separation_mm"],
                "closed": pf_zx["closed"]["separation_mm"],
            },
            "constraint_residuals": {
                "reset_eq": pf_zx["reset"]["constraint_residuals"]["joint_eq_max"],
                "reset_connect_m": pf_zx["reset"]["constraint_residuals"]["connect_max_m"],
                "open_connect_m": pf_zx["open"]["constraint_residuals"]["connect_max_m"],
                "closed_connect_m": pf_zx["closed"]["constraint_residuals"]["connect_max_m"],
            },
            "legacy_closest_pair_metric_mm": {
                "reset": g(zx, "reset")["nip_offset_approach_mm"],
                "note": "作废口径：平行面下最近点对退化，会跳到角点，低估约 16 mm",
            },
            "fingertip_deepest_vertex_mm": {
                "reset_near_closed": g(zx, "reset")["gripper_extent_plus_z_m"] * 1000.0,
                "open": g(zx, "open")["gripper_extent_plus_z_m"] * 1000.0,
            },
            "flange_offset_mm": -g(zx)["gripper_extent_minus_z_m"] * 1000.0,
            "jaw_opening_range": {
                "closed_pad_clearance_m": g(zx, "closed")["nip_clearance_m"],
                "open_pad_clearance_m": g(zx, "open")["nip_clearance_m"],
            },
            "site_cross_check_mm": {
                "Left_Support_tip": next(
                    s["local_mm"] for s in sites["eco65_zhixing"]["sites"]
                    if s["name"].endswith("Left_Support_tip")
                ),
                "Right_Support_tip": next(
                    s["local_mm"] for s in sites["eco65_zhixing"]["sites"]
                    if s["name"].endswith("Right_Support_tip")
                ),
                "eef_body_offset": next(
                    s["local_mm"] for s in sites["eco65_zhixing"]["bodies"]
                    if s["name"].endswith("eef")
                ),
            },
            "clip": (
                "夹持区位于声明 grip_site **前方 +82.5 mm（闭合）～ +65.1 mm（全开）**；"
                "指尖最深 +102 mm。推荐把 grip_site 定义在闭合位夹持点，即 eef 前移至 0.20254 m"
            ),
        },
        "mismatch": {
            "grasp_point_vs_panda_reference_mm": initp["current_landed"]["grasp_point_vs_ref_mm"],
            "declared_grip_site_vs_reference_mm": initp["current_landed"]["grip_site_vs_ref_mm"],
            "fingertip_reach_difference_mm": (
                g(zx)["gripper_extent_plus_z_m"] - g(pd)["gripper_extent_plus_z_m"]
            ) * 1000.0,
        },
        "camera_contract": {
            "camera": "robot0_eye_in_hand (fovy=75, 256x256)",
            "T_cam2grip_trans_delta_zhixing_vs_panda_mm": audit["comparison"]
            ["eco65_zhixing_vs_eco65_panda"]["T_cam2grip_trans_delta_mm"],
            "T_cam2grip_rot_delta_deg": audit["comparison"]
            ["eco65_zhixing_vs_eco65_panda"]["T_cam2grip_rot_delta_deg"],
            "pandagripper_alignment_still_holds": audit["comparison"]
            ["eco65_zhixing_vs_eco65_panda"]["conclusion_holds"],
            "note": (
                "ECO65+PandaGripper 与 Panda+PandaGripper 的 T_cam2grip 逐位相同"
                "（差 2e-13 mm），故 PandaGripper 的结论对 Panda 路径成立；"
                "但装上 ZhiXing 后平移差 23.000 mm，**该结论不成立**"
                "（23 mm = 0.120 m − 0.097 m，即两者 eef 安装长度之差）。"
            ),
        },
        "initial_pose": {
            "landed_init_qpos": initp["current_landed"]["init_qpos"],
            "candidate_init_qpos": initp["candidate"]["init_qpos"],
            "candidate_grasp_point_residual_mm": initp["candidate"]["grasp_point_residual_mm"],
            "candidate_rot_residual_deg": initp["candidate"]["grasp_point_rot_residual_deg"],
            "candidate_limit_margin_min_rad": initp["candidate"]["limit_margin_min_rad"],
            "candidate_landed": False,
        },
    }

    # ---------- 相机对齐量化 ----------
    def cam_to_grasp_mm(cfg):
        """相机 → 夹持点 的真实距离（由 T_cam2grip 与夹持点局部偏移精确计算）。"""
        T = np.array(cfg["camera"]["T_cam2grip"])
        off = np.array(cfg["geometry"]["reset"]["nip_offset_local_m"], dtype=float)
        p = T[:3, :3] @ off + T[:3, 3]
        return float(np.linalg.norm(p) * 1000.0)

    dz, dp = cam_to_grasp_mm(zx), cam_to_grasp_mm(pd)
    print(f"[cam] 相机→夹持点距离: ZhiXing {dz:.1f} mm, Panda {dp:.1f} mm")

    rows = {}
    for key in ("wrist_open", "wrist_reset", "wrist_closed"):
        fz = zx["views"][key]["gripper_pixel_fraction"]
        fp = pd["views"][key]["gripper_pixel_fraction"]
        ratio = fz / fp
        scale = math.sqrt(ratio)  # 一阶近似：像面积 ∝ 1/距离²
        fovy_new = 2.0 * math.degrees(math.atan(scale * math.tan(math.radians(FOVY0_DEG / 2))))
        rows[key] = {
            "zhixing_gripper_pixel_fraction": fz,
            "panda_gripper_pixel_fraction": fp,
            "ratio": ratio,
            "cam_to_grasp_mm": dz,
            "equivalent_camera_pullback_mm": float(dz * (scale - 1.0)),
            "equivalent_fovy_deg": float(fovy_new),
            "fovy_delta_deg": float(fovy_new - FOVY0_DEG),
            "basis": "一阶点源近似（像面积 ∝ 1/d²）；实际值需经验扫掠确定",
        }

    plan = {
        "node": "N2",
        "goal": "让 ZhiXing 在腕部画面中的占比接近 PandaGripper，缓解视觉 OOD",
        "measured_occupancy": {
            "wrist_reset": {"zhixing": rows["wrist_reset"]["zhixing_gripper_pixel_fraction"],
                            "panda": rows["wrist_reset"]["panda_gripper_pixel_fraction"],
                            "ratio": rows["wrist_reset"]["ratio"]},
            "wrist_open": {"zhixing": rows["wrist_open"]["zhixing_gripper_pixel_fraction"],
                           "panda": rows["wrist_open"]["panda_gripper_pixel_fraction"],
                           "ratio": rows["wrist_open"]["ratio"]},
            "wrist_closed": {"zhixing": rows["wrist_closed"]["zhixing_gripper_pixel_fraction"],
                             "panda": rows["wrist_closed"]["panda_gripper_pixel_fraction"],
                             "ratio": rows["wrist_closed"]["ratio"]},
        },
        "per_state_matching_options": rows,
        "blocker": (
            "eye_in_hand 定义在 ECO65-B/robot.xml，**Panda 参考路径与 ZhiXing 路径共用同一份相机参数**；"
            "直接改会同时改变 9/10 基线的视觉输入。需做成按夹爪条件化的相机变体"
            "（例如夹爪专属 robot XML 变体或挂载时覆写），否则不得直接落地。"
        ),
        "recommendation": (
            "优先级低于 TCP 修复：当前 ZhiXing 夹持点比 Panda 参考低 89 mm，属几何错误，"
            "必须先在 N3 前解决；相机占比属感知层缓解，建议在 TCP 修复后用同一评测流程单独消融，"
            "以免两个变量同时变化无法归因。"
        ),
    }

    (RESULTS / "tcp_camera_contract.json").write_text(
        json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / "camera_alignment_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== 契约要点 ===")
    print(f"Panda  夹持点沿接近轴: {pf_pd['closed']['pinch_approach_mm']:+.2f} mm（开合无关）；"
          f"指尖最深 {g(pd)['gripper_extent_plus_z_m']*1000:+.2f} mm")
    print(f"ZhiXing 夹持点沿接近轴: open {pf_zx['open']['pinch_approach_mm']:+.2f} mm / "
          f"closed {pf_zx['closed']['pinch_approach_mm']:+.2f} mm "
          f"(行程内前移 {pf_zx['closed']['pinch_approach_mm']-pf_zx['open']['pinch_approach_mm']:.2f} mm)；"
          f"指尖最深 {g(zx)['gripper_extent_plus_z_m']*1000:+.2f} mm")
    print(f"夹持点与 Panda 基准偏差: {initp['current_landed']['grasp_point_vs_ref_mm']:.2f} mm")
    print(f"指尖前伸差(ZhiXing-Panda): {contract['mismatch']['fingertip_reach_difference_mm']:.2f} mm")
    print(f"★ 建议 eef 位置: 0.12 + {pf_zx['closed']['pinch_approach_mm']/1000:.5f} = "
          f"{0.12 + pf_zx['closed']['pinch_approach_mm']/1000:.5f} m")
    print(f"T_cam2grip 平移差: "
          f"{audit['comparison']['eco65_zhixing_vs_eco65_panda']['T_cam2grip_trans_delta_mm']:.3f} mm"
          f" → PandaGripper 结论成立? "
          f"{audit['comparison']['eco65_zhixing_vs_eco65_panda']['conclusion_holds']}")
    print("\n=== 腕部占比与对齐选项 ===")
    for k, r in rows.items():
        print(f"  {k:<14s} ratio={r['ratio']:.3f}  →  相机后移 {r['equivalent_camera_pullback_mm']:.1f} mm"
              f"  或 fovy 75°→{r['equivalent_fovy_deg']:.1f}° (+{r['fovy_delta_deg']:.1f}°)")
    print(f"\n[out] {RESULTS/'tcp_camera_contract.json'}")
    print(f"[out] {RESULTS/'camera_alignment_plan.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
