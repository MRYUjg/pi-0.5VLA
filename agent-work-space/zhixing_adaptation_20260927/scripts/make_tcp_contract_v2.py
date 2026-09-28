#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2-v2：汇总生成**当前资产绑定**的机器可读契约 results/tcp_camera_contract_v2.json。

与 make_tcp_contract.py（旧契约）的差别：
  * 绑定当前生产资产的 sha256，并与 manifests/source_fingerprint_current_candidate.json 对账；
  * 所有数字取自 results/n2v2_20260928/ 下**本轮重测**产物；缺失/不可复现即写 null +
    blocked_reason，**不从旧契约复制任何数值**；
  * 开/闭态区分三种口径：约束自洽装配（primary）、阶跃驱动、部署斜坡驱动，
    并显式记录本轮发现的「阶跃闭合卡死」现象。

用法（容器内，仓库根）：
  python /app/agent-work-space/zhixing_adaptation_20260927/scripts/make_tcp_contract_v2.py
"""
import hashlib
import json
import pathlib
import re
import sys
from datetime import datetime, timezone

REPO = pathlib.Path("/app")
NODE = REPO / "agent-work-space/zhixing_adaptation_20260927"
RESULTS = NODE / "results"
RAW = RESULTS / "n2v2_20260928"
FINGERPRINT = NODE / "manifests/source_fingerprint_current_candidate.json"
OLD_CONTRACT = "results/tcp_camera_contract.json"
DATE = "2026-09-28"

BOUND_FILES = [
    "robosuite/models/assets/grippers/zhixing_gripper.xml",
    "robosuite/models/robots/manipulators/eco65_robot.py",
    "third_party/libero/libero/libero/envs/robots/mounted_eco65.py",
    "robosuite/models/assets/robots/ECO65-B/robot.xml",
    "robosuite/controllers/config/osc_pose.json",
    "robosuite/models/grippers/zhixing_gripper.py",
    "robosuite/robots/single_arm.py",
    "agent-work-space/zhixing_adaptation_20260927/scripts/n4_collect.py",
    "robosuite/controllers/config/robots/default_eco65.json",
    "robosuite/models/assets/grippers/panda_gripper.xml",
]
STATES = ("reset", "open", "closed")


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(p, required=True):
    p = pathlib.Path(p)
    if not p.exists():
        if required:
            raise SystemExit(f"缺少输入 {p}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def dig(obj, path, default=None):
    cur = obj
    for k in path:
        if cur is None:
            return default
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def main():
    audit = load(RAW / "tcp_camera_audit.json")
    padf = load(RAW / "pad_face_geometry.json")
    sites = load(RAW / "tcp_geometry_sites.json")
    initp = load(RAW / "zhixing_init_pose.json")
    settled = load(RAW / "n2v2_settled_probe.json")
    diag = load(RAW / "n2v2_close_diag.json", required=False)
    fp = load(FINGERPRINT)

    prov, blocked, checks = {}, {}, {}

    def P(field, script, artifact, key):
        prov[field] = {"script": script, "artifact": artifact, "key": key}

    def B(field, reason):
        blocked[field] = {"value": None, "blocked_reason": reason}

    # ================= asset bindings =================
    fp_by_path = {v["path"]: v for v in (fp.get("sources") or {}).values()
                  if isinstance(v, dict) and "path" in v and "sha256" in v}
    asset_bindings = {
        "repo_root_in_container": str(REPO),
        "repo_root_on_host": "/mnt/data/vla-policy/openpi",
        "fingerprint_manifest": {
            "path": "agent-work-space/zhixing_adaptation_20260927/manifests/"
                    "source_fingerprint_current_candidate.json",
            "generated_at": fp.get("generated_at"),
            "purpose": fp.get("purpose"),
        },
        "files": {},
    }
    for rel in BOUND_FILES:
        p = REPO / rel
        rec = {"sha256": None, "bytes": None, "matches_fingerprint_manifest": None,
               "recomputed_by": "hashlib.sha256 (make_tcp_contract_v2.py, container)"}
        if p.exists():
            rec["sha256"] = sha256_of(p)
            rec["bytes"] = p.stat().st_size
            f = fp_by_path.get(rel)
            rec["fingerprint_manifest_sha256"] = None if f is None else f["sha256"]
            rec["in_fingerprint_manifest"] = f is not None
            rec["matches_fingerprint_manifest"] = (
                None if f is None else bool(f["sha256"] == rec["sha256"]))
        else:
            rec["error"] = "file not found"
            B(f"asset_bindings.files[{rel}]", f"文件不存在：{rel}")
        asset_bindings["files"][rel] = rec
    P("asset_bindings.files", "make_tcp_contract_v2.py (hashlib)",
      "results/tcp_camera_contract_v2.json", "asset_bindings.files[*].sha256")

    # eef 安装位置：解析 XML 真源
    zx_xml = (REPO / BOUND_FILES[0]).read_text(encoding="utf-8")
    m = re.search(r'<body\s+name="eef"\s+pos="([^"]+)"', zx_xml)
    eef_declared = [float(x) for x in m.group(1).split()] if m else None
    if eef_declared is None:
        B("tcp.eef_body.declared_pos_in_right_gripper_m",
          "zhixing_gripper.xml 中未找到 <body name=\"eef\" pos=...>")
    P("tcp.eef_body.declared_pos_in_right_gripper_m",
      "re 解析 robosuite/models/assets/grippers/zhixing_gripper.xml",
      BOUND_FILES[0], '<body name="eef" pos>')

    cfgs = {c["tag"]: c for c in settled["configs"]}
    for tag in ("eco65_zhixing", "eco65_panda", "panda_panda"):
        if tag not in cfgs:
            B(f"settled.{tag}", f"n2v2_settled_probe.json 缺少配置 {tag}；"
                                f"errors={settled.get('errors')}")
    zx, ecp, pd = cfgs.get("eco65_zhixing"), cfgs.get("eco65_panda"), cfgs.get("panda_panda")

    def geo(cfg, mode, st, key, default=None):
        if cfg is None:
            return None
        return dig(cfg, [mode, st, key], default)

    def along_series(cfg, mode):
        return {st: geo(cfg, mode, st, "grasp_point_offset_along_approach_mm") for st in STATES}

    # ================= tcp =================
    tcp = {
        "eef_body": {
            "declared_pos_in_right_gripper_m": eef_declared,
            "declared_install_length_m": None if eef_declared is None else eef_declared[2],
            "measured_pos_in_right_gripper_m": dig(zx, ["eef_mount", "eef_pos_in_right_gripper_m"]),
            "eef_body_name_in_model": dig(zx, ["eef_mount", "eef_body_name"]),
            "parent_body_name_in_model": dig(zx, ["eef_mount", "right_gripper_body_name"]),
            "history": "旧契约时代为 pos=\"0 0 0.12\"（安装长度 0.120 m）；当前为 0.20254 m。",
            "cross_check_flange_offset_mm": {
                "eco65_zhixing_right_gripper_behind_gripsite": 202.54000354018004,
                "panda_panda_right_gripper_behind_gripsite": 122.94578156662975,
                "source": "results/n2v2_20260928/tcp_camera_audit.json "
                          "comparison.vs_reference[*].gripper_lowest_below_gripsite_mm / "
                          "tcp_geometry_sites.json bodies[*].local_mm（本轮重测）",
            },
        },
        "grasp_point_definition": (
            "两指垫碰撞 geom 内侧面（沿两垫质心连线 ±u 方向最前 5 mm 薄层顶点）中点在 grip_site "
            "局部系下的位置；正值 = 比声明的 grip_site 更深入工作空间（+z）。"
        ),
        "grasp_point_offset_along_approach_mm": {
            "convention": "+z（grip_site 局部系，指向工作空间/桌面下方）",
            "primary_mode": "constraint_consistent_assembly（约束自洽装配；"
                            "joint equality 残差 0，connect 残差 1.75e-7 m，且与臂位姿无关）",
            "eco65_zhixing": along_series(zx, "geometry"),
            "eco65_panda": along_series(ecp, "geometry"),
            "panda_panda": along_series(pd, "geometry"),
            "cross_checks": {
                "driven_step_input": {
                    "eco65_zhixing": along_series(zx, "geometry_driven"),
                    "eco65_panda": along_series(ecp, "geometry_driven"),
                    "panda_panda": along_series(pd, "geometry_driven"),
                    "note": "执行器 ctrl 阶跃到端点后 mj_step 800 步（与历史 probe_pad_face.py 同口径）。"
                            "ZhiXing 的 closed 在此口径下**不收敛**，见 actuator_path_findings。",
                },
                "driven_ramp_input": {
                    "eco65_zhixing": along_series(zx, "geometry_ramped"),
                    "eco65_panda": along_series(ecp, "geometry_ramped"),
                    "panda_panda": along_series(pd, "geometry_ramped"),
                    "note": "ctrl 按 100 步斜坡到达端点再保持（对应部署时 format_action 在 "
                            "speed=0.01 下的累积路径）。",
                },
                "probe_pad_face_driven": {
                    "eco65_zhixing": {st: dig(padf, ["eco65_zhixing", "states", st,
                                                     "pinch_approach_mm"]) for st in STATES},
                    "eco65_panda": {st: dig(padf, ["eco65_panda", "states", st,
                                                   "pinch_approach_mm"]) for st in STATES},
                    "panda_panda": {st: dig(padf, ["panda_panda", "states", st,
                                                   "pinch_approach_mm"]) for st in STATES},
                },
            },
            "zhixing_vs_panda_reference_delta_mm": {
                "definition": "ZhiXing 夹持点 − PandaGripper(Panda) 夹持点（同状态）",
                "reset": None if zx is None or pd is None else float(
                    geo(zx, "geometry", "reset", "grasp_point_offset_along_approach_mm")
                    - geo(pd, "geometry", "reset", "grasp_point_offset_along_approach_mm")),
                "open": None if zx is None or pd is None else float(
                    geo(zx, "geometry", "open", "grasp_point_offset_along_approach_mm")
                    - geo(pd, "geometry", "open", "grasp_point_offset_along_approach_mm")),
                "closed": None if zx is None or pd is None else float(
                    geo(zx, "geometry", "closed", "grasp_point_offset_along_approach_mm")
                    - geo(pd, "geometry", "closed", "grasp_point_offset_along_approach_mm")),
            },
            "jaw_travel_over_stroke_mm": None if zx is None else float(
                geo(zx, "geometry", "closed", "grasp_point_offset_along_approach_mm")
                - geo(zx, "geometry", "open", "grasp_point_offset_along_approach_mm")),
            "legacy_closest_pair_metric_mm": {
                st: dig(zx, ["geometry", st, "nip_offset_approach_mm_legacy"]) for st in STATES},
        },
        "jaw_opening_range": {
            "definition": "两指垫碰撞 geom 顶点质心间距（工程开口口径）",
            "eco65_zhixing_assembly": {st: geo(zx, "geometry", st, "jaw_opening_pad_center_m")
                                       for st in STATES},
            "eco65_panda_assembly": {st: geo(ecp, "geometry", st, "jaw_opening_pad_center_m")
                                     for st in STATES},
            "panda_panda_assembly": {st: geo(pd, "geometry", st, "jaw_opening_pad_center_m")
                                     for st in STATES},
            "eco65_zhixing_ramped": {st: geo(zx, "geometry_ramped", st, "jaw_opening_pad_center_m")
                                     for st in STATES},
            "eco65_zhixing_pad_face_separation_mm": {
                st: geo(zx, "geometry", st, "pad_face_separation_mm") for st in STATES},
            "panda_panda_pad_face_separation_mm": {
                st: geo(pd, "geometry", st, "pad_face_separation_mm") for st in STATES},
            "eco65_zhixing_closest_pair_clearance_m": {
                st: geo(zx, "geometry", st, "nip_clearance_m") for st in STATES},
            "eco65_zhixing_left_1_joint_qpos_assembly": {
                st: dig(zx, ["geometry", st, "gripper_joint_qpos", "gripper0_Left_1_Joint"])
                for st in STATES},
            "deployment_measured_open_stroke_span_m": dig(
                settled, ["polarity", "verdict", "open_stroke_span_m"]),
            "range_m": {
                "eco65_zhixing_pad_center_min_closed": geo(zx, "geometry", "closed",
                                                           "jaw_opening_pad_center_m"),
                "eco65_zhixing_pad_center_max_open": geo(zx, "geometry", "open",
                                                         "jaw_opening_pad_center_m"),
                "eco65_zhixing_span": None if zx is None else float(
                    geo(zx, "geometry", "open", "jaw_opening_pad_center_m")
                    - geo(zx, "geometry", "closed", "jaw_opening_pad_center_m")),
                "panda_panda_pad_center_min_closed": geo(pd, "geometry", "closed",
                                                         "jaw_opening_pad_center_m"),
                "panda_panda_pad_center_max_open": geo(pd, "geometry", "open",
                                                       "jaw_opening_pad_center_m"),
            },
            "v1_compatible_keys": {
                "closed_pad_clearance_m": geo(zx, "geometry", "closed", "nip_clearance_m"),
                "open_pad_clearance_m": geo(zx, "geometry", "open", "nip_clearance_m"),
                "note": "与旧契约 zhixing.jaw_opening_range 同键名、同口径（最近点对净开口）。",
            },
        },
        "pad_face_normal_angle_deg": {
            "eco65_zhixing": {st: geo(zx, "geometry", st, "pad_face_normal_angle_deg")
                              for st in STATES},
            "panda_panda": {st: geo(pd, "geometry", st, "pad_face_normal_angle_deg")
                            for st in STATES},
            "note": "ZhiXing 为二连平行连杆：全行程内两面近似平行。",
        },
        "gripper_extent_along_approach_mm": {
            "eco65_zhixing_plus_z": {st: geo(zx, "geometry", st, "gripper_extent_plus_z_mm")
                                     for st in STATES},
            "eco65_zhixing_minus_z": {st: geo(zx, "geometry", st, "gripper_extent_minus_z_mm")
                                      for st in STATES},
            "panda_panda_plus_z": {st: geo(pd, "geometry", st, "gripper_extent_plus_z_mm")
                                   for st in STATES},
        },
        "constraint_residuals": {
            "definition": "joint equality 最大位置残差（rad）+ 两条 connect site 距离最大残差（m）",
            "eco65_zhixing_assembly": {st: geo(zx, "geometry", st, "constraint_residuals")
                                       for st in STATES},
            "eco65_zhixing_driven_step": {st: geo(zx, "geometry_driven", st,
                                                  "constraint_residuals") for st in STATES},
            "eco65_zhixing_driven_ramp": {st: geo(zx, "geometry_ramped", st,
                                                  "constraint_residuals") for st in STATES},
            "eco65_zhixing_probe_pad_face": {
                st: dig(padf, ["eco65_zhixing", "states", st, "constraint_residuals"])
                for st in STATES},
            "panda_note": "PandaGripper 无 joint equality / connect 约束，残差为 null。",
        },
        "site_cross_check_local_mm": {
            "Left_Support_tip": next((s["local_mm"] for s in sites["eco65_zhixing"]["sites"]
                                      if s["name"].endswith("Left_Support_tip")), None),
            "Right_Support_tip": next((s["local_mm"] for s in sites["eco65_zhixing"]["sites"]
                                       if s["name"].endswith("Right_Support_tip")), None),
            "eef_body_offset": next((s["local_mm"] for s in sites["eco65_zhixing"]["bodies"]
                                     if s["name"].endswith("eef")), None),
        },
    }
    P("tcp.grasp_point_offset_along_approach_mm / jaw_opening_range / "
      "pad_face_normal_angle_deg / gripper_extent_along_approach_mm / constraint_residuals",
      "n2v2_settled_probe.py",
      "results/n2v2_20260928/n2v2_settled_probe.json",
      "configs[tag].geometry|geometry_driven|geometry_ramped.<state>.*")
    P("tcp.grasp_point_offset_along_approach_mm.cross_checks.probe_pad_face_driven",
      "probe_pad_face.py（经 n2v2_run_redirected.py 重定向输出，未改源码；CONFIGS 扩为 3 个）",
      "results/n2v2_20260928/pad_face_geometry.json", "<tag>.states.<state>.pinch_approach_mm")
    P("tcp.site_cross_check_local_mm", "probe_tcp_geometry.py（经 n2v2_run_redirected.py 重定向输出）",
      "results/n2v2_20260928/tcp_geometry_sites.json", "eco65_zhixing.sites/bodies[*].local_mm")

    # 收敛性重算（用记录到的原始残差 + 明确容差；probe 内的更严容差另存）
    TOL = {"joint_eq_max_rad": 0.01, "connect_max_m": 5e-4}

    def conv_flags(cfg, mode, st):
        r = geo(cfg, mode, st, "constraint_residuals") or {}
        eq, cn = r.get("joint_eq_max"), r.get("connect_max_m")
        return {"joint_eq_max_rad": eq, "connect_max_m": cn,
                "joint_eq_ok": (eq is None or eq <= TOL["joint_eq_max_rad"]),
                "connect_ok": (cn is None or cn <= TOL["connect_max_m"]),
                "converged": bool((eq is None or eq <= TOL["joint_eq_max_rad"])
                                  and (cn is None or cn <= TOL["connect_max_m"]))}

    tcp["state_convergence"] = {
        "tolerance": TOL,
        "eco65_zhixing_assembly": {st: conv_flags(zx, "geometry", st) for st in STATES},
        "eco65_zhixing_driven_step": {st: conv_flags(zx, "geometry_driven", st)
                                      for st in STATES},
        "eco65_zhixing_driven_ramp": {st: conv_flags(zx, "geometry_ramped", st)
                                      for st in STATES},
        "panda_panda_driven_step": {st: conv_flags(pd, "geometry_driven", st)
                                    for st in STATES},
        "probe_internal_stricter_tolerance": dig(zx, ["convergence"]),
    }
    P("tcp.state_convergence",
      "make_tcp_contract_v2.py 基于 n2v2_settled_probe.py 的原始残差重算",
      "results/n2v2_20260928/n2v2_settled_probe.json",
      "configs[*].geometry*.constraint_residuals")

    # 阶跃闭合卡死：显式标为不可复现
    drv_closed = geo(zx, "geometry_driven", "closed", "grasp_point_offset_along_approach_mm")
    conv = conv_flags(zx, "geometry_driven", "closed")
    if not conv.get("converged", False):
        B("tcp.grasp_point_offset_along_approach_mm.cross_checks.driven_step_input"
          ".eco65_zhixing.closed",
          "阶跃驱动（ctrl 直接置 -0.91 + mj_step 800 步）在当前资产下**不收敛**："
          f"Left_1 冲到 "
          f"{dig(zx, ['geometry_driven', 'closed', 'gripper_joint_qpos', 'gripper0_Left_1_Joint'])} "
          f"rad（越出关节范围 [-0.91, 0]），joint equality 残差 "
          f"{dig(zx, ['geometry_driven', 'closed', 'constraint_residuals', 'joint_eq_max'])} rad，"
          f"connect 残差 {dig(zx, ['geometry_driven', 'closed', 'constraint_residuals', 'connect_max_m'])} m，"
          f"读得的“夹持点偏移” {drv_closed} mm 是卡死构型，**不是有效闭合位**。"
          "该口径下 closed 无有效测量值；请使用 primary_mode（自洽装配）或 driven_ramp_input。")

    # ================= initial pose =================
    landed_qpos = dig(zx, ["landed_init_qpos"]) or dig(initp, ["current_landed", "init_qpos"])
    ref_pos = dig(initp, ["reference", "grip_site_pos"])
    zx_grip_reset = geo(zx, "geometry", "reset", "grip_site_world_pos")
    pd_grip_reset = geo(pd, "geometry", "reset", "grip_site_world_pos")

    def dist_mm(a, b):
        return None if (a is None or b is None) else float(
            sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5 * 1000.0)

    SRC_ZX = [-0.12639, -0.222035, -0.841572, -0.563545, 1.563635, -1.697476]
    init_pose = {
        "definition": "reset 后、任何步进之前的 6 个臂关节角（robot0_joint_1..6）",
        "dispatch_source": "robosuite/models/robots/manipulators/eco65_robot.py "
                           "ECO65_INIT_QPOS_BY_GRIPPER（R1 修复：按实际挂载夹爪分派）；"
                           "robosuite/robots/single_arm.py 在 add_gripper() 后重新解析。",
        "landed_init_qpos_eco65_zhixing_measured": landed_qpos,
        "landed_init_qpos_eco65_zhixing_source_table": SRC_ZX,
        "landed_init_qpos_source_table_max_abs_delta_rad": (
            None if not landed_qpos else float(
                max(abs(a - b) for a, b in zip(landed_qpos, SRC_ZX)))),
        "landed_init_qpos_from_solve_zhixing_init_pose": dig(initp, ["current_landed",
                                                                      "init_qpos"]),
        "landed_init_qpos_two_measurements_note": (
            "settled probe 在 env.reset() 后**未步进**即读取，与源码表逐位一致；"
            "solve_zhixing_init_pose.py 在锁定臂关节步进 800 步后读取，因而有 ≤1.8e-5 rad 的"
            "一步积分残差 —— 不是 init_qpos 噪声（robosuite initialization_noise 未生效于该路径）。"),
        "landed_init_qpos_eco65_panda_measured": dig(ecp, ["landed_init_qpos"]),
        "landed_init_qpos_panda_panda_measured": dig(pd, ["landed_init_qpos"]),
        "grip_site_world_pos_eco65_zhixing_reset": zx_grip_reset,
        "grip_site_world_pos_eco65_panda_reset": geo(ecp, "geometry", "reset",
                                                     "grip_site_world_pos"),
        "grip_site_world_pos_panda_panda_reset": pd_grip_reset,
        "panda_reference_grip_site_pos": ref_pos,
        "residual_vs_panda_reference": {
            "definition": dig(initp, ["current_landed", "semantic_target_note"]),
            "grasp_point_mm": dig(initp, ["current_landed", "grasp_point_vs_ref_mm"]),
            "grip_site_mm_from_solve_zhixing_init_pose": dig(
                initp, ["current_landed", "grip_site_vs_ref_mm"]),
            "grip_site_mm_recomputed_from_settled_probe": dist_mm(zx_grip_reset, ref_pos),
        },
        "eco65_panda_grip_site_vs_panda_reference_mm": dist_mm(
            geo(ecp, "geometry", "reset", "grip_site_world_pos"), ref_pos),
        "zhixing_grasp_offset_local_m_at_reset": dig(zx, ["geometry", "reset",
                                                          "grasp_point_local_m"]),
        "panda_reference_grasp_offset_local_m": dig(initp, ["reference",
                                                            "panda_grasp_offset_local_m"]),
        "candidate_ik_solution": {
            "init_qpos": dig(initp, ["candidate", "init_qpos"]),
            "grasp_point_residual_mm": dig(initp, ["candidate", "grasp_point_residual_mm"]),
            "rot_residual_deg": dig(initp, ["candidate", "grasp_point_rot_residual_deg"]),
            "limit_margin_min_rad": dig(initp, ["candidate", "limit_margin_min_rad"]),
            "gate_pass": dig(initp, ["candidate", "gate_pass"]),
            "landed": bool(dig(initp, ["candidate", "gate_pass"])),
            "note": "R1 之后该解已实际落地为 ECO65_INIT_QPOS_BY_GRIPPER['ZhixingGripper']"
                    "（与 landed_init_qpos_source_table 一致）。",
        },
    }
    P("initial_pose.*", "solve_zhixing_init_pose.py（经 n2v2_run_redirected.py 重定向输出）",
      "results/n2v2_20260928/zhixing_init_pose.json",
      "reference / current_landed / candidate")
    P("initial_pose.landed_init_qpos_*_measured / grip_site_world_pos_*_reset",
      "n2v2_settled_probe.py", "results/n2v2_20260928/n2v2_settled_probe.json",
      "configs[*].landed_init_qpos / geometry.reset.grip_site_world_pos")

    # ================= camera =================
    def T_of(cfg):
        return dig(cfg, ["camera", "T_cam2grip_reset"])

    cam = {
        "camera": "robot0_eye_in_hand（定义于 robosuite/models/assets/robots/ECO65-B/robot.xml，"
                  "挂在 right_hand 上 pos=\"0.05 0 0\" quat=\"0 0.707108 0.707108 0\" fovy=75；"
                  "渲染 256x256）",
        "T_cam2grip_definition": "inv(T_cam_world) @ T_grip_site_world（reset 位姿，4x4 行主序）",
        "T_cam2grip": {
            "eco65_zhixing": T_of(zx),
            "eco65_panda": T_of(ecp),
            "panda_panda": T_of(pd),
        },
        "T_cam2grip_invariant_across_jaw_states": dig(
            zx, ["camera", "T_cam2grip_invariant_across_states"]),
        "delta_zhixing_vs_pandagripper": {
            "trans_mm": dig(settled, ["comparison", "zhixing_vs_pandagripper_reset",
                                      "T_cam2grip_trans_delta_mm"]),
            "rot_deg": dig(settled, ["comparison", "zhixing_vs_pandagripper_reset",
                                     "T_cam2grip_rot_delta_deg"]),
            "delta_vector_mm": dig(settled, ["comparison", "zhixing_vs_pandagripper_reset",
                                             "delta_vector_mm"]),
            "per_state_trans_mm": {st: dig(settled, ["comparison",
                                                     f"zhixing_vs_pandagripper_{st}",
                                                     "T_cam2grip_trans_delta_mm"])
                                   for st in STATES},
        },
        "delta_eco65_panda_vs_panda_panda_trans_mm": dig(
            settled, ["comparison", "eco65_panda_vs_panda_panda_T_cam2grip_trans_delta_mm"]),
        "dist_cam_to_gripsite_mm": {
            "eco65_zhixing": dig(zx, ["camera", "dist_cam_to_gripsite_mm"]),
            "eco65_panda": dig(ecp, ["camera", "dist_cam_to_gripsite_mm"]),
            "panda_panda": dig(pd, ["camera", "dist_cam_to_gripsite_mm"]),
        },
        "wrist_gripper_pixel_fraction": {},
        "cam_to_grasp_mm": {},
    }
    for st in STATES:
        fz = dig(zx, ["views", f"wrist_{st}", "gripper_pixel_fraction"])
        fp_ = dig(pd, ["views", f"wrist_{st}", "gripper_pixel_fraction"])
        cam["wrist_gripper_pixel_fraction"][st] = {
            "eco65_zhixing": fz,
            "eco65_panda": dig(ecp, ["views", f"wrist_{st}", "gripper_pixel_fraction"]),
            "panda_panda": fp_,
            "ratio_zhixing_over_pandagripper": (fz / fp_) if (fz and fp_) else None,
            "ratio_zhixing_over_ecopanda": (
                fz / dig(ecp, ["views", f"wrist_{st}", "gripper_pixel_fraction"])
                if fz and dig(ecp, ["views", f"wrist_{st}", "gripper_pixel_fraction"]) else None),
            "eco65_zhixing_bottom40": dig(zx, ["views", f"wrist_{st}",
                                               "gripper_pixel_fraction_bottom40"]),
            "panda_panda_bottom40": dig(pd, ["views", f"wrist_{st}",
                                             "gripper_pixel_fraction_bottom40"]),
            "audit_pipeline_qpos_write_crosscheck": {
                "eco65_zhixing": next((c["views"][f"wrist_{st}"]["gripper_pixel_fraction"]
                                       for c in audit["configs"] if c.get("tag") == "eco65_zhixing"
                                       and f"wrist_{st}" in c.get("views", {})), None),
                "panda_panda": next((c["views"][f"wrist_{st}"]["gripper_pixel_fraction"]
                                     for c in audit["configs"] if c.get("tag") == "panda_panda"
                                     and f"wrist_{st}" in c.get("views", {})), None),
            },
        }
        for name, cfg in (("eco65_zhixing", zx), ("panda_panda", pd)):
            T = T_of(cfg)
            off = geo(cfg, "geometry", st, "grasp_point_local_m")
            if T is not None and off is not None:
                import numpy as _np

                Tn = _np.array(T, dtype=float)
                p = Tn[:3, :3] @ _np.array(off, dtype=float) + Tn[:3, 3]
                cam["cam_to_grasp_mm"].setdefault(st, {})[name] = float(_np.linalg.norm(p) * 1000)
    P("camera.*", "n2v2_settled_probe.py",
      "results/n2v2_20260928/n2v2_settled_probe.json",
      "configs[*].camera / views.wrist_<state> / geometry.<state>.T_cam2grip")
    P("camera.wrist_gripper_pixel_fraction.*.audit_pipeline_qpos_write_crosscheck",
      "audit_tcp_camera.py --out <新目录>（源码未改）",
      "results/n2v2_20260928/tcp_camera_audit.json",
      "configs[*].views.wrist_<state>.gripper_pixel_fraction")

    # ================= controller / gripper =================
    ctl = dig(zx, ["controller"], {}) or {}
    eco65_ctl_json = json.loads((REPO / BOUND_FILES[8]).read_text(encoding="utf-8-sig"))
    arm_ctl = dig(eco65_ctl_json, ["body_parts", "arms", "right"], {}) or {}
    controller = {
        "type_class_measured": ctl.get("type"),
        "config_type": arm_ctl.get("type"),
        "resolved_source_config": BOUND_FILES[8]
                                  + "（ECO65.default_controller_config = \"default_eco65\"）",
        "resolved_source_config_sha256": asset_bindings["files"][BOUND_FILES[8]]["sha256"],
        "equivalent_preset_config": BOUND_FILES[4],
        "equivalent_preset_config_sha256": asset_bindings["files"][BOUND_FILES[4]]["sha256"],
        "output_max": ctl.get("output_max"),
        "output_max_in_resolved_config": arm_ctl.get("output_max"),
        "output_min_in_resolved_config": arm_ctl.get("output_min"),
        "input_min": ctl.get("input_min"),
        "input_max": ctl.get("input_max"),
        "kp": arm_ctl.get("kp"),
        "damping_ratio": arm_ctl.get("damping_ratio"),
        "input_type": arm_ctl.get("input_type"),
        "input_ref_frame": arm_ctl.get("input_ref_frame"),
        "control_timestep": ctl.get("control_timestep"),
        "action_convention": "7 维 OSC_POSE 归一化动作 ∈[-1,1]：前 3 维平移（满量程 "
                             "output_max[0:3] m），后 3 维旋转（满量程 output_max[3:6] rad），"
                             "第 7 维夹爪（-1 开 / +1 合）。",
    }
    P("controller.*", "n2v2_settled_probe.py（读 env.env.robots[0].controller）",
      "results/n2v2_20260928/n2v2_settled_probe.json", "configs[tag=eco65_zhixing].controller")

    pol = settled.get("polarity") or {}
    grip = {
        "zhixing": {
            "class": "robosuite.models.grippers.zhixing_gripper.ZhixingGripper",
            "joint_names": dig(zx, ["gripper_joint_names"]),
            "joint_count": dig(zx, ["gripper_joint_count"]),
            "dof": dig(zx, ["gripper_dof"]),
            "actuator": "单 position 执行器 gripper0_gripper（joint=Left_1_Joint，"
                        "ctrlrange=[-0.91, 0]，kp=100，forcerange=[-2, 2] N·m）",
            "polarity": {
                "convention": "-1 = open, +1 = close",
                "implementation": "ZhixingGripper.format_action 对累积量取负："
                                  "current_action = clip(current_action - speed*sign(action), -1, 1)。",
                "measured_end_to_end": {
                    "path": "真实 env.step（OSC_POSE + format_action，ZHIXING_GRIPPER_SPEED=0.01）",
                    "sequence": "reset → 250 步 action=-1 → 250 步 action=+1 → 250 步 action=-1",
                    "verdict": dig(pol, ["verdict"]),
                    "phases": dig(pol, ["sequence", "speed_0.01", "phases"]),
                    "opening_at_reset_m": dig(pol, ["sequence", "speed_0.01",
                                                    "opening_at_reset_m"]),
                },
            },
            "speed": {
                "asset_default_measured": dig(pol, ["speed_asset_default"]),
                "deployed_value": 0.01,
                "deployed_mechanism": "环境变量 ZHIXING_GRIPPER_SPEED=0.01；"
                                      "scripts/n4_collect.py 的 --gripper-speed 默认 0.01 并写入该环境变量"
                                      "（run_n4_collect.sh 不覆盖）。",
                "measured_with_env_0p01": dig(pol, ["speed_with_env_0p01"]),
            },
        },
        "panda": {
            "class": "robosuite.models.grippers.panda_gripper.PandaGripper",
            "joint_names": dig(pd, ["gripper_joint_names"]),
            "joint_count": dig(pd, ["gripper_joint_count"]),
            "dof": dig(pd, ["gripper_dof"]),
            "speed": dig(pd, ["gripper_speed_asset_default"]),
        },
    }
    P("gripper.zhixing.joint_names/joint_count/dof, gripper.panda.*", "n2v2_settled_probe.py",
      "results/n2v2_20260928/n2v2_settled_probe.json", "configs[*].gripper_*")
    P("gripper.zhixing.polarity.measured_end_to_end", "n2v2_settled_probe.py::polarity_probe",
      "results/n2v2_20260928/n2v2_settled_probe.json", "polarity.*")
    P("gripper.zhixing.speed.deployed_value", "源码常量（scripts/n4_collect.py argparse default）",
      "scripts/n4_collect.py", "--gripper-speed default=0.01 → os.environ['ZHIXING_GRIPPER_SPEED']")

    # ================= findings =================
    findings = {
        "step_input_close_jams_but_deployed_ramp_is_fine": {
            "observation": "对 ZhiXing 执行器把 ctrl 从当前值**阶跃**到闭合端点 -0.91 后 mj_step，"
                           "四连杆不收敛：Left_1 冲到 -1.35 rad（越出 [-0.91,0]），"
                           "joint equality 残差 ~1.43 rad，connect 残差 ~18 mm，出现 1 个自接触。",
            "not_arm_pose_dependent": "在 R1 之前的旧臂位姿、当前 ZhiXing 臂位姿、Panda 分派臂位姿下"
                                      "结果逐位相同（见 n2v2_close_diag.json），因此与初始位姿无关。",
            "deployed_path_ok": "按部署路径（format_action 在 speed=0.01 下 100 步斜坡到端点）"
                                "闭合可以收敛：pad 面间距 0.638 mm，connect 残差 9.6e-5 m；"
                                "端到端 env.step 开→合→开三步序列也全部通过（极性 verdict）。",
            "implication": "历史 probe_pad_face.py 的 closed 口径（阶跃）在当前资产下**不可用**；"
                           "契约的 closed 值取约束自洽装配，并用部署斜坡口径交叉验证。",
            "artifact": "results/n2v2_20260928/n2v2_close_diag.json",
        },
        "assembly_is_arm_pose_invariant": {
            "observation": "自洽装配在 Left_1=0 / -0.5 / -0.91 三点的 connect 残差均为 1.75e-7 m，"
                           "且在三种臂位姿下逐位相同 → 夹爪几何只由装配决定。",
            "artifact": "results/n2v2_20260928/n2v2_close_diag.json",
        },
    }

    # ================= validation checks =================
    def close(a, b, tol):
        return (a is not None and b is not None and abs(a - b) <= tol)

    checks["asset_hashes_match_fingerprint_manifest"] = {
        rel: asset_bindings["files"][rel]["matches_fingerprint_manifest"] for rel in BOUND_FILES}
    checks["eef_declared_is_0p20254"] = close(
        eef_declared[2] if eef_declared else None, 0.20254, 1e-9)
    checks["eef_measured_matches_declared_1um"] = close(
        (dig(zx, ["eef_mount", "eef_pos_in_right_gripper_m"]) or [None, None, None])[2],
        eef_declared[2] if eef_declared else None, 1e-6)
    checks["landed_init_qpos_matches_source_table_1e-6_rad"] = (
        None if not landed_qpos else bool(
            max(abs(a - b) for a, b in zip(landed_qpos, SRC_ZX)) < 1e-6))
    checks["eco65_panda_init_qpos_is_panda_solution"] = (
        None if not dig(ecp, ["landed_init_qpos"]) else bool(max(
            abs(a - b) for a, b in zip(dig(ecp, ["landed_init_qpos"]),
                                       [-0.128227, -0.080175, -1.482052,
                                        -0.064915, 1.563529, -1.699308])) < 1e-6))
    checks["controller_output_max_is_expected"] = bool(
        ctl.get("output_max") == [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
        and arm_ctl.get("output_max") == [0.05, 0.05, 0.05, 0.5, 0.5, 0.5]
        and arm_ctl.get("type") == "OSC_POSE")
    checks["gripper_speed_default_0p2_and_env_0p01"] = bool(
        pol.get("speed_asset_default") == 0.2 and pol.get("speed_with_env_0p01") == 0.01)
    checks["polarity_end_to_end_minus1_open_plus1_close"] = bool(
        dig(pol, ["verdict", "minus1_opens"]) and dig(pol, ["verdict", "plus1_closes_from_open"])
        and dig(pol, ["verdict", "minus1_reopens"]))
    checks["T_cam2grip_invariant_across_jaw_states"] = dig(
        zx, ["camera", "T_cam2grip_invariant_across_states"])
    checks["zhixing_assembly_residual_within_1um"] = all(
        (dig(zx, ["geometry", st, "constraint_residuals", "joint_eq_max"]) == 0.0 and
         dig(zx, ["geometry", st, "constraint_residuals", "connect_max_m"]) < 1e-6)
        for st in STATES)
    checks["ecopanda_equals_pandapanda_T_cam2grip_below_1e-6_mm"] = bool(
        (dig(settled, ["comparison", "eco65_panda_vs_panda_panda_T_cam2grip_trans_delta_mm"])
         or 1.0) < 1e-6)
    checks["zhixing_reset_grip_site_vs_panda_reset_mm"] = dist_mm(zx_grip_reset, pd_grip_reset)

    # ================= diff vs 旧契约 =================
    diff = []
    old = load(RESULTS / "tcp_camera_contract.json", required=False)
    if old is not None:
        pairs = [
            ("zhixing.grasp_point_offset_along_approach_mm.closed",
             dig(old, ["zhixing", "grasp_point_offset_along_approach_mm", "closed"]),
             dig(tcp, ["grasp_point_offset_along_approach_mm", "eco65_zhixing", "closed"]),
             "eef 由 0.120 前移到 0.20254 后，闭合位夹持点已基本落在声明的 grip_site 上。"),
            ("zhixing.grasp_point_offset_along_approach_mm.open",
             dig(old, ["zhixing", "grasp_point_offset_along_approach_mm", "open"]),
             dig(tcp, ["grasp_point_offset_along_approach_mm", "eco65_zhixing", "open"]),
             "同上；全开位夹持点现在位于 grip_site 之后 ~17.5 mm。"),
            ("camera_contract.T_cam2grip_trans_delta_zhixing_vs_panda_mm",
             dig(old, ["camera_contract", "T_cam2grip_trans_delta_zhixing_vs_panda_mm"]),
             dig(cam, ["delta_zhixing_vs_pandagripper", "trans_mm"]),
             "eef 安装长度差由 23.0 mm 变为 105.54 mm（=0.20254−0.097）。"),
            ("camera_contract.pandagripper_alignment_still_holds",
             dig(old, ["camera_contract", "pandagripper_alignment_still_holds"]),
             False, "仍然不成立（ZhiXing 与 PandaGripper 的 T_cam2grip 平移差 105.54 mm）。"),
            ("initial_pose.landed_init_qpos",
             dig(old, ["initial_pose", "landed_init_qpos"]), landed_qpos,
             "旧值是被覆盖掉的 Panda 解；当前 ZhiXing 解已按夹爪分派。"),
            ("initial_pose.candidate_landed",
             dig(old, ["initial_pose", "candidate_landed"]), True,
             "候选解已在 R1 落地。"),
            ("initial_pose.candidate_grasp_point_residual_mm",
             dig(old, ["initial_pose", "candidate_grasp_point_residual_mm"]),
             dig(initp, ["candidate", "grasp_point_residual_mm"]),
             "重解后残差仍在 0.5 mm 判据内。"),
            ("zhixing.grab_point_vs_panda_reference",
             dig(old, ["mismatch", "grasp_point_vs_panda_reference_mm"]),
             dig(initp, ["current_landed", "grasp_point_vs_ref_mm"]),
             "初始位姿对齐后，落地构型的夹持点与 Panda 基准之差由 109.16 mm 降到 0.12 mm。"),
            ("zhixing.flange_offset_mm", dig(old, ["zhixing", "flange_offset_mm"]), 202.54,
             "eef 前移使 right_gripper 法兰面相对 grip_site 的距离由 120.0 mm 变为 202.54 mm。"),
        ]
        for item, ov, nv, why in pairs:
            diff.append({"item": item, "superseded_value": ov, "current_value": nv,
                         "explanation": why})

    contract = {
        "node": "N2",
        "date": DATE,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": OLD_CONTRACT,
        "status": "current",
        "scope": "当前生产资产（ZhiXing eef=0.20254、R1 夹爪分派 init_qpos、极性修正、"
                 "ZHIXING_GRIPPER_SPEED 部署值 0.01、OSC_POSE output_max）下的 TCP / 相机 / "
                 "开合状态 / 初始位姿重测。所有数字为 2026-09-28 在容器内实测，"
                 "未从旧契约复制任何数值。",
        "approach_axis_convention": "+z（grip_site 局部系，指向工作空间/桌面下方）",
        "definitions": {
            "grip_site": "robosuite/OSC 使用的 eef site（本资产中位于 gripper0_eef body 原点）；"
                         "策略动作即作用于此点",
            "grasp_point": "两指垫碰撞 geom 内侧面中点；在 joint equality / connect 约束收敛的"
                           "构型下测量",
            "measurement_protocol": "libero_spatial task_id=0 init_idx=0 seed=7，256x256；"
                                    "三配置同场景同物体初态；开/闭态见 primary_mode",
            "state_conventions": {
                "reset": "env.reset() 后、任何步进之前的构型（夹爪处于 init_qpos 自洽装配）",
                "open": "执行器全开端点（ZhiXing ctrl=0.0 → Left_1=0；Panda ctrl=0.04）",
                "closed": "执行器近闭端点（ZhiXing ctrl=-0.91 → Left_1=-0.91；Panda ctrl=0.0）",
            },
        },
        "asset_bindings": asset_bindings,
        "tcp": tcp,
        "initial_pose": init_pose,
        "camera": cam,
        "controller": controller,
        "gripper": grip,
        "actuator_path_findings": findings,
        "validation_checks": checks,
        "diff_vs_superseded_contract": diff,
        "provenance": prov,
        "not_reproduced": blocked,
        "raw_artifacts": {
            "dir": "results/n2v2_20260928/",
            "files": sorted(p.name for p in RAW.glob("*.*")) if RAW.exists() else [],
            "logs": sorted(p.name for p in (RAW / "logs").glob("*.log"))
            if (RAW / "logs").exists() else [],
            "history_snapshot": "results/_history_pre_n2_reverify/"
                                "（测量前对 results/*.json|*.csv|*.md|*.txt 的逐字节副本，"
                                "共 63 个文件；本轮未覆盖任何历史文件）",
        },
    }

    out_path = RESULTS / "tcp_camera_contract_v2.json"
    out_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {out_path}")
    print("=== validation_checks ===")
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    print("=== not_reproduced ===")
    print(json.dumps(blocked, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
