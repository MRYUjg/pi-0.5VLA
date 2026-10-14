#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""措施 B1 复验：初始化分派在**三个 ECO65 类**上是否一致（复核 §4）。

复核指出：`OnTheGroundECO65`（Object / libero_10 走的地面 arena）仍硬编码
ZhiXing（桌面）解，导致 ECO65+PandaGripper 与 Panda 参考偏 109.14 mm。

本探针核验：
  1. 三个类是否都走同一张 `ECO65_INIT_QPOS_BY_GRIPPER` 表（按实际挂载夹爪分派）；
  2. 桌面（libero_spatial → MountedECO65）与地面（libero_object → OnTheGroundECO65）
     两种环境下，Panda+Panda / ECO65+Panda / ECO65+ZhiXing 三组的初始 grip_site；
  3. 相机回归：`T_cam2grip` 与相机世界位姿在 ECO65+Panda 与 Panda 之间是否一致；
  4. 初态接触与夹爪净开口。

用法：
  docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
    /bin/bash -c "source /.venv/bin/activate && python \
      /app/agent-work-space/zhixing_adaptation_20260927/scripts/verify_b1_init_dispatch.py"
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402
from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402

OUT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927/results")
OUT.mkdir(parents=True, exist_ok=True)

PANDA_INIT = np.array([-0.128227, -0.080175, -1.482052, -0.064915, 1.563529, -1.699308])
ZHIXING_INIT = np.array([-0.12639, -0.222035, -0.841572, -0.563545, 1.563635, -1.697476])

# env 种类 → (suite, 期望的机器人类名)
ENVS = {
    "table": ("libero_spatial", "MountedECO65"),
    "ground": ("libero_object", "OnTheGroundECO65"),
}
COMBOS = [("Panda", "PandaGripper"), ("ECO65", "PandaGripper"), ("ECO65", "ZhixingGripper")]

POS_TOL_MM = 0.5      # ECO65+Panda vs Panda 参考
ZHIXING_TOL_MM = 5.0  # ECO65+ZhiXing 按设计对齐的是**夹持点**，grip_site 允许 ~3.6 mm


def build(robot, gripper, suite, task_id=0, seed=7):
    task = benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    p = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    e = ControlEnv(bddl_file_name=p, robots=[robot], gripper_types=gripper,
                   use_camera_obs=True, has_offscreen_renderer=True,
                   camera_names=["agentview", "robot0_eye_in_hand"],
                   camera_heights=256, camera_widths=256)
    e.seed(seed)
    e.reset()
    return e


def cam_pose(m, d, name="robot0_eye_in_hand"):
    import mujoco

    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        return None, None
    return np.array(d.cam_xpos[cid], float), np.array(d.cam_xmat[cid], float).reshape(3, 3)


def probe(robot, gripper, suite):
    e = build(robot, gripper, suite)
    sim, m, d, gid = N.ids(e)
    r = e.env.robots[0]
    p, R = np.array(d.site_xpos[gid], float), np.array(d.site_xmat[gid], float).reshape(3, 3)
    cpos, cmat = cam_pose(m, d)
    obs = e.env._get_observations(force_update=True)
    av = np.asarray(obs["agentview_image"], dtype=np.int16)
    wr = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.int16)

    # T_cam2grip = inv(cam) @ grip
    T = None
    if cpos is not None:
        T = np.eye(4)
        T[:3, :3] = cmat.T @ R
        T[:3, 3] = cmat.T @ (p - cpos)

    rec = {
        "robot": robot, "gripper": gripper, "suite": suite,
        "robot_model_class": type(r.robot_model).__name__,
        "attached_gripper": type(r.gripper).__name__,
        "robot_init_qpos": np.asarray(r.init_qpos).tolist(),
        "model_init_qpos": np.asarray(r.robot_model.init_qpos).tolist(),
        "init_qpos_at_reset": np.asarray(d.qpos[r._ref_joint_pos_indexes], float).tolist(),
        "grip_site_pos": p.tolist(),
        "grip_site_R": R.tolist(),
        "camera_world_pos": None if cpos is None else cpos.tolist(),
        "T_cam2grip": None if T is None else T.tolist(),
        "ncon_at_reset": int(d.ncon),
        "jaw_sep_mm": N.jaw_face_separation_mm(e),
        "gripper_qpos": [float(d.qpos[i]) for i in r._ref_gripper_joint_pos_indexes],
        "agentview_mean": float(av.mean()), "wrist_mean": float(wr.mean()),
        "agentview_masked_mean": float(av[av > 0].mean()) if (av > 0).any() else 0.0,
        "wrist_masked_mean": float(wr[wr > 0].mean()) if (wr > 0).any() else 0.0,
    }
    # 供相机对比用的图像（只留 numpy，序列化时去掉）
    rec["_av"], rec["_wr"] = av, wr
    e.close()
    return rec


def main():
    out = {"node": "B1", "envs": {}, "checks": {}}
    checks = out["checks"]

    for env_kind, (suite, expect_cls) in ENVS.items():
        recs = {}
        for robot, gripper in COMBOS:
            key = "%s_%s" % (robot, gripper)
            recs[key] = probe(robot, gripper, suite)
            print("[%s] %-22s 类=%-18s init_qpos=%s  grip=%s"
                  % (env_kind, key, recs[key]["robot_model_class"],
                     np.round(recs[key]["robot_init_qpos"], 6).tolist(),
                     np.round(recs[key]["grip_site_pos"], 6).tolist()))

        ref = np.array(recs["Panda_PandaGripper"]["grip_site_pos"])
        ecp = np.array(recs["ECO65_PandaGripper"]["grip_site_pos"])
        ecz = np.array(recs["ECO65_ZhixingGripper"]["grip_site_pos"])
        recs["eco65_panda_offset_mm"] = float(np.linalg.norm(ecp - ref) * 1000)
        recs["eco65_zhixing_offset_mm"] = float(np.linalg.norm(ecz - ref) * 1000)
        # T_cam2grip 差异
        Tp = np.array(recs["Panda_PandaGripper"]["T_cam2grip"])
        Te = np.array(recs["ECO65_PandaGripper"]["T_cam2grip"])
        Tz = np.array(recs["ECO65_ZhixingGripper"]["T_cam2grip"])
        recs["T_cam2grip_eco65panda_vs_panda_mm"] = float(np.linalg.norm(Te[:3, 3] - Tp[:3, 3]) * 1000)
        recs["T_cam2grip_zhixing_vs_panda_mm"] = float(np.linalg.norm(Tz[:3, 3] - Tp[:3, 3]) * 1000)
        # 相机世界位姿差（ECO65+Panda vs Panda）
        cw_e = np.array(recs["ECO65_PandaGripper"]["camera_world_pos"])
        cw_p = np.array(recs["Panda_PandaGripper"]["camera_world_pos"])
        recs["camera_world_pos_delta_mm"] = float(np.linalg.norm(cw_e - cw_p) * 1000)

        # 相机渲染差（掩膜后，排除背景 0 像素）
        av_e, av_p = recs["ECO65_PandaGripper"].pop("_av"), recs["Panda_PandaGripper"].pop("_av")
        wr_e, wr_p = recs["ECO65_PandaGripper"].pop("_wr"), recs["Panda_PandaGripper"].pop("_wr")
        for k in ("ECO65_ZhixingGripper",):
            recs[k].pop("_av"); recs[k].pop("_wr")
        msk = (av_p > 0) & (av_e > 0)
        recs["agentview_masked_diff_255"] = float(np.abs(av_e[msk].astype(np.float64)
                                                         - av_p[msk]).mean()) if msk.any() else None
        mskw = (wr_p > 0) & (wr_e > 0)
        recs["wrist_masked_diff_255"] = float(np.abs(wr_e[mskw].astype(np.float64)
                                                     - wr_p[mskw]).mean()) if mskw.any() else None

        out["envs"][env_kind] = recs

        # ---- 判据 ----
        p = "%s_" % env_kind
        checks[p + "robot_class_is_expected"] = bool(
            recs["ECO65_PandaGripper"]["robot_model_class"] == expect_cls)
        checks[p + "eco65_panda_uses_panda_solution"] = bool(np.allclose(
            recs["ECO65_PandaGripper"]["robot_init_qpos"], PANDA_INIT, atol=1e-9))
        checks[p + "eco65_zhixing_uses_zhixing_solution"] = bool(np.allclose(
            recs["ECO65_ZhixingGripper"]["robot_init_qpos"], ZHIXING_INIT, atol=1e-9))
        checks[p + "eco65_panda_matches_panda_reference"] = bool(
            recs["eco65_panda_offset_mm"] < POS_TOL_MM)
        checks[p + "eco65_zhixing_within_design_tol"] = bool(
            recs["eco65_zhixing_offset_mm"] < ZHIXING_TOL_MM)
        checks[p + "cam2grip_panda_path_unchanged"] = bool(
            recs["T_cam2grip_eco65panda_vs_panda_mm"] < 1e-6)
        checks[p + "cameras_rendered_nonblank"] = bool(
            all(recs[k]["agentview_mean"] > 1.0 and recs[k]["wrist_mean"] > 1.0
                for k in ("Panda_PandaGripper", "ECO65_PandaGripper", "ECO65_ZhixingGripper")))

    checks["all_envs_agree"] = bool(
        abs(out["envs"]["table"]["eco65_panda_offset_mm"]
            - out["envs"]["ground"]["eco65_panda_offset_mm"]) < 0.5)
    out["verdict"] = "PASS" if all(checks.values()) else "FAIL"

    p = OUT / "b1_init_dispatch_verify.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    for env_kind in ENVS:
        r = out["envs"][env_kind]
        print("[%s] ECO65+Panda vs Panda 参考 : %.4f mm   (修复前 109.1435 mm)"
              % (env_kind, r["eco65_panda_offset_mm"]))
        print("[%s] ECO65+ZhiXing vs Panda    : %.4f mm   (设计值 ~3.6 mm，对齐的是夹持点)"
              % (env_kind, r["eco65_zhixing_offset_mm"]))
        print("[%s] T_cam2grip Δ(Panda 路径)  : %.3e mm   Δ(ZhiXing) %.3f mm"
              % (env_kind, r["T_cam2grip_eco65panda_vs_panda_mm"],
                 r["T_cam2grip_zhixing_vs_panda_mm"]))
        print("[%s] 相机世界位姿差 %.3f mm  渲染掩膜差 外部 %.3f / 腕部 %.3f (/255)"
              % (env_kind, r["camera_world_pos_delta_mm"],
                 r["agentview_masked_diff_255"] or -1, r["wrist_masked_diff_255"] or -1))
    print("-" * 78)
    for k, v in checks.items():
        print("  %-52s %s" % (k, "✅" if v else "❌"))
    print("NODE_VERDICT = %s" % out["verdict"])
    print("[out] %s" % p)
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
