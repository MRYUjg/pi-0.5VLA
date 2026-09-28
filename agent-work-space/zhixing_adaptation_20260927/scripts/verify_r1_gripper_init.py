"""R1 复验：ECO65 初始关节角是否按**实际挂载的夹爪**分派。

审核记录 R1 要求：
  * Panda 分支保留旧解 [-0.128227, -0.080175, -1.482052, -0.064915, 1.563529, -1.699308]
  * ZhiXing 分支使用新解 [-0.12639, -0.222035, -0.841572, -0.563545, 1.563635, -1.697476]
  * 修复后做两分支 FK、相机回归与同协议策略复评

本脚本覆盖两件事：
  (A) 单元级分派检查（不需要 sim）：证明 Robot.load_model() 在挂载夹爪后**重新解析**
      init_qpos，且显式传入的 initial_qpos 优先。
      —— 这是关键：ECO65+ZhiXing 分支若仍拿到 Panda 解，说明重新解析没生效。
  (B) 端到端 FK 检查（独立构建 LIBERO Spatial task 0，无策略、无渲染）：
      ECO65+PandaGripper 的初始 grip_site 应回到 Panda 参考（< 0.5 mm / 0.2°）。

用法（仓库根目录）：
  docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
    /bin/bash -c "source /.venv/bin/activate && python \
      /app/agent-work-space/zhixing_adaptation_20260927/scripts/verify_r1_gripper_init.py"
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

OUT = pathlib.Path(os.environ.get(
    "R1_OUT", "/app/agent-work-space/zhixing_adaptation_20260927/results"))
OUT.mkdir(parents=True, exist_ok=True)

PANDA_INIT = np.array([-0.128227, -0.080175, -1.482052, -0.064915, 1.563529, -1.699308])
ZHIXING_INIT = np.array([-0.12639, -0.222035, -0.841572, -0.563545, 1.563635, -1.697476])

POS_TOL_MM = 0.5      # ECO65+Panda vs Panda 参考 位置容差
ROT_TOL_DEG = 0.2     # 姿态容差


# --------------------------------------------------------------------------- #
# (A) 单元级分派检查（无 sim）
# --------------------------------------------------------------------------- #
def unit_dispatch_check():
    from robosuite.models.grippers import gripper_factory
    from robosuite.models.robots.manipulators.eco65_robot import (
        ECO65,
        ECO65_INIT_QPOS_BY_GRIPPER,
        ECO65_INIT_QPOS_FALLBACK,
        attached_gripper_name,
    )
    from robosuite.robots import SingleArm

    rec = {}

    # A1. 模型层：未挂载夹爪时返回兜底（Panda 解），挂载后按夹爪分派
    m = ECO65()
    rec["model_before_attach_name"] = attached_gripper_name(m)
    rec["model_before_attach_qpos"] = m.init_qpos.tolist()
    m.add_gripper(gripper_factory("PandaGripper"))
    rec["model_after_panda_qpos"] = m.init_qpos.tolist()
    m2 = ECO65()
    m2.add_gripper(gripper_factory("ZhixingGripper"))
    rec["model_after_zhixing_qpos"] = m2.init_qpos.tolist()

    # A2. Robot 层：load_model() 后 self.init_qpos 必须是**夹爪专属**值
    def robot_init(gripper_type, initial_qpos=None):
        r = SingleArm(robot_type="ECO65", gripper_type=gripper_type,
                      initial_qpos=initial_qpos)
        r.load_model()
        return {
            "attached_gripper": type(r.gripper).__name__,
            "init_qpos_explicit_flag": bool(getattr(r, "_init_qpos_explicit", None)),
            "init_qpos": np.asarray(r.init_qpos).tolist(),
        }

    rec["robot_eco65_panda"] = robot_init("PandaGripper")
    rec["robot_eco65_zhixing"] = robot_init("ZhixingGripper")
    forced = [0.1, -0.2, -0.3, -0.4, 0.5, -0.6]
    rec["robot_eco65_forced_override"] = robot_init("PandaGripper", initial_qpos=forced)

    checks = {
        "table_has_both_grippers": bool(
            "PandaGripper" in ECO65_INIT_QPOS_BY_GRIPPER
            and "ZhixingGripper" in ECO65_INIT_QPOS_BY_GRIPPER),
        "before_attach_returns_fallback": bool(
            rec["model_before_attach_name"] is None
            and np.allclose(rec["model_before_attach_qpos"], ECO65_INIT_QPOS_FALLBACK)),
        "model_after_panda_is_panda_solution": bool(
            np.allclose(rec["model_after_panda_qpos"], PANDA_INIT, atol=1e-9)),
        "model_after_zhixing_is_zhixing_solution": bool(
            np.allclose(rec["model_after_zhixing_qpos"], ZHIXING_INIT, atol=1e-9)),
        "robot_panda_branch_gets_panda_solution": bool(
            np.allclose(rec["robot_eco65_panda"]["init_qpos"], PANDA_INIT, atol=1e-9)),
        # 关键判据：预解析阶段必然拿到 fallback(Panda)，只有重新解析生效才会变成 ZhiXing 解
        "robot_zhixing_branch_reresolved_ok": bool(
            np.allclose(rec["robot_eco65_zhixing"]["init_qpos"], ZHIXING_INIT, atol=1e-9)),
        "explicit_override_wins": bool(
            rec["robot_eco65_forced_override"]["init_qpos_explicit_flag"]
            and np.allclose(rec["robot_eco65_forced_override"]["init_qpos"], forced, atol=1e-9)),
    }
    return rec, checks


# --------------------------------------------------------------------------- #
# (B) 端到端 FK 检查
# --------------------------------------------------------------------------- #
def build(robot, gripper, suite="libero_spatial"):
    task = benchmark.get_benchmark_dict()[suite]().get_task(0)
    p = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    e = ControlEnv(bddl_file_name=p, robots=[robot], gripper_types=gripper,
                   use_camera_obs=False, has_offscreen_renderer=False)
    e.seed(7)
    e.reset()
    return e


def probe(robot, gripper):
    e = build(robot, gripper)
    s, m, d, g = N.ids(e)
    r = e.env.robots[0]
    rec = {
        "robot": robot, "gripper": gripper,
        "robot_model_class": type(r.robot_model).__name__,
        "attached_gripper_class": type(r.gripper).__name__,
        "robot_model_init_qpos": r.robot_model.init_qpos.tolist(),
        "robot_init_qpos": np.asarray(r.init_qpos).tolist(),
        "init_qpos_at_reset": d.qpos[r._ref_joint_pos_indexes].copy().tolist(),
        "grip_site_pos": d.site_xpos[g].copy().tolist(),
        "grip_site_R": d.site_xmat[g].copy().reshape(3, 3).tolist(),
    }
    e.close()
    return rec


def main():
    rec_u, checks = unit_dispatch_check()

    ref = probe("Panda", "PandaGripper")
    eco_panda = probe("ECO65", "PandaGripper")
    eco_zhixing = probe("ECO65", "ZhixingGripper")

    dp = float(np.linalg.norm(np.array(eco_panda["grip_site_pos"])
                              - np.array(ref["grip_site_pos"])) * 1000)
    dr = float(np.linalg.norm(N.rot_err_vec(np.array(ref["grip_site_R"]),
                                            np.array(eco_panda["grip_site_R"]))) * 180 / np.pi)
    dz = float(np.linalg.norm(np.array(eco_zhixing["grip_site_pos"])
                              - np.array(ref["grip_site_pos"])) * 1000)

    checks["fk_eco65_panda_matches_reference"] = bool(dp < POS_TOL_MM and dr < ROT_TOL_DEG)
    checks["fk_eco65_panda_reset_equals_init"] = bool(
        np.allclose(eco_panda["init_qpos_at_reset"], PANDA_INIT, atol=1e-6))
    checks["fk_eco65_zhixing_reset_equals_init"] = bool(
        np.allclose(eco_zhixing["init_qpos_at_reset"], ZHIXING_INIT, atol=1e-6))

    out = {
        "node": "R1", "unit_checks": rec_u, "fk": {
            "reference": ref, "eco65_panda": eco_panda, "eco65_zhixing": eco_zhixing},
        "eco65_panda_pos_offset_mm": dp,
        "eco65_panda_rot_offset_deg": dr,
        "eco65_zhixing_pos_offset_mm": dz,
        "tolerances": {"pos_mm": POS_TOL_MM, "rot_deg": ROT_TOL_DEG},
        "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    p = OUT / "r1_gripper_init_verify.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 78)
    print("[A] 单元级分派（无 sim）")
    print("  未挂载夹爪 → %s" % np.round(rec_u["model_before_attach_qpos"], 6).tolist())
    print("  挂 PandaGripper   → %s" % np.round(rec_u["model_after_panda_qpos"], 6).tolist())
    print("  挂 ZhixingGripper → %s" % np.round(rec_u["model_after_zhixing_qpos"], 6).tolist())
    for k in ("robot_eco65_panda", "robot_eco65_zhixing", "robot_eco65_forced_override"):
        print("  %-32s init_qpos=%s" % (k, np.round(rec_u[k]["init_qpos"], 6).tolist()))
    print("-" * 78)
    print("[B] 端到端 FK（LIBERO Spatial task 0, seed=7）")
    for k, rec in [("Panda+PandaGripper(参考)", ref), ("ECO65+PandaGripper", eco_panda),
                   ("ECO65+ZhixingGripper", eco_zhixing)]:
        print("  %-26s grip_site=%s" % (k, np.round(rec["grip_site_pos"], 6).tolist()))
    print("  ECO65+Panda  vs Panda 参考 : 位置差 %.4f mm  姿态差 %.5f deg （阈值 %.2f / %.2f）"
          % (dp, dr, POS_TOL_MM, ROT_TOL_DEG))
    print("  ECO65+ZhiXing vs Panda 参考: 位置差 %.4f mm  （R1 记录误用解时为 109.1435 mm）" % dz)
    print("-" * 78)
    for k, v in checks.items():
        print("  %-46s %s" % (k, "✅" if v else "❌"))
    print("NODE_VERDICT = %s" % out["verdict"])
    print("[out] %s" % p)
    return 0 if out["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
