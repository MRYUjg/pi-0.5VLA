"""Independently compare policy proposals, actual controls, native flags and video."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_episode(path: Path) -> dict:
    result = json.loads((path / "result.json").read_text())
    registration = json.loads((path / "env/registration.json").read_text())
    transitions = lines(path / "env/transitions.jsonl")
    predictions = lines(path / "vla/predictions.jsonl")
    initial_obs = dict(np.load(path / "env/initial_policy_obs.npz", allow_pickle=False))
    feedback_current = True
    prediction_archives_match = True
    for index, prediction in enumerate(predictions):
        archive = dict(np.load(path / f"vla/prediction{index:04d}.npz", allow_pickle=False))
        previous = initial_obs["states"] if index == 0 else np.asarray(transitions[index * 5 - 1]["state8"], np.float32)
        feedback_current &= np.array_equal(archive["state"], previous)
        prediction_archives_match &= np.array_equal(archive["actions"], np.asarray(prediction["raw_actions"], np.float32))
        prediction_archives_match &= hashlib.sha256(archive["image"].tobytes()).hexdigest() == prediction["image_sha256"]
        prediction_archives_match &= hashlib.sha256(archive["wrist_image"].tobytes()).hexdigest() == prediction["wrist_sha256"]
    proposals = np.asarray([action for p in predictions for action in p["raw_actions"][:5]], np.float32)
    executed = np.asarray([t["executed_action"] for t in transitions], np.float32)
    raw = np.asarray([t["proposed_action"] for t in transitions], np.float32)
    source_lock = json.loads((path / "source_lock.json").read_text())
    source_match = all(hashlib.sha256((path / "executed_source" / name).read_bytes()).hexdigest() == sha
                       for name, sha in source_lock.items())
    reader = imageio.get_reader(path / "env/dual_camera.mp4")
    shapes = [frame.shape for frame in reader]
    reader.close()
    responses = lines(path / "api_responses.jsonl")
    models = sorted(set(r.get("model") for r in responses if r.get("model")))
    native = bool(transitions[-1]["native_success"])
    steps = len(transitions)
    transfer = registration["initialization_transfer"]
    official = np.load(path / "env/official_initial.npy", allow_pickle=False)
    eco = np.load(path / "env/eco_initial_before_settle.npy", allow_pickle=False)
    transfer_exact = True
    for name, mapping in transfer["joint_index_map"].items():
        for field in ["qpos", "qvel"]:
            source_offset = 1 + (transfer["source_nq"] if field == "qvel" else 0)
            target_offset = 1 + (transfer["target_nq"] if field == "qvel" else 0)
            a = official[source_offset + np.asarray(mapping["source_" + field])]
            b = eco[target_offset + np.asarray(mapping["target_" + field])]
            transfer_exact &= np.array_equal(a, b)
            transfer_exact &= np.array_equal(a, np.atleast_1d(transfer["object_" + field][name]))
    state_selection = all(np.array_equal(np.asarray(t["state8"][-2:], np.float32),
                            np.asarray(t["diagnostics"]["gripper_qpos"], np.float32)[registration["gripper_state_indices"]])
                          for t in transitions)
    checks = {
        "initial_objects_recomputed_from_archived_states": bool(transfer_exact),
        "robot_reset_recomputed_from_archived_state": np.array_equal(
            eco[1 + np.asarray(transfer["robot_qpos_indices"])], transfer["robot_qpos_before"]),
        "state_selection_matches_actual_gripper_qpos": state_selection,
        "runtime_chain_passed": bool(result["runtime_chain_passed"]),
        "clean_component_exits": bool(result["owned_components_exited_cleanly"]),
        "frozen_executed_source_matches": source_match,
        "ECO65_and_ZhixingGripper": registration["robot"] == "ECO65" and registration["gripper"] == "ZhixingGripper"
            and registration["controller"]["robot_model"] == "MountedECO65"
            and registration["controller"]["gripper_model"] == "ZhixingGripper",
        "official_scene_transfer_exact": registration["initialization_transfer"]["objects_exact_before_settle"],
        "existing_robot_reset_retained": registration["initialization_transfer"]["robot_reset_retained"],
        "ZhiXing_speed_unchanged": registration["gripper_speed"] == 0.01,
        "ZhiXing_state_named_active_joints": all(name.endswith(suffix) for name, suffix in zip(
            registration["gripper_state_joint_names"], ["Left_1_Joint", "Right_1_Joint"])),
        "controller_20Hz_OSC_unchanged": registration["controller"]["control_freq"] == 20
            and registration["controller"]["controller"] == "OperationalSpaceController"
            and registration["controller"]["control_delta"] is True
            and registration["controller"]["extra_joint_pd_profile"] == "legacy",
        "prediction_actual_action_exact": len(proposals) >= steps and np.array_equal(proposals[:steps], raw),
        "normalization_clipping_exact": np.array_equal(np.clip(raw, -1, 1), executed),
        "finite_actions_and_states": bool(np.isfinite(executed).all()
                                          and np.isfinite([t["state8"] for t in transitions]).all()),
        "every_control_interval_50ms": all(np.isclose(t["time_after"] - t["time_before"], .05, atol=1e-8) for t in transitions),
        "bounded_control_budget": 0 < steps <= 220,
        "native_result_matches_actual_terminal": native == result["native_success"],
        "LLM_finish_claim_matches_native": bool(result["finish_claim_matches_native"]),
        "decoded_video_T_plus_1_dual_camera": len(shapes) == steps + 1
            and all(shape == (256, 512, 3) for shape in shapes),
        "real_VLA_requests": len(predictions) > 0,
        "every_prediction_uses_latest_state": bool(feedback_current),
        "archived_VLA_inputs_actions_match_request_records": bool(prediction_archives_match),
        "policy_instructions_match_original_task": all(p["prompt"] == registration["language"] for p in predictions),
        "official_initial_state_archive_matches_registered_fingerprint": hashlib.sha256(
            np.ascontiguousarray(np.load(path / "env/official_initial.npy", allow_pickle=False)).tobytes()
            ).hexdigest() == registration["initial_state_sha256"],
        "real_component_guards_and_camera_checks": all(json.loads((path / "component_checks.json").read_text()).values()),
        "initial_frame_and_reset_present": (path / "env/initial_policy_obs.npz").exists()
            and (path / "env/official_initial.npy").exists(),
    }
    if result["mode"] == "planner":
        checks.update(real_DeepSeek_responses=len(responses) > 0 and all(r["http_status"] == 200 for r in responses),
                      actual_model_identity_matches=models == ["deepseek-flash"],
                      no_planner_error=result["planner_error"] is None,
                      planner_finish_recorded=result.get("finish") is not None,
                      native_terminal_or_budget_exhausted=native or steps == 220)
    report = dict(task=result["task"], seed=result["seed"], language=registration["language"],
                  native_success=native, steps=steps, VLA_requests=len(predictions),
                  actual_models=models, planner_stats=result.get("planner_stats"),
                  checks=checks, audit_passed=all(checks.values()),
                  task_success_independent_of_audit=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = []
    for path in sorted(args.run.glob("task*_seed*")):
        if not path.is_dir():
            continue
        try:
            reports.append(audit_episode(path))
        except Exception as exc:
            reports.append({"episode": path.name, "audit_passed": False,
                            "audit_error": f"{type(exc).__name__}: {exc}"})
    summary = {"episodes": reports, "trials": len(reports),
               "successes": sum(r.get("native_success", False) for r in reports),
               "all_audits_passed": bool(reports) and all(r["audit_passed"] for r in reports),
               "formal_benchmark_accepted": False}
    with args.output.open("x") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
