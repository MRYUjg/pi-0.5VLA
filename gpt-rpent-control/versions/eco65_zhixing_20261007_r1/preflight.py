"""Actual new embodiment/reset, camera and one hold action; no model success claim."""
import argparse
import json
from pathlib import Path
import numpy as np
from env_server import ProjectLiberoFacade

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
env = ProjectLiberoFacade("libero_spatial", 0, 0, args.output)
try:
    obs, info = env.reset()
    assert obs["states"].shape == (8,)
    assert all(obs[k].shape == (256, 256, 3) for k in ["main_images", "wrist_images"])
    registration = info["initialization"]
    assert registration["controller"]["robot_model"] == "MountedECO65"
    assert registration["controller"]["gripper_model"] == "ZhixingGripper"
    assert registration["initialization_transfer"]["objects_exact_before_settle"]
    obs, reward, terminated, truncated, _ = env.step([0.] * 6 + [-1.])
    assert np.isfinite(obs["states"]).all()
    (args.output / "preflight_checks.json").write_text(json.dumps(dict(
        actual_env_reset_transfer_action=True, native_success=bool(terminated),
        policy_state_shape=list(obs["states"].shape), gripper_indices=env.gripper_state_indices,
        scope="one valid hold action; not policy or adaptation acceptance"), indent=2))
finally:
    env.close()
print("ECO65/ZhiXing actual reset, official-object transfer, RGB/state and action passed")
