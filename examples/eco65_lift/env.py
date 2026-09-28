# examples/eco65_sim/env.py
# ECO65 + RethinkGripper environment wrapper for robosuite.
import logging
import math

import numpy as np
import robosuite as suite
from robosuite.models.robots.manipulators.eco65_robot import ECO65  # noqa: F401  trigger metaclass
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

logger = logging.getLogger(__name__)


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat = quat / np.linalg.norm(quat)
    elif quat[3] < -1.0:
        quat = quat / np.linalg.norm(quat)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class ECO65SimEnvironment(_environment.Environment):
    def __init__(self, task="Lift", seed=0, camera_height=224, camera_width=224):
        self._rng = np.random.default_rng(seed)
        controller_config = suite.load_controller_config(default_controller="OSC_POSE")
        self._env = suite.make(
            env_name=task, robots="ECO65", gripper_types="RethinkGripper",
            has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True,
            camera_names="agentview", camera_heights=camera_height,
            camera_widths=camera_width, controller_configs=controller_config,
            ignore_done=True,
        )
        self._last_obs = None
        self._done = True
        logger.info("ECO65SimEnvironment: task=%s", task)

    @override
    def reset(self):
        self._last_obs = self._env.reset()
        self._done = False

    @override
    def is_episode_complete(self):
        return self._done

    @override
    def get_observation(self):
        if self._last_obs is None:
            raise RuntimeError("Call reset() first.")
        obs = self._last_obs
        img = np.asarray(obs["agentview_image"])
        img = image_tools.resize_with_pad(img, 224, 224)
        img = image_tools.convert_to_uint8(img)
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        eef_quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32)
        eef_rot = _quat2axisangle(eef_quat).astype(np.float32)
        gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
        state = np.concatenate([eef_pos, eef_rot, gripper_qpos])
        wrist_img = np.zeros((224, 224, 3), dtype=np.uint8)
        return {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": "lift the cube",
        }

    @override
    def apply_action(self, action):
        actions = np.asarray(action["actions"], dtype=np.float64)
        obs, reward, terminated, info = self._env.step(actions.tolist())
        self._last_obs = obs
