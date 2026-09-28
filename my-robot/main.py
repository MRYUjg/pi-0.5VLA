import os
os.environ["MUJOCO_GL"] = "wgl"

import numpy as np
import robosuite as suite
from robosuite.robots import ROBOT_CLASS_MAPPING, SingleArm
ROBOT_CLASS_MAPPING["ECO65"] = SingleArm
ROBOT_CLASS_MAPPING["ECO65-B"] = SingleArm
import mujoco
import mujoco.viewer
import time

env = suite.make(
    env_name="Lift",
    robots="ECO65",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    ignore_done=True,
)

env.reset()

model = env.sim.model._model
data = env.sim.data._data

with mujoco.viewer.launch(model, data) as viewer:
    viewer.opt.geomgroup[0] = 0

    while viewer.is_running():
        step_start = time.time()
        action = np.zeros(7)
        env.step(action)
        viewer.sync()

        elapsed = time.time() - step_start
        if elapsed < env.control_timestep:
            time.sleep(env.control_timestep - elapsed)

env.close()