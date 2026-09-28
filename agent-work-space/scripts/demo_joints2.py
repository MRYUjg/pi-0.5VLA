import pathlib
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import mujoco

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

for gripper in ("ZhixingGripper", "PandaGripper"):
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                            robots=["ECO65"], gripper_types=gripper)
    env.seed(7); env.reset()
    sim = env.env.sim
    raw = sim.model._model
    r0 = env.env.robots[0]
    print(f"\n===== {gripper} =====")
    print(f"  gripper.joints 属性 = {list(r0.gripper.joints)}")
    print(f"  _ref_gripper_joint_pos_indexes = {list(r0._ref_gripper_joint_pos_indexes)}")
    print(f"  该机器人全部关节名（前 16 个）:")
    names = [mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_JOINT, j) or "" for j in range(raw.njnt)]
    rob = [n for n in names if "robot0" in n or "gripper" in n or "finger" in n or "Left" in n or "Right" in n]
    for i, n in enumerate(rob):
        print(f"      [{i}] {n}")
    env.close()
print("\nJOINTS_OK")
