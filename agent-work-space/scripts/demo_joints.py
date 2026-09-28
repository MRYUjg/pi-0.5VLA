"""列出两种夹爪在编译模型里的关节顺序（决定 robot0_gripper_qpos 的索引含义）。"""
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
    idx = sim.model.get_joint_qpos_addr("robot0_gripper_qpos") if False else None
    print(f"\n===== {gripper}：robot0_gripper_qpos 的索引 -> 关节名 =====")
    n = 0
    for j in range(raw.njnt):
        name = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if not name.startswith("robot0_"):
            continue
        # 夹爪关节：不属于 6 个机械臂关节
        if name.startswith("robot0_joint_"):
            continue
        print(f"  [{n}] {name}")
        n += 1
    env.close()
print("\nJOINTS_OK")
