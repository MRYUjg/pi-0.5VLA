"""取证：走正式 LIBERO 路径建环境，打印实际使用的 robot_model 类、init_qpos 与初始末端位姿。"""
import pathlib
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

print(f"{'路径':<26} {'robot_model 类':<18} {'init_qpos':<62} {'grip_site 初始位置'}")
for robot, gripper in (("Panda", "PandaGripper"), ("ECO65", "PandaGripper")):
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                            robots=[robot], gripper_types=gripper)
    env.seed(7)
    env.reset()
    r0 = env.env.robots[0].robot_model
    q = np.round(np.asarray(r0.init_qpos), 6)
    obs = env.env._get_observations()
    eef = np.round(np.asarray(obs["robot0_eef_pos"]), 5)
    print(f"{robot + '+' + gripper:<26} {type(r0).__name__:<18} {str(q.tolist()):<62} {eef.tolist()}")
    env.close()
print("APPLIED_CHECK_OK")
