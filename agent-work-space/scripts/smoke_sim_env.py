"""D7 冒烟：在 libero 镜像里创建 ECO65 + 知行夹爪 的 LIBERO 环境，跑几步 dummy action。
不依赖策略服务与权重，专门验证 仿真侧适配链路（BDDL / MountedECO65 / ZhixingGripper / OSC / EGL 相机）。"""
import pathlib

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
print("task      :", task.language)
print("bddl      :", bddl, "| exists:", bddl.exists())

env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, robots=["ECO65"])
env.seed(7)
obs = env.reset()

robot = env.env.robots[0]
print("robot_model :", type(robot.robot_model).__name__)
print("gripper     :", type(robot.gripper).__name__, "| dof:", robot.gripper.dof)
print("action_dim  :", robot.action_dim, "| controller:", type(robot.controller).__name__)
print("obs keys    :", sorted(obs.keys()))
print("agentview   :", obs["agentview_image"].shape, obs["agentview_image"].dtype)
print("wrist       :", obs["robot0_eye_in_hand_image"].shape)
print("eef_pos     :", np.round(obs["robot0_eef_pos"], 4))
print("gripper_qpos:", np.round(obs["robot0_gripper_qpos"], 4), "len:", len(obs["robot0_gripper_qpos"]))

state = np.concatenate((obs["robot0_eef_pos"], obs["robot0_gripper_qpos"][[1, 5]]))
print("client state dim (eef_pos+gripper[[1,5]]):", state.shape)

action = np.zeros(7, dtype=np.float32)
action[-1] = -1.0  # 张开
for i in range(5):
    obs, reward, done, info = env.step(action.tolist())
print("5 steps OK  : done =", done, "| gripper_qpos =", np.round(obs["robot0_gripper_qpos"], 4))

action[-1] = 1.0  # 闭合
for i in range(5):
    obs, reward, done, info = env.step(action.tolist())
print("close 5步 OK: gripper_qpos =", np.round(obs["robot0_gripper_qpos"], 4))
env.close()
print("SMOKE_SIM_OK")
