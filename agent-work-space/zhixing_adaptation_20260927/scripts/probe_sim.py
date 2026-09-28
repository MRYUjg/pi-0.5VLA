#!/usr/bin/env python3
"""一次性探针：确认本环境下 sim / qpos / site 的正确访问路径，避免盲写。"""
import pathlib
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

bench = benchmark.get_benchmark_dict()["libero_spatial"]()
task = bench.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(robots=["ECO65"], bddl_file_name=bddl,
                         camera_heights=256, camera_widths=256,
                         gripper_types="ZhixingGripper")
env.seed(7)
obs = env.reset()
robot = env.env.robots[0]

print("=== env chain ===")
print("type(env) =", type(env))
print("has env.env =", hasattr(env, "env"), type(getattr(env, "env", None)))
print("type(env.env.robots[0]) =", type(robot))

print("\n=== robot.sim ===")
s = robot.sim
print("type(robot.sim) =", type(s))
print("module =", type(s).__module__)
print("has .data =", hasattr(s, "data"))
print("attrs with 'data' =", [a for a in dir(s) if "data" in a.lower()][:20])
print("has qpos attr =", hasattr(s, "qpos"))

print("\n=== env.env.sim ===")
s2 = getattr(env.env, "sim", None)
print("type =", type(s2), "| is robot.sim =", s2 is s, "| has .data =", hasattr(s2, "data") if s2 else None)

print("\n=== obs keys (gripper) ===")
print([k for k in obs.keys() if "gripper" in k or "robot0" in k][:15])
print("obs['robot0_gripper_qpos'] =", np.round(np.asarray(obs["robot0_gripper_qpos"]), 5).tolist())

print("\n=== robot helpers ===")
print("has _joint_positions =", hasattr(robot, "_joint_positions"))
try:
    print("_joint_positions =", np.round(np.asarray(robot._joint_positions), 5).tolist())
except Exception as e:
    print("_joint_positions err:", e)
print("_ref_gripper_joint_pos_indexes =", list(robot._ref_gripper_joint_pos_indexes))

print("\n=== sim model access for sites/joints ===")
m = getattr(s, "model", None)
print("type(sim.model) =", type(m), "| has _model =", hasattr(m, "_model"))
mm = getattr(m, "_model", m)
print("njnt =", getattr(mm, "njnt", None), "| nsite =", getattr(mm, "nsite", None))
try:
    import mujoco
    print("site names:")
    for i in range(mm.nsite):
        n = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_SITE, i)
        if n and ("Support_tip" in n or "connect" in n or "grip" in n):
            print("   ", i, n)
except Exception as e:
    print("site enum err:", e)

print("\n=== qpos via mujoco direct ===")
try:
    import mujoco
    d = getattr(s, "data", None)
    dd = getattr(d, "_data", d)
    print("qpos len =", len(dd.qpos) if dd is not None else None)
    print("site_xpos shape =", None if dd is None else np.asarray(dd.site_xpos).shape)
except Exception as e:
    print("err:", e)
print("PROBE_DONE")
