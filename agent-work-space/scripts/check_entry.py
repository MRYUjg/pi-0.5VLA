"""验证 examples/libero/main.py 的夹爪 state 分派：Panda->[:2]，Zhixing->[1,5]。"""
import importlib.util

import numpy as np

spec = importlib.util.spec_from_file_location("libero_main", "/app/examples/libero/main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

from libero.libero import benchmark

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
print("GRIPPER_STATE_SELECTOR =", m.GRIPPER_STATE_SELECTOR)
for g in ("default", "PandaGripper", "ZhixingGripper"):
    env, _ = m._get_libero_env(task, 256, 7, g)
    name, sel = m._resolve_gripper_state_selector(env)
    obs = env.reset()
    gq = obs["robot0_gripper_qpos"]
    state = np.concatenate((obs["robot0_eef_pos"], m._quat2axisangle(obs["robot0_eef_quat"]), gq[sel]))
    print(f"  --gripper {g:15s} -> 实际 {name:15s} 关节数={len(gq)} 选择器={sel} "
          f"state维度={state.shape[0]} 夹爪两位={np.round(state[-2:], 4).tolist()}")
    env.close()
print("ENTRY_CHECK_OK")
