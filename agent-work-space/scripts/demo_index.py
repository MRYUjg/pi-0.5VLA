"""演示 examples/libero/main.py:147 的 obs["robot0_gripper_qpos"][[1, 5]] 在不同夹爪下的行为。"""
import pathlib
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

for gripper in ("ZhixingGripper", "PandaGripper"):
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256,
                            robots=["ECO65"], gripper_types=gripper)
    env.seed(7)
    obs = env.reset()
    gq = obs["robot0_gripper_qpos"]
    print(f"\n===== {gripper} =====")
    print(f"  robot0_gripper_qpos 长度 = {len(gq)}   值 = {np.round(gq, 4).tolist()}")
    try:
        sel = gq[[1, 5]]
        print(f"  main.py 的 gq[[1,5]]      -> OK, {np.round(sel,4).tolist()}  (state 维度 = "
              f"{len(obs['robot0_eef_pos']) + 3 + len(sel)})")
    except Exception as exc:
        print(f"  main.py 的 gq[[1,5]]      -> ❌ {type(exc).__name__}: {exc}")
    try:
        sel2 = gq[:2]
        print(f"  改成 gq[:2]               -> OK, {np.round(sel2,4).tolist()}  (state 维度 = "
              f"{len(obs['robot0_eef_pos']) + 3 + len(sel2)})")
    except Exception as exc:
        print(f"  改成 gq[:2]               -> ❌ {type(exc).__name__}: {exc}")
    env.close()
print("\nDEMO_OK")
