# -*- coding: utf-8 -*-
"""小命令是否会停在中间开度？（只读）逐命令从**全开**出发保持 300 步。"""
import json, os, sys
import numpy as np
sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
sys.path.insert(0, "/app/agent-work-space/zhixing_gripper_solve/scripts")
import eco65_expert_validation as EX
import n3_grasp_test as N
import n4_collect as N4

os.environ.setdefault("ZHIXING_GRIPPER_SPEED", "0.01")
env, task = N4.build("libero_object", 0, robot="ECO65", gripper="ZhixingGripper", seed=7)
EX.install_generic_pad_geoms(lambda m: None)
EX.install_generic_state_gripper(env, lambda m: None)
sim, m, d, gid = N.ids(env)
g = env.env.robots[0].gripper

def step_raw(cmd):
    try: env.env.done = False
    except Exception: pass
    return env.step(np.concatenate([np.zeros(6), [cmd]]))

rows = []
for cmd in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
    env.reset()
    for _ in range(300): step_raw(-1.0)          # 先全开
    a0 = EX.jaw_gap_mm(env)
    for _ in range(300): step_raw(cmd)            # 再发小命令
    ca = float(np.atleast_1d(g.current_action)[0])
    rows.append({"cmd": cmd, "aperture_before_mm": round(a0, 3),
                 "aperture_after_mm": round(EX.jaw_gap_mm(env), 3),
                 "current_action": round(ca, 4),
                 "ctrl": round(float(np.atleast_1d(d.ctrl[6])[0]), 4),
                 "Left_1": round(float(d.qpos[int(env.env.robots[0]._ref_gripper_joint_pos_indexes[0])]), 4)})
    print("cmd=%+.2f → current_action=%+.3f ctrl=%+.4f 开口 %.3f → %.3f mm" %
          (cmd, ca, rows[-1]["ctrl"], rows[-1]["aperture_before_mm"], rows[-1]["aperture_after_mm"]), flush=True)
print(json.dumps({"note": "只读：小命令是否停在中间开度（从全开出发）", "rows": rows}, ensure_ascii=False, indent=1))
