#!/usr/bin/env python3
"""诊断：为什么 gate 脚本里极性测量出现 182 cm 这种不可能的开口值。"""
import os
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
from gripper_polarity import make_env, get_robot, live_sim, jaw, gripper_actuator_ids, sites_ends

env = make_env("ECO65", "ZhixingGripper")
env.reset()
r = get_robot(env)
sim = live_sim(env, r)


def snap(label):
    r = get_robot(env)
    s = live_sim(env, r)
    acts = gripper_actuator_ids(s, r)
    q = np.asarray([s.data.qpos[i] for i in r._ref_gripper_joint_pos_indexes], dtype=float)
    ctrl = [round(float(s.data.ctrl[i]), 4) for i in acts]
    ca = np.asarray(r.gripper.current_action, dtype=float).ravel()
    j, how = jaw(s, "zhixing", r)
    lt = sites_ends(s, "Left_Support_tip", "Left_2_connect")
    # 指尖两点的绝对坐标，确认是否真的分开了
    idx = {}
    for i in range(s.model.nsite):
        n = s.model.site_id2name(i) or ""
        if n.endswith("Left_Support_tip") or n.endswith("Right_Support_tip"):
            idx[n] = np.asarray(s.data.site_xpos[i], dtype=float)
    print("%-22s ctrl=%s cur_act=%s q0=%8.4f jaw=%s m  conn_left=%.2e" %
          (label, ctrl, np.round(ca, 3).tolist(), q[0],
           "None" if j is None else "%.5f" % j, lt))
    for n, p in idx.items():
        print("      %-34s %s" % (n, np.round(p, 4).tolist()))
    return q


snap("after reset")

for k in range(10):
    env.step(np.zeros(7, dtype=float).tolist())
snap("after 10 zero-action")

for k in range(25):
    a = np.zeros(7, dtype=float)
    a[6] = -1.0
    env.step(a.tolist())
snap("after drive(-1,25)")

for k in range(25):
    a = np.zeros(7, dtype=float)
    a[6] = +1.0
    env.step(a.tolist())
snap("after drive(+1,25)")
print("DEBUG_DONE")
