# -*- coding: utf-8 -*-
"""命令→ctrl→关节→开口 映射 + 空载开闭重复性（只读，精简快速版）。"""
import json, os, sys, datetime
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
def joint_layout():
    r, g, sim, m, d, gid = fresh()
    idx = [int(i) for i in r._ref_gripper_joint_pos_indexes]
    jn = [str(j) for j in g.joints]
    li = next(k for k, nm in enumerate(jn) if nm.endswith("Left_1_Joint"))
    return idx, li

def step_raw(cmd):
    try: env.env.done = False
    except Exception: pass
    return env.step(np.concatenate([np.zeros(6), [cmd]]))

def fresh():
    """env.reset() 会重建 robot 对象；必须每次重新取，否则读到旧对象的冻结状态。"""
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    return r, r.gripper, sim, m, d, gid

def snap(extra=None):
    r, g, sim, m, d, gid = fresh()
    s = {"current_action": round(float(np.atleast_1d(g.current_action)[0]), 4),
         "ctrl": round(float(np.atleast_1d(d.ctrl[6])[0]), 4),
         "Left_1": round(float(d.qpos[joint_layout()[0][joint_layout()[1]]]), 4),
         "gripper_qpos": [round(float(d.qpos[i]), 4) for i in joint_layout()[0]],
         "aperture_mm": None if EX.jaw_gap_mm(env) is None else round(EX.jaw_gap_mm(env), 4),
         "min_clearance_mm": None if EX.min_clearance_mm(env) is None else round(EX.min_clearance_mm(env), 4)}
    if extra: s.update(extra)
    return s

RES = {"node": "ZHIXING-MAP-REPEAT", "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
       "gripper_speed_env": os.environ["ZHIXING_GRIPPER_SPEED"], "note": "只读：未修改任何文件",
       "speed_effective": float(fresh()[1].speed)}

# A. 命令 → 状态（从全开出发，命令前后各复位到全开）
print("----- A. 命令映射（每个命令都从全开出发） -----", flush=True)
A = []
for cmd in (-1.0, -0.5, 0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
    env.reset()
    for _ in range(200): step_raw(-1.0)
    before = snap()
    for _ in range(300): step_raw(cmd)
    row = snap({"cmd": cmd, "aperture_before_mm": before["aperture_mm"]})
    A.append(row)
    print("  cmd=%+.1f → action=%+.2f ctrl=%+.3f Left_1=%+.4f 开口 %s → %s mm" %
          (cmd, row["current_action"], row["ctrl"], row["Left_1"],
           row["aperture_before_mm"], row["aperture_mm"]), flush=True)
RES["A_command_map"] = A

# B. 动态开合轨迹
print("----- B. 动态：reset → +1 → −1 -----", flush=True)
env.reset()
close_tr = []
for i in range(320):
    step_raw(1.0)
    if i % 20 == 0 or i < 3:
        close_tr.append(snap({"i": i}))
env.reset()
for _ in range(200): step_raw(-1.0)
open_tr = []
for i in range(320):
    step_raw(-1.0)
    if i % 20 == 0 or i < 3:
        open_tr.append(snap({"i": i}))
RES["B_dynamic"] = {"close_traj": close_tr, "open_traj": open_tr,
                    "closed_aperture_mm": close_tr[-1]["aperture_mm"],
                    "open_aperture_mm": open_tr[-1]["aperture_mm"]}
print("  闭合 320 步后开口 = %s mm；全开 = %s mm" %
      (close_tr[-1]["aperture_mm"], open_tr[-1]["aperture_mm"]), flush=True)

# C. 空载开闭 ×10（两种起始状态）
print("----- C. 空载开闭 ×10 -----", flush=True)
reps = {}
for tag, force_open in (("from_reset", False), ("from_full_open", True)):
    opens, closes, mins = [], [], []
    for _ in range(10):
        env.reset()
        if force_open:
            for _ in range(200): step_raw(-1.0)
        for _ in range(300): step_raw(-1.0)
        so = snap(); opens.append(so["aperture_mm"])
        for _ in range(300): step_raw(1.0)
        sc = snap(); closes.append(sc["aperture_mm"]); mins.append(sc["min_clearance_mm"])
    reps[tag] = {"n": 10,
                 "open_mm": {"mean": float(np.mean(opens)), "std": float(np.std(opens)),
                             "min": float(np.min(opens)), "max": float(np.max(opens)),
                             "range": float(np.max(opens) - np.min(opens))},
                 "closed_mm": {"mean": float(np.mean(closes)), "std": float(np.std(closes)),
                               "min": float(np.min(closes)), "max": float(np.max(closes)),
                               "range": float(np.max(closes) - np.min(closes))},
                 "closed_min_clearance_mean_mm": float(np.mean(mins)),
                 "pass_n3_criterion": bool(np.std(opens) < 0.5 and np.std(closes) < 0.5),
                 "raw_open_mm": [round(v, 4) for v in opens],
                 "raw_closed_mm": [round(v, 4) for v in closes]}
    print("  [%s] 开 %.3f±%.3f (极差 %.3f) / 闭 %.3f±%.3f (极差 %.3f) 判据=%s" %
          (tag, reps[tag]["open_mm"]["mean"], reps[tag]["open_mm"]["std"], reps[tag]["open_mm"]["range"],
           reps[tag]["closed_mm"]["mean"], reps[tag]["closed_mm"]["std"], reps[tag]["closed_mm"]["range"],
           reps[tag]["pass_n3_criterion"]), flush=True)
RES["C_repeatability"] = reps
out = "/app/agent-work-space/zhixing_gripper_solve/evidence/zhixing_control_mapping.json"
open(out, "w", encoding="utf-8").write(json.dumps(RES, ensure_ascii=False, indent=1, default=str))
print("[out]", out, flush=True)
