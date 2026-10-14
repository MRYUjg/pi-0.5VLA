#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 ZhiXing 夹爪速度默认值改动（只读）。

分别在**不设置** ZHIXING_GRIPPER_SPEED 与显式设置两种情况下，走真实
`env.step([0]*6 + [grip])` 路径，记录 `gripper.speed`、闭合到位的步数、
以及等效闭合线速度，确认默认值改动确实生效。

用法：
  python verify_zhixing_speed_default.py --out <json>            # 用资产默认
  ZHIXING_GRIPPER_SPEED=0.2 python ... --label env_0p2 --out ...
"""
import argparse
import datetime
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N              # noqa: E402
import n4_collect as N4                # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--label", default="asset_default")
ap.add_argument("--suite", default="libero_object")
ap.add_argument("--task-id", type=int, default=0)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--max-steps", type=int, default=400)
ap.add_argument("--out", default="/app/agent-work-space/outputs/diag/zhixing_speed_default.json")
a = ap.parse_args()

REC = {
    "node": "VERIFY-ZHIXING-SPEED-DEFAULT",
    "label": a.label,
    "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "env_var_ZHIXING_GRIPPER_SPEED": os.environ.get("ZHIXING_GRIPPER_SPEED"),
    "note": "只读验证：未修改任何文件；走真实 env.step 路径",
}


def log(m):
    print(m, flush=True)


def step_raw(env, cmd):
    try:
        env.env.done = False
    except Exception:                                          # noqa: BLE001
        pass
    return env.step(np.concatenate([np.zeros(6), [cmd]]))


def main():
    env, task = N4.build(a.suite, a.task_id, robot="ECO65", gripper="ZhixingGripper", seed=a.seed)
    g = env.env.robots[0].gripper
    REC["gripper_class"] = type(g).__name__
    REC["speed_effective"] = float(g.speed)
    REC["speed_source"] = ("env var" if REC["env_var_ZHIXING_GRIPPER_SPEED"] is not None
                           else "asset default")
    log("[%s] env=%s → speed=%.4f（来源：%s）"
        % (a.label, REC["env_var_ZHIXING_GRIPPER_SPEED"], REC["speed_effective"], REC["speed_source"]))

    env.reset()
    # 先开到全开
    for _ in range(a.max_steps):
        step_raw(env, -1.0)
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]
    idx = [int(i) for i in r._ref_gripper_joint_pos_indexes]
    joints = [str(j) for j in r.gripper.joints]
    li = next(j for j, nm in enumerate(joints) if nm.endswith("Left_1_Joint"))
    q_open = float(d.qpos[idx[li]])
    REC["q_open"] = round(q_open, 6)

    # 闭合到停住（判定：Left_1 连续 20 步变化 < 1e-6，或 current_action 饱和且不再变）
    steps, hist, prev, stable = 0, [], None, 0
    for i in range(a.max_steps):
        step_raw(env, 1.0)
        steps = i + 1
        q = float(d.qpos[idx[li]])
        if i % 10 == 0 or i < 5:
            hist.append({"i": i, "Left_1": round(q, 6),
                         "current_action": round(float(np.atleast_1d(g.current_action)[0]), 4)})
        # 停住判据：命令累积量已饱和（|current_action|→1）且 Left_1 相邻步变化 < 1e-6
        ca = float(np.atleast_1d(g.current_action)[0])
        if prev is not None and abs(q - prev) < 1e-6 and abs(abs(ca) - 1.0) < 1e-9:
            stable += 1
            if stable >= 20:
                break
        else:
            stable = 0
        prev = q
    q_closed = float(d.qpos[idx[li]])
    REC["closing_steps_to_stop"] = steps
    REC["closing_seconds_to_stop"] = round(steps * float(env.env.control_timestep), 3)
    REC["q_closed"] = round(q_closed, 6)
    REC["joint_stroke_rad"] = round(abs(q_open - q_closed), 6)
    REC["traj_sample"] = hist[:14]
    REC["current_action_end"] = round(float(np.atleast_1d(g.current_action)[0]), 4)
    # 等效闭合线速度：指垫净开口行程 / 闭合时间
    try:
        sys.path.insert(0, "/app/agent-work-space/outputs/scripts")
        import eco65_expert_validation as EX
        EX.install_generic_pad_geoms(lambda m: None)
        gap = EX.jaw_gap_mm(env)
        REC["pad_aperture_closed_mm"] = None if gap is None else round(gap, 4)
    except Exception as e:                                     # noqa: BLE001
        REC["pad_aperture_closed_mm"] = "measure_failed: %s" % e
    # 理论等效速度：全行程 89.5 mm 由 speed 决定所需步数
    REC["expected_steps_full_stroke"] = round(1.0 / max(REC["speed_effective"], 1e-9), 1)
    REC["implied_close_speed_mm_s"] = (
        None if not isinstance(REC["pad_aperture_closed_mm"], float) else
        round(89.5 / max(REC["closing_seconds_to_stop"], 1e-9), 1))
    log("  闭合到停住：%d 步 = %.3f s；Left_1 %.4f → %.4f（行程 %.4f rad）；current_action=%.3f"
        % (steps, REC["closing_seconds_to_stop"], q_open, q_closed,
           REC["joint_stroke_rad"], REC["current_action_end"]))
    log("  期望步数(1/speed)=%.1f  等效闭合线速度≈%s mm/s"
        % (REC["expected_steps_full_stroke"], REC["implied_close_speed_mm_s"]))

    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(REC, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    log("[out] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
