#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4 诊断：为什么 `n4_collect` 里夹爪一 step 就砸到底，而 `n4_probe_grasp_geometry`
的扫描里却是渐进的？

现象
----
`n4_collect` 闭合第 1 步 `Left_1` 就从 -0.0008 跳到 **-0.5400**，第 2 步
**-1.0183**（超出关节 range [-0.91, 0]）——这是 N3 根因诊断里
「执行器无力上限 → 位置伺服压过关节限位」的**未限力**签名。
而同一 XML、同一 `C.build()` 的扫描里，闭合用 37 步、停在物体宽度 63.2 mm。

若限力生效，全行程应约 1 s（N3 实测 0.976 s ≈ 20 env step）。
本脚本直接读**编译后模型**的执行器属性与夹爪 `speed`，并逐 step 追踪
`Left_1`，以定位差异来源。只读探测。
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N          # noqa: E402
import n4_collect as C             # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"


def actuator_info(env, tag):
    import mujoco

    sim, m, d, gid = N.ids(env)
    aid = N.gripper_actuator(env)
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or ""
    info = {
        "tag": tag,
        "actuator_name": nm,
        "forcelimited": int(m.actuator_forcelimited[aid]),
        "forcerange": [float(v) for v in m.actuator_forcerange[aid]],
        "ctrllimited": int(m.actuator_ctrllimited[aid]),
        "ctrlrange": [float(v) for v in m.actuator_ctrlrange[aid]],
        "gainprm_kp": [float(v) for v in m.actuator_gainprm[aid][:3]],
        "nq_gripper": 6,
    }
    # 夹爪对象上的 speed（N1 之后仍可用环境变量覆盖）
    try:
        g = env.env.robots[0].gripper
        info["gripper_class"] = type(g).__name__
        info["speed"] = float(g.speed)
    except Exception as e:                                   # noqa: BLE001
        info["speed_error"] = str(e)
    print("[%s] actuator=%s forcelimited=%d forcerange=%s ctrlrange=%s "
          "speed=%s"
          % (tag, nm, info["forcelimited"], info["forcerange"],
             info["ctrlrange"], info.get("speed", info.get("speed_error"))))
    return info


def close_trace(env, label, n=12):
    """从全开起，命令闭合 n 步，记录每步 Left_1 与夹爪净距。"""
    import mujoco

    N.hold(env, -1.0, 90)                       # 先全开
    sim, m, d, gid = N.ids(env)
    trace = []
    for k in range(n):
        p, R = N.gripsite(sim, m, d, gid)
        a = np.concatenate([np.zeros(3), np.zeros(3), [1.0]])
        env.step(a)
        sim, m, d, gid = N.ids(env)
        aid = N.gripper_actuator(env)
        trace.append({
            "step": k + 1,
            "Left_1": round(C.left1(env), 5),
            "actuator_force": round(float(d.actuator_force[aid]), 4),
            "jaw_sep_mm": round(N.jaw_face_separation_mm(env) or -1.0, 2),
        })
        print("    [%s step %2d] Left_1=%+.5f  act_force=%+8.4f  sep=%6.2f mm"
              % (label, k + 1, trace[-1]["Left_1"],
                 trace[-1]["actuator_force"], trace[-1]["jaw_sep_mm"]))
    return trace


def main():
    res = {"node": "N4-diag", "check": "actuator_force_limit"}
    env, task = C.build("libero_object", 0, seed=1)

    print("=" * 78)
    print("[A] 刚 build，未 reset")
    res["after_build"] = actuator_info(env, "build")

    print("\n[B] env.reset() 之后")
    env.reset()
    res["after_reset"] = actuator_info(env, "reset")
    res["trace_after_reset"] = close_trace(env, "reset")

    print("\n[C] seed + reset（与采集器一致）")
    env.seed(1)
    env.reset()
    res["after_seed_reset"] = actuator_info(env, "seed_reset")
    res["trace_after_seed_reset"] = close_trace(env, "seed+reset")

    print("\n[D] 再跑一次 seed + reset，看是否稳定复现")
    env.seed(7)
    env.reset()
    res["after_seed7_reset"] = actuator_info(env, "seed7")
    res["trace_seed7"] = close_trace(env, "seed7")

    # 判定：限力生效时，单步 |ΔLeft_1| 应远小于 0.5（全行程 0.91 需 ~20 step）
    steps = [t["Left_1"] for t in res["trace_after_seed_reset"]]
    dmax = max(abs(steps[i + 1] - steps[i]) for i in range(len(steps) - 1)) \
        if len(steps) > 1 else None
    res["max_delta_left1_per_step"] = dmax
    res["force_limit_effective"] = bool(
        res["after_seed_reset"]["forcelimited"] == 1
        and res["after_seed_reset"]["forcerange"] == [-2.0, 2.0])
    res["verdict"] = "FORCE_LIMIT_OK" if res["force_limit_effective"] \
        else "FORCE_LIMIT_MISSING"

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(OUT) / "n4_actuator_check.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + "=" * 78)
    print("[verdict] %s  单步最大 |ΔLeft_1| = %s" % (res["verdict"], dmax))
    print("[out] %s" % p)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
