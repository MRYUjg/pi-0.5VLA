#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""措施 C-3：带载 transport/place **定因**（复核 §6 最后一项）。

复核实测：教师 transport 恒耗满 220 步、place 恒耗满 160 步，两段**从未满足 8 mm 容差**；
n4v8 实测 6 次尝试 0 条通过。复核要求："对带载 transport/place 做控制器消融，
记录稳态位置/角度误差、目标可达性及夹持滑移，区分额外 PD、增益、姿态约束与目标点因素。
……不要只缩短等待、增大 MAX_STEPS 或放松成功条件。"

本探针把 transport 段做成**可复现的带载单元测试**：把目标物体放到两片指垫之间 →
闭合 → 抬升 5 cm → 带载伺服到搬运目标点。然后**一次只改一个因素**做消融：

  基线        额外PD=eco65_no_extra_pd(0)  增益=0.55  姿态=保持   目标=篮沿+0.12
  abl_pd      额外PD=legacy(3.0)          同上        同上       同上
  abl_gain    同上                         增益=1.0    同上       同上
  abl_att     同上                         0.55        姿态=不约束 同上
  abl_target  同上                         0.55        保持       目标=篮沿+0.06

每档记录：是否收敛、稳态位置/角度误差、**可达性**（给足步数后能达到的最小误差）、
**夹持滑移**（物体中心相对 grip_site 的位移变化）、指垫穿透峰值。

用法：
  docker compose -f examples/libero/compose.yml run --rm --no-deps runtime \
    /bin/bash -c "source /.venv/bin/activate && export ZHIXING_GRIPPER_SPEED=0.01 && python \
      /app/agent-work-space/zhixing_adaptation_20260927/scripts/verify_c_load_diag.py"
"""
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N      # noqa: E402
import n4_collect as N4        # noqa: E402
import runtime_config as RC    # noqa: E402

OUT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927/results")
OUT.mkdir(parents=True, exist_ok=True)

TOL_M = 0.008          # 与教师 transport 同容差
BUDGET = 220           # 与教师 transport 同步数上限
LONG_BUDGET = 900      # 可达性探测：给足步数
STEADY_N = 10          # 稳态误差取最后 N 步
LIFT_UP = 0.05

VARIANTS = [
    ("baseline",    {"profile": "eco65_no_extra_pd", "gain": 0.55, "attitude": True,  "target_dz": 0.12}),
    ("abl_pd",      {"profile": "legacy",            "gain": 0.55, "attitude": True,  "target_dz": 0.12}),
    ("abl_gain",    {"profile": "eco65_no_extra_pd", "gain": 1.00, "attitude": True,  "target_dz": 0.12}),
    ("abl_att",     {"profile": "eco65_no_extra_pd", "gain": 0.55, "attitude": False, "target_dz": 0.12}),
    ("abl_target",  {"profile": "eco65_no_extra_pd", "gain": 0.55, "attitude": True,  "target_dz": 0.06}),
]


def pads_world(m, d):
    import mujoco

    out = {}
    for g in range(m.ngeom):
        nm = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if "left_fingerpad" in nm:
            out["left"] = np.array(d.geom_xpos[g], float)
        if "right_fingerpad" in nm:
            out["right"] = np.array(d.geom_xpos[g], float)
    return out


def pad_pen_mm(m, d):
    import mujoco

    pads = set()
    for g in range(m.ngeom):
        if "fingerpad" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower():
            pads.add(g)
    worst = 0.0
    for i in range(int(d.ncon)):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in pads or g2 in pads) and float(c.dist) < 0:
            worst = max(worst, -float(c.dist))
    return worst * 1000.0


def run_variant(env, bid, region_bb, cfg, log):
    import mujoco

    os.environ["EXTRA_JOINT_PD_PROFILE"] = cfg["profile"]
    env.seed(1)
    env.reset()
    sim, m, d, gid = N.ids(env)
    r = env.env.robots[0]

    # ---- 摆好带载状态：物体放在两指垫中点，闭合并抬升 5 cm ----
    bodies = N.find_object_bodies(sim, "alphabet_soup")
    _bid, bname = bodies[0]
    jn = N.free_joint_of(sim, bname)
    pw = pads_world(m, d)
    mid = 0.5 * (pw["left"] + pw["right"])
    N.set_free_joint(sim, jn, (float(mid[0]), float(mid[1]), float(mid[2])))
    for _ in range(14):                                   # 张开到端点
        env.step(np.concatenate([np.zeros(6), [-1.0]]))
    for _ in range(40):                                   # 闭合夹住
        env.step(np.concatenate([np.zeros(6), [1.0]]))
    sim, m, d, gid = N.ids(env)
    p0, R0 = N.gripsite(sim, m, d, gid)
    adr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)])
    obj0 = np.array(d.qpos[adr:adr + 3], float)
    off0 = obj0 - p0                                      # 初始夹持偏置
    for _ in range(30):                                   # 抬升
        sim, m, d, gid = N.ids(env)
        p, R = N.gripsite(sim, m, d, gid)
        ap = np.clip((p0 + np.array([0, 0, LIFT_UP]) - p) / N4.TEACHER_GAIN_POS_M, -1, 1)
        ar = np.clip(N.rot_err_vec(R0, R) / N4.TEACHER_GAIN_ROT_RAD, -1, 1)
        env.step(np.concatenate([ap, ar, [1.0]]))
    sim, m, d, gid = N.ids(env)
    p1, R1 = N.gripsite(sim, m, d, gid)
    adr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)])
    loaded_ok = bool(np.linalg.norm(np.array(d.qpos[adr:adr + 3], float) - p1 - off0) < 0.05)

    # ---- 搬运目标（与教师同一算法：篮沿 + dz，侧向留 35%）----
    rb, rh = region_bb
    base = np.array(env.env.sim.data.body_xpos[
        env.env.sim.model.body_name2id("robot0_base")], float)
    frac = 0.35
    near_x = rb[0] + frac * (rh[0] - rb[0]) if base[0] < rb[0] else rh[0] - frac * (rh[0] - rb[0])
    target = np.array([near_x, 0.5 * (rb[1] + rh[1]), float(rh[2]) + cfg["target_dz"]])

    # ---- 带载伺服，逐帧记录 ----
    errs, rots, slips, pens = [], [], [], []
    conv_at = None
    for k in range(BUDGET):
        sim, m, d, gid = N.ids(env)
        p, R = N.gripsite(sim, m, d, gid)
        e = float(np.linalg.norm(target - p))
        er = float(np.linalg.norm(N.rot_err_vec(R1, R)))
        errs.append(e)
        rots.append(er)
        pens.append(pad_pen_mm(m, d))
        adr = int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)])
        objc = np.array(d.qpos[adr:adr + 3], float)
        slips.append(float(np.linalg.norm((objc - p) - off0) * 1000.0))
        if conv_at is None and e < TOL_M:
            conv_at = k + 1
        ap = np.clip(cfg["gain"] * (target - p) / N4.TEACHER_GAIN_POS_M, -1.0, 1.0)
        ar = (np.clip(N.rot_err_vec(R1, R) / N4.TEACHER_GAIN_ROT_RAD, -1.0, 1.0)
              if cfg["attitude"] else np.zeros(3))
        env.step(np.concatenate([ap, ar, [1.0]]))

    # ---- 可达性：给足步数后能达到的最小误差 ----
    reach_errs = []
    for _ in range(LONG_BUDGET):
        sim, m, d, gid = N.ids(env)
        p, R = N.gripsite(sim, m, d, gid)
        e = float(np.linalg.norm(target - p))
        reach_errs.append(e)
        if e < 1e-4 or not np.isfinite(e):
            break
        ap = np.clip((target - p) / N4.TEACHER_GAIN_POS_M, -1.0, 1.0)
        ar = np.clip(N.rot_err_vec(R1, R) / N4.TEACHER_GAIN_ROT_RAD, -1.0, 1.0)
        env.step(np.concatenate([ap, ar, [1.0]]))

    return {
        "variant": None, "profile": cfg["profile"], "gain": cfg["gain"],
        "attitude_held": cfg["attitude"], "target_dz": cfg["target_dz"],
        "loaded_ok": loaded_ok,
        "converged_within_budget": bool(conv_at is not None),
        "converged_at_step": conv_at,
        "steady_pos_err_mm": float(np.mean(errs[-STEADY_N:]) * 1000.0),
        "steady_rot_err_deg": float(np.degrees(np.mean(rots[-STEADY_N:]))),
        "min_pos_err_mm_reachability": float(np.min(reach_errs) * 1000.0),
        "reachable": bool(np.min(reach_errs) < TOL_M),
        "slip_mm_final": float(slips[-1]),
        "slip_mm_peak": float(np.max(slips)),
        "penetration_peak_mm": float(np.max(pens)),
        "target": [float(v) for v in target],
    }


def main():
    RC.apply()
    env, task = N4.build("libero_object", 0, seed=1)
    # 探针要跑 84 步准备 + 220 步运输 + 最长 900 步可达性探测，远超默认 horizon=1000；
    # 且原生谓词成立会让 robosuite 抛 "executing action in terminated episode"。
    # 定因探针只关心控制跟踪，故关闭 done 终止并放大 horizon（不改任何生产文件）。
    env.env.ignore_done = True
    env.env.horizon = 1000000
    sim, m, d, gid = N.ids(env)
    obj_name, region_name = N4.goal_objects(env)
    bid, bname = N4.find_body(sim, obj_name)
    rid, rname = N4.find_body(sim, region_name)
    region_bb = N4.bbox(sim, rid)
    print("[C-3] 目标=%s 区域=%s 篮 z 上沿=%.4f" % (bname, rname, region_bb[1][2]))

    out = {"node": "C-3", "purpose": "带载 transport/place 定因（复核 §6）",
           "runtime": RC.fingerprint(), "tol_m": TOL_M, "budget": BUDGET,
           "results": [], "checks": {}}
    for name, cfg in VARIANTS:
        try:
            r = run_variant(env, bid, region_bb, cfg, print)
        except Exception as e:                            # noqa: BLE001
            r = {"variant": name, "error": "%s: %s" % (type(e).__name__, e),
                 "profile": cfg["profile"], "gain": cfg["gain"],
                 "attitude_held": cfg["attitude"], "target_dz": cfg["target_dz"]}
        r["variant"] = name
        out["results"].append(r)
        if "error" in r:
            print("  %-11s 异常：%s" % (name, r["error"]))
        else:
            print("  %-11s 收敛=%-5s(第%s步)  稳态位置=%7.2f mm  稳态姿态=%5.2f°  "
                  "可达最小=%7.2f mm  滑移(峰)=%6.2f mm  穿透=%5.2f mm"
                  % (name, r["converged_within_budget"], r["converged_at_step"],
                     r["steady_pos_err_mm"], r["steady_rot_err_deg"],
                     r["min_pos_err_mm_reachability"], r["slip_mm_peak"],
                     r["penetration_peak_mm"]))
    env.close()

    by = {r["variant"]: r for r in out["results"]}
    base = by.get("baseline", {})
    out["checks"]["baseline_reproduces_nonconvergence"] = bool(
        base and "steady_pos_err_mm" in base
        and not base.get("converged_within_budget"))
    # 逐因素归因：改这一项后是否变好
    for k in ("abl_pd", "abl_gain", "abl_att", "abl_target"):
        v = by.get(k, {})
        out["checks"]["%s_improves" % k] = bool(
            v and "steady_pos_err_mm" in v and base.get("steady_pos_err_mm") is not None
            and v["steady_pos_err_mm"] < base["steady_pos_err_mm"])
    out["checks"]["target_reachable_under_some_config"] = bool(
        any(r.get("reachable") for r in out["results"] if "error" not in r))
    out["verdict"] = "DIAGNOSED" if all(
        isinstance(v, bool) for v in out["checks"].values()) else "INCOMPLETE"

    p = OUT / "c3_load_diag.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print("-" * 78)
    if base and "steady_pos_err_mm" in base:
        print("基线稳态位置误差 = %.2f mm（容差 %.0f mm）→ 复现未收敛：%s"
              % (base["steady_pos_err_mm"], TOL_M * 1000,
                 out["checks"]["baseline_reproduces_nonconvergence"]))
    else:
        print("基线异常，未取得稳态误差：%s" % base.get("error"))
    print("各因素单独改善：%s" % json.dumps(
        {k: out["checks"][k] for k in out["checks"] if k.endswith("_improves")},
        ensure_ascii=False))
    print("[out] %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
