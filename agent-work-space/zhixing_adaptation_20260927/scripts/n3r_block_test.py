#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3-redo 主测试：用**标准实心试块**标定 ZhiXing 的抓持能力。

为什么重做 N3
-------------
已有 N3 用任务物体（akita 黑碗，薄壁、外径 111.8 mm）测夹爪，把四个变量混在一起：
  ① 物体模型缺陷  ② 物体重摆放不稳（20 个落点全异常）  ③ 薄壁力学  ④ 夹爪接触模型
结论 0/10 无法归因到夹爪本身。

本脚本按 N3 §5.1 的建议，改用**已知几何的实心方块**（无薄壁、无搬运、确定性重置），
把 ①②③ 全部消掉，只留 ④ ——这样测的才是夹爪。

注入方式（**不改任何生产资产**）
--------------------------------
`libero` 的 arena 是 `TableArena`（=`robosuite.models.base.MujocoXML`），
其在 env 构造时把 worldbody 交给最终 MJCF 编译。故在测试进程内 monkeypatch
`TableArena.__init__`，在 super() 之后 `merge()` 一个只含试块的临时 MJCF，
试块即随 arena 一起进入编译后的模型。退出进程即失效，磁盘资产零改动。

真机约束（已核实，2026-09-27）
------------------------------
ChangingTek CTAG2F90「90 Collaborative Two Finger Hands」，
**Maximum travel distance = 90 mm**，最大夹持力 23 N。
与仿真 N2 权威口径的内面净距 89.96 mm 一致 → 90 mm 开口是**正确的**，
碗（111.8 mm）在真机上同样抓不了，属硬件行程限制，不是仿真 bug。
"""
import argparse
import json
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"
BLOCK = "n3r_block"
GRIP_CTRL_CLOSED = -0.91      # N1 极性修复后的闭合端
GRIP_CTRL_OPEN = 0.0

BLOCK_XML = """<mujoco model="n3r_test_block">
  <worldbody>
    <body name="{name}" pos="{x} {y} {z}">
      <freejoint name="{name}_joint"/>
      <geom name="{name}_geom" type="box" size="{hx} {hy} {hz}"
            rgba="0.85 0.22 0.18 1" contype="1" conaffinity="1"
            friction="1 0.005 0.0001" solref="0.02 1" condim="4"
            density="700"/>
    </body>
  </worldbody>
</mujoco>
"""


_PATCHED = {"done": False}
_TMP_XML = pathlib.Path(tempfile.gettempdir()) / "n3r_block.xml"


def install_block_injection(size_m, pos):
    """在测试进程内 monkeypatch TableArena，把试块并进 arena 的 worldbody。

    只 patch 一次（`_PATCHED` 守卫）：多宽度循环里若重复包裹，会累积成多个试块。
    改尺寸只需重写同一个临时 XML —— `MujocoXML` 在每次 arena 构造时重新读取。
    """
    import libero.libero.envs.arenas.table_arena as TA
    from robosuite.models.base import MujocoXML

    hx, hy, hz = size_m
    _TMP_XML.write_text(
        BLOCK_XML.format(name=BLOCK, x=pos[0], y=pos[1], z=pos[2],
                         hx=hx, hy=hy, hz=hz),
        encoding="utf-8")

    if _PATCHED["done"]:
        return str(_TMP_XML)

    orig = TA.TableArena.__init__

    def patched(self, *a, **kw):
        orig(self, *a, **kw)
        self.merge(MujocoXML(str(_TMP_XML)))

    TA.TableArena.__init__ = patched
    _PATCHED["done"] = True
    return str(_TMP_XML)


def block_id(sim):
    import mujoco

    m = sim.model._model
    for b in range(m.nbody):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "") == BLOCK:
            return b
    return None


def pad_geoms(sim):
    import mujoco

    m = sim.model._model
    out = {}
    for g in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if "left_fingerpad_col" in n:
            out["left"] = g
        elif "right_fingerpad_col" in n:
            out["right"] = g
    return out


def contact_report(env, bid):
    """指垫↔试块的接触：条数、最小有符号距离（负=穿透）、最大法向力。"""
    sim, m, d, _ = N.ids(env)
    pads = pad_geoms(sim)
    pg = set(pads.values())
    n_ct = 0
    min_dist = None
    max_force = 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if int(m.geom_bodyid[g1]) != bid and int(m.geom_bodyid[g2]) != bid:
            continue
        if g1 not in pg and g2 not in pg:
            continue
        n_ct += 1
        dist = float(c.dist)
        min_dist = dist if min_dist is None else min(min_dist, dist)
        f = np.zeros(6)
        try:
            import mujoco

            mujoco.mj_contactForce(m, d, i, f)
            max_force = max(max_force, float(np.linalg.norm(f[:3])))
        except Exception:
            pass
    return {"n_contacts": n_ct,
            "min_dist_mm": None if min_dist is None else min_dist * 1000.0,
            "max_force_N": max_force}


def step_hold_bounded(env, q_arm, gripper_ctrl, n=1):
    """与 N.step_hold 相同，但额外记录夹爪执行器实际输出力的峰值。"""
    import mujoco

    sim, m, d, gid, aj, aadr, adof = N._arm_idx(env)
    aid = N.gripper_actuator(env)
    peak = peak_abs = 0.0
    for _ in range(n):
        N.set_arm(sim, m, d, aadr, adof, q_arm)
        d.ctrl[aid] = float(gripper_ctrl)
        sim.step()
        f = abs(float(d.actuator_force[aid]))
        peak_abs = max(peak_abs, f)
    mujoco.mj_forward(m, d)
    return {"aid": aid, "forcelimited": int(m.actuator_forcelimited[aid]),
            "forcerange": [float(v) for v in m.actuator_forcerange[aid]],
            "peak_abs_actuator_force": peak_abs,
            "final_actuator_force": float(d.actuator_force[aid])}


def left1_qpos(env):
    """夹爪主动关节 Left_1_Joint 的实际角度（0=全开，-0.91=全闭）。"""
    import mujoco

    sim, m, d, _ = N.ids(env)
    for j in range(m.njnt):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        if nm.endswith("Left_1_Joint"):
            return float(d.qpos[int(m.jnt_qposadr[j])])
    return float("nan")


def close_until_stall(env, q_arm, args):
    """闭合到「关节停住」为止。

    ★ 为什么不能固定步数：一旦给执行器加力上限（真机 23 N 等效），
      闭合速度大幅下降 —— 实测 600 步只从 0 走到 -0.097，指垫**根本没碰到试块**，
      于是抬升时夹爪还是张开的，试块当然带不走。这是台架时序问题，不是夹爪问题。
    """
    l1_prev, stable, total = None, 0, 0
    act = None
    while total < args.close_max_steps:
        act = step_hold_bounded(env, q_arm, GRIP_CTRL_CLOSED, n=args.close_chunk)
        total += args.close_chunk
        l1 = left1_qpos(env)
        if l1_prev is not None and abs(l1 - l1_prev) < 1e-4:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0
        l1_prev = l1
    act = act or {}
    act["close_steps_used"] = total
    act["left_1_qpos_after_close"] = left1_qpos(env)
    return act


def run_one(env, size_m, args, trial_seed):
    """单次试验：接近 → 闭合 → 抬升 5 cm → 保持 2 s → 放回 → 张开。"""
    env.reset()
    sim, m, d, gid = N.ids(env)
    # ★ 必须在 env.reset() **之后**打补丁：实测 reset 会把 model 数组恢复
    #   （未打补丁时 trial 内读到 forcelimited=0 / forcerange=[0,0]，
    #    而 peak|actuator_force| 高达 60，等于限力根本没生效）。
    if getattr(args, "torque_cap", None) is not None:
        cap = float(args.torque_cap)
        aid_c = N.gripper_actuator(env)
        m.actuator_forcerange[aid_c] = [-cap, cap]
        m.actuator_forcelimited[aid_c] = 1
    bid = block_id(sim)
    if bid is None:
        raise RuntimeError("试块未注入成功")

    # 让试块落稳
    for _ in range(args.settle_steps):
        env.step(np.concatenate([np.zeros(6), [GRIP_CTRL_OPEN]]))
    sim, m, d, gid = N.ids(env)
    pts = N.body_world_pts(sim, bid, collision_only=True)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    cx, cy = float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1]))
    z_lo, z_hi = float(lo[2]), float(hi[2])
    on_table = z_lo > 0.5          # 桌面高度量级；掉到地面会是 ~0.0

    R_ref = N.gripsite(sim, m, d, gid)[1].copy()
    _, _, _, _, _, aadr, adof = N._arm_idx(env)
    q_home = np.array([d.qpos[x] for x in aadr], dtype=float)

    # ★ 基准位姿必须在**接近之前**测：接近过程可能已经碰到试块
    obj0 = N.body_world_pts(sim, bid).mean(axis=0)

    # 抓取高度：夹持点需高于桌面 >= 指尖超出量(19.9mm)
    z_g = min(z_lo + args.grasp_above_table, z_hi - 0.005)
    rng = np.random.default_rng(trial_seed)
    jitter = (rng.random(2) - 0.5) * 2 * args.jitter_mm / 1000.0
    pin = np.array([cx + jitter[0], cy, z_g])

    rec = {"trial": None, "size_mm": [v * 1000 for v in size_m],
           "block_lo": lo.tolist(), "block_hi": hi.tolist(),
           "on_table": bool(on_table), "z_g": float(z_g), "pin": pin.tolist()}

    if not on_table:
        rec.update({"success": False, "reason": "block_not_on_table"})
        return rec

    q_pre, res_pre = N.ik_goto(env, q_home, pin + np.array([0, 0, 0.06]),
                               R_ref, GRIP_CTRL_OPEN, settle=15)
    q_g, res_g = N.ik_goto(env, q_pre, pin, R_ref, GRIP_CTRL_OPEN, settle=15)
    rec["ik_res_mm"] = {"pre": res_pre, "grasp": res_g}

    # 闭合（直驱执行器）。加力上限后闭合很慢，故「闭合到停住」
    act = close_until_stall(env, q_g, args)
    rec["actuator"] = act
    jaw_close = N.jaw_face_separation_mm(env)
    sim2, m2, d2, _ = N.ids(env)
    obj_close = N.body_world_pts(sim2, bid).mean(axis=0)
    rec["jaw_close_mm"] = jaw_close
    rec["obj_shift_on_close_mm"] = float(np.linalg.norm(obj_close - obj0) * 1000)
    rec["contact_close"] = contact_report(env, bid)
    rec["max_qvel"] = float(np.abs(N.ids(env)[2].qvel).max())

    # 抬升 5 cm —— 必须**准静态**（分步爬升）。
    # 一次 ik_goto 到 +5cm 会让 arm qpos 在单个 2ms 子步内"瞬移"，软接触传不了
    # 这个冲量，指垫会直接从物体侧面滑上去，物体原地不动 —— 那是**台架假象**，
    # 不是夹爪抓不住。故按 --lift-ramp 步逐步抬升，每步之间正常步进物理。
    q_prev, res_up = q_g, 0.0
    if args.lift_mode == "osc":
        # ★ 力驱动抬升：用 robosuite 的 OSC 控制器闭环移动（真实控制路径）。
        #   ik_ramp 是"把臂 qpos 直接传送"，属于运动学覆盖，接触求解器看到的是
        #   位置阶跃而非力；把它当成"夹爪抓不住"可能是**台架假象**。
        _s, _m, _d, _g = N.ids(env)
        gs0 = np.array(_d.site_xpos[_g], dtype=float)
        osc = N.drive_to(env, gs0 + np.array([0, 0, args.lift_m]), +1.0,
                         max_steps=args.osc_steps, tol=0.003)
        rec["osc_lift"] = osc
        res_up = osc.get("final_err_mm") or 0.0
    else:
        n_ramp = max(1, args.lift_ramp)
        for k in range(1, n_ramp + 1):
            tgt = pin + np.array([0, 0, args.lift_m * k / n_ramp])
            q_prev, res_up = N.ik_goto(env, q_prev, tgt, R_ref, GRIP_CTRL_CLOSED,
                                       settle=args.ramp_settle)
    sim3, _, _, _ = N.ids(env)
    obj_lift = N.body_world_pts(sim3, bid).mean(axis=0)
    # 保持 2 s（20 Hz → 40 步）
    if args.lift_mode == "osc":
        N.hold(env, +1.0, args.hold_steps)
    else:
        N.step_hold(env, q_prev, GRIP_CTRL_CLOSED, n=args.hold_steps)
    sim4, _, _, _ = N.ids(env)
    obj_hold = N.body_world_pts(sim4, bid).mean(axis=0)
    rec["ik_res_up_mm"] = res_up
    rec["dz_lift_mm"] = float(obj_lift[2] - obj0[2]) * 1000
    rec["dz_hold_mm"] = float(obj_hold[2] - obj0[2]) * 1000
    rec["contact_hold"] = contact_report(env, bid)

    # 放回 + 张开
    N.ik_goto(env, q_prev, pin, R_ref, GRIP_CTRL_CLOSED, settle=15)
    N.step_hold(env, q_g, GRIP_CTRL_OPEN, n=args.close_steps)
    sim5, _, _, _ = N.ids(env)
    obj_end = N.body_world_pts(sim5, bid).mean(axis=0)
    rec["dz_end_mm"] = float(obj_end[2] - obj0[2]) * 1000

    ok = rec["dz_hold_mm"] > 30.0
    rec["success"] = bool(ok)
    rec["reason"] = "hold" if ok else "no_hold"
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths-mm", default="30,50,70",
                    help="试块沿闭合轴(世界 y)的宽度，逗号分隔；用于给出抓取包络")
    ap.add_argument("--height-mm", type=float, default=70.0)
    ap.add_argument("--depth-mm", type=float, default=60.0)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--place-x", type=float, default=-0.22)
    ap.add_argument("--place-y", type=float, default=0.10)
    ap.add_argument("--drop-z", type=float, default=1.30)
    ap.add_argument("--settle-steps", type=int, default=400)
    ap.add_argument("--close-steps", type=int, default=600,
                    help="仅在未启用 --close-stall 时使用")
    ap.add_argument("--close-max-steps", type=int, default=8000,
                    help="「闭合到停住」的步数上限")
    ap.add_argument("--close-chunk", type=int, default=50,
                    help="每检查一次停住与否所步进的步数")
    ap.add_argument("--hold-steps", type=int, default=40)
    ap.add_argument("--lift-m", type=float, default=0.05)
    ap.add_argument("--lift-mode", choices=["ik_ramp", "osc"], default="ik_ramp",
                    help="osc = 用力驱动的 OSC 控制器抬升（真实控制路径）")
    ap.add_argument("--osc-steps", type=int, default=300)
    ap.add_argument("--lift-ramp", type=int, default=25,
                    help="抬升分几步爬升（准静态）。1 = 旧行为（单步瞬移，会产生台架假象）")
    ap.add_argument("--ramp-settle", type=int, default=8,
                    help="每个爬升子步之间步进的 env step 数")
    ap.add_argument("--grasp-above-table", type=float, default=0.035)
    ap.add_argument("--jitter-mm", type=float, default=0.0)
    ap.add_argument("--torque-cap", type=float, default=None,
                    help="夹爪执行器力矩上限。不设=无限力（现状），会把物体压穿")
    ap.add_argument("--tag", default="N3R_block")
    a = ap.parse_args()

    widths = [float(v) for v in a.widths_mm.split(",") if v.strip()]
    hz = a.height_mm / 2000.0
    hx = a.depth_mm / 2000.0

    summary = {"node": "N3-redo", "tag": a.tag, "method": "standard_solid_block",
               "injection": "monkeypatch TableArena + MujocoXML.merge (no asset change)",
               "opening_mm": 89.96, "threshold_per_width": 9,
               "hold_steps": a.hold_steps, "lift_m": a.lift_m,
               "results": []}

    for w in widths:
        hy = w / 2000.0
        size_m = (hx, hy, hz)
        install_block_injection(size_m, (a.place_x, a.place_y, a.drop_z))
        env, task = N.build(task_id=0)
        if a.torque_cap is not None:
            sim_c, m_c, _, _ = N.ids(env)
            aid_c = N.gripper_actuator(env)
            m_c.actuator_forcerange[aid_c] = [-a.torque_cap, a.torque_cap]
            m_c.actuator_forcelimited[aid_c] = 1
            print(f"[grip] 执行器力矩上限 = ±{a.torque_cap:.3f}"
                  f"（等效夹持力约 {a.torque_cap * 17.3:.1f} N；真机 CTAG2F90 额定 23 N）")
        print(f"\n{'='*76}\n[block] y-width={w:.0f} mm  size(x,y,z)={a.depth_mm:.0f}x{w:.0f}x{a.height_mm:.0f} mm"
              f"  place=({a.place_x},{a.place_y})  task0='{task.language}'")
        print(f"[method] 脚本专家 + 底层控制；π0.5 未加载")

        trials, n_ok = [], 0
        for t in range(a.trials):
            try:
                r = run_one(env, size_m, a, trial_seed=1000 + t)
            except Exception as e:                       # noqa: BLE001
                r = {"trial": t + 1, "success": False, "reason": f"exception:{e}"}
            r["trial"] = t + 1
            trials.append(r)
            n_ok += int(r.get("success", False))
            cc = r.get("contact_close", {}) or {}
            print(f"  [{t+1:2d}/{a.trials}] IK={r.get('ik_res_mm',{}).get('grasp', float('nan')):5.2f}mm "
                  f"on_table={int(r.get('on_table', False))} "
                  f"闭合净距={r.get('jaw_close_mm', float('nan')):6.2f}mm "
                  f"推挤={r.get('obj_shift_on_close_mm', float('nan')):6.1f}mm "
                  f"接触={cc.get('n_contacts','-')} 穿透={cc.get('min_dist_mm','-')} "
                  f"抬升={r.get('dz_lift_mm', float('nan')):+7.1f} 保持={r.get('dz_hold_mm', float('nan')):+7.1f}mm"
                  f"  {'OK 成功' if r.get('success') else 'XX ' + str(r.get('reason'))}")

        passed = (n_ok / a.trials) >= 0.9
        summary["results"].append({"width_mm": w, "n_success": n_ok, "n_total": a.trials,
                                   "success_rate": n_ok / a.trials, "pass": bool(passed),
                                   "trials": trials})
        print(f"  → 宽度 {w:.0f} mm：成功 {n_ok}/{a.trials}  "
              f"{'PASS' if passed else 'FAIL'}")
        env.close()

    best = max(summary["results"], key=lambda r: r["n_success"])
    summary["best_width_mm"] = best["width_mm"]
    summary["best_success"] = f"{best['n_success']}/{best['n_total']}"
    summary["any_width_pass"] = bool(any(r["pass"] for r in summary["results"]))
    summary["verdict"] = "PASS" if summary["any_width_pass"] else "NOT_PASSED"

    pathlib.Path(OUT).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(OUT) / f"n3r_{a.tag}.json"
    p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{'='*76}\n[verdict] {summary['verdict']}  best={best['width_mm']:.0f}mm "
          f"{best['n_success']}/{best['n_total']}")
    print(f"[out] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
