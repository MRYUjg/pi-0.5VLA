#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N4 前置验证：确认 N2 的 TCP 修复已生效，并**实测**脚本专家所需的新标定常数。

验证项
------
1. 编译模型里 `eef` body 的 pos 是否已是 `0 0 0.20254`。
2. **夹持点是否与 `grip_site` 重合**（这是修复的判据本身）：
   在开 / 闭两个位形下，测量两指垫**内侧面中点**沿接近轴相对 `grip_site` 的偏移。
   闭合位应 ≈ 0；全开位应 ≈ −17.43 mm（行程内后退）。
3. **reset 初始位姿**：ZhiXing 的夹持点是否落在 Panda 的夹持点上。
   同时建 Panda + PandaGripper 参考环境，实测其 `grip_site` 与夹持点。
4. 关节余量、open/close 净距（应与 N3 的 89.89 / 0.65 mm 一致）。

为什么必须实测而不是照抄 N2 的 82.54
------------------------------------
N2 的 82.54 mm 是在**旧** `grip_site`（手指根部）下量的偏移。修复把它前移后，
该偏移按定义应归零——但这依赖「XML 的 eef pos 改动确实传递到编译模型」。
本脚本直接从**编译后的 MuJoCo 模型**读数，不信任任何中间假设。

测量口径（沿用 N2 权威口径，避免「最近点对」退化）
--------------------------------------------------
ZhiXing 是二连**平行连杆**，两指垫内面在行程内保持平行（夹角 2.71°~3.22°）。
平行平面间最近点对是**退化**的，`argmin` 会跳到网格角点、低估约 16 mm。
故取「各自朝对方**最近的一薄层**顶点的质心」，再取两点中点作为夹持点。
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"
RES = 128
SETTLE = 90          # env step；全行程 ~1500 子步 = 60 env step，留量
CTRL_OPEN = 0.0
CTRL_CLOSED = -0.91
SLAB_M = 0.005       # 「最近一薄层」的厚度


def build(suite, task_id, robot, gripper, seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    s = benchmark.get_benchmark_dict()[suite]()
    task = s.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kw = {"bddl_file_name": bddl, "camera_heights": RES, "camera_widths": RES,
          "robots": [robot]}
    if gripper != "default":
        kw["gripper_types"] = gripper
    env = OffScreenRenderEnv(**kw)
    env.seed(seed)
    return env, task


def find_pad_geoms(m):
    """找出两个**指垫**碰撞 geom，不依赖具体命名（ZhiXing / PandaGripper 通用）。

    ZhiXing: left_fingerpad_col / right_fingerpad_col（mesh 垫）
    Panda:   finger1_pad_collision / finger2_pad_collision（box 垫）
    兜底：取名字含 "pad" 且参与碰撞的 geom 中、形心相距最远的两个。
    """
    import mujoco

    cands = []
    for g in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").lower()
        if "pad" in n and int(m.geom_contype[g]) != 0:
            cands.append(g)
    if len(cands) < 2:
        return None
    if len(cands) == 2:
        return cands[0], cands[1]
    # 多于两个：取形心相距最远的一对
    c = {g: np.array(m.geom_pos[g], dtype=float) for g in cands}
    best, pair = -1.0, None
    for i, g1 in enumerate(cands):
        for g2 in cands[i + 1:]:
            dd = float(np.linalg.norm(c[g1] - c[g2]))
            if dd > best:
                best, pair = dd, (g1, g2)
    return pair


def pad_face_pts(env):
    """两指垫碰撞 geom 的世界顶点（各自朝对方最近的一薄层质心 + 净距）。"""
    import mujoco

    sim, m, d, _ = N.ids(env)
    pair = find_pad_geoms(m)
    if pair is None:
        return None
    gl, gr = pair

    def verts(g):
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            v = np.array(m.mesh_vert[a:a + n], dtype=float)
        else:
            s = np.array(m.geom_size[g], dtype=float)
            v = s * np.array([[sx, sy, sz] for sx in (-1, 1)
                              for sy in (-1, 1) for sz in (-1, 1)])
        R = np.array(d.geom_xmat[g], dtype=float).reshape(3, 3)
        return (R @ v.T).T + np.array(d.geom_xpos[g], dtype=float)

    # 不假设左右命名顺序：u 由两垫形心连线定义，指向 gr，故 slab 选取与命名无关
    Pl, Pr = verts(gl), verts(gr)
    u = Pr.mean(axis=0) - Pl.mean(axis=0)
    u /= np.linalg.norm(u)
    fl = Pl[(Pl @ u) >= (Pl @ u).max() - SLAB_M].mean(axis=0)
    fr = Pr[(Pr @ (-u)) >= (Pr @ (-u)).max() - SLAB_M].mean(axis=0)
    return fl, fr, float(np.linalg.norm(fl - fr))


def measure(env, ctrl, label):
    """驱动夹爪到指定位形，测 grip_site 与夹持点的关系。"""
    sim, m, d, gid = N.ids(env)
    N.hold(env, ctrl, SETTLE)
    sim, m, d, gid = N.ids(env)
    p, R = N.gripsite(sim, m, d, gid)
    r = pad_face_pts(env)
    if r is None:
        return {"label": label, "error": "找不到 fingerpad geom"}
    fl, fr, sep = r
    pinch = 0.5 * (fl + fr)
    approach = R[:, 2]                       # grip_site 局部 +z = 接近轴
    off = float((pinch - p) @ approach)
    panda_off = -3.60e-3                     # N2：PandaGripper 夹持点 −3.60 mm
    return {
        "label": label,
        "grip_site_world": [float(v) for v in p],
        "pinch_world": [float(v) for v in pinch],
        "pinch_offset_along_approach_mm": off * 1000.0,
        "jaw_separation_mm": sep * 1000.0,
        "perpendicular_offset_mm": float(np.linalg.norm(
            (pinch - p) - off * approach) * 1000.0),
        "delta_vs_panda_pinch_mm": (off - panda_off) * 1000.0,
    }


def joint_margins(env):
    import mujoco

    sim, m, d, gid, aj, aadr, adof = N._arm_idx(env)
    jids = [sim.model.joint_name2id(
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)) for j in aj]
    q = np.array([d.qpos[a] for a in aadr], dtype=float)
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])
    margin = np.minimum(q - lo, hi - q)
    return {"qpos": [round(float(v), 6) for v in q],
            "margins_rad": [round(float(v), 4) for v in margin],
            "min_margin_rad": round(float(margin.min()), 4)}


def eef_pos_in_model(env):
    import mujoco

    sim, m, d, _ = N.ids(env)
    for b in range(m.nbody):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").endswith("eef"):
            return [round(float(v), 6) for v in m.body_pos[b]]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="libero_object")
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    res = {"node": "N4-pre", "check": "tcp_fix_verify", "suite": a.suite,
           "task_id": a.task_id, "settle_steps": SETTLE}

    # ---------- Panda 参考 ----------
    print("=" * 78)
    print("[ref] 建 Panda + PandaGripper 参考环境，实测夹持点")
    env_p, _ = build(a.suite, a.task_id, "Panda", "PandaGripper")
    env_p.reset()
    p_ref = measure(env_p, CTRL_OPEN, "panda_open")
    res["panda_reference"] = p_ref
    print(f"  Panda grip_site = {np.round(p_ref['grip_site_world'], 5).tolist()}")
    print(f"  Panda 夹持点偏移 = {p_ref['pinch_offset_along_approach_mm']:+.2f} mm"
          f"  (N2 记载 −3.60 mm)")
    print(f"  Panda 全开净距 = {p_ref['jaw_separation_mm']:.2f} mm")
    env_p.close()

    # ---------- ZhiXing ----------
    print("\n" + "=" * 78)
    print("[tgt] 建 ECO65 + ZhixingGripper 环境")
    env, task = build(a.suite, a.task_id, "ECO65", "ZhixingGripper")
    env.reset()
    res["eef_body_pos_in_model"] = eef_pos_in_model(env)
    print(f"  task: {task.language}")
    print(f"  编译模型 eef body pos = {res['eef_body_pos_in_model']}  (期望 0 0 0.20254)")

    m_open = measure(env, CTRL_OPEN, "zhixing_open")
    m_close = measure(env, CTRL_CLOSED, "zhixing_closed")
    res["zhixing_open"] = m_open
    res["zhixing_closed"] = m_close

    print(f"\n  全开：夹持点偏移 = {m_open['pinch_offset_along_approach_mm']:+.3f} mm"
          f"   净距 = {m_open['jaw_separation_mm']:.2f} mm")
    print(f"  闭合：夹持点偏移 = {m_close['pinch_offset_along_approach_mm']:+.3f} mm"
          f"   净距 = {m_close['jaw_separation_mm']:.2f} mm")
    print(f"  闭合位垂直偏离（应≈0）= {m_close['perpendicular_offset_mm']:.3f} mm")

    # ---------- 判据 ----------
    off_c = m_close["pinch_offset_along_approach_mm"]
    off_o = m_open["pinch_offset_along_approach_mm"]
    travel = off_c - off_o
    checks = {
        "eef_pos_is_0_0_0_20254": res["eef_body_pos_in_model"] == [0.0, 0.0, 0.20254],
        "pinch_coincides_with_gripsite_closed_mm": abs(off_c) < 2.0,
        "open_offset_is_negative_mm": off_o < -10.0,
        "pinch_travel_matches_17_44mm": abs(travel - 17.44) < 2.0,
        "jaw_open_matches_n3_89_89mm": abs(m_open["jaw_separation_mm"] - 89.89) < 0.5,
        "jaw_closed_matches_n3_0_65mm": abs(m_close["jaw_separation_mm"] - 0.65) < 0.5,
    }

    # ---------- 初始位姿 ----------
    env.reset()
    sim, m, d, gid = N.ids(env)
    jm = joint_margins(env)
    res["init_joint_state"] = jm
    print(f"\n  reset 关节余量最小 = {jm['min_margin_rad']} rad (N2 候选报 1.5424)")

    # reset 后（近闭）夹持点 vs Panda 夹持点
    fl, fr, sep = pad_face_pts(env)
    pinch = 0.5 * (fl + fr)
    p_panda = np.array(p_ref["pinch_world"])
    R_panda = np.array(p_ref["grip_site_world"])  # 仅用于打印
    d_reset = float(np.linalg.norm(pinch - p_panda) * 1000)
    res["reset_pinch_world"] = [float(v) for v in pinch]
    res["reset_pinch_vs_panda_pinch_mm"] = d_reset
    checks["init_pinch_within_1mm_of_panda"] = d_reset < 1.0
    print(f"  reset 夹持点 = {np.round(pinch, 5).tolist()}")
    print(f"  Panda 夹持点 = {np.round(p_panda, 5).tolist()}")
    print(f"  → 偏差 = {d_reset:.4f} mm  (N2 候选报 0.0007 mm)")

    res["checks"] = {k: bool(v) for k, v in checks.items()}
    res["new_calibration_constants"] = {
        "GRASP_OFFSET_CLOSED_m": [0.0, 0.0, round(off_c / 1000.0, 6)],
        "GRASP_OFFSET_OPEN_m": [0.0, 0.0, round(off_o / 1000.0, 6)],
        "note": "相对 grip_site 的局部系偏移（沿接近轴 +z）；直接替换 n3_grasp_test 的旧常数",
    }
    res["verdict"] = "PASS" if all(checks.values()) else "FAIL"

    print("\n" + "=" * 78)
    for k, v in checks.items():
        print(f"  [{'OK ' if v else 'XX '}] {k}")
    print(f"\n  NODE_VERDICT = {res['verdict']}")
    print(f"  新标定常数：闭合 {off_c/1000:+.5f} m   全开 {off_o/1000:+.5f} m")

    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    p = pathlib.Path(a.out) / "n4_verify_tcp_fix.json"
    p.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {p}")
    env.close()
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
