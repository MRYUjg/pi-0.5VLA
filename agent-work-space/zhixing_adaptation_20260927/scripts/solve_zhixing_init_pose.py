#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2：为 ZhiXing **重解 ECO65 初始位姿**（针对该夹爪独立求解，不照搬 Panda 工具安装变换）。

依据 N2 实测：
  * ZhiXing 的实际夹持区（两指垫最近点对中点）位于声明的 `grip_site`
    **沿接近轴前方 +66.0 mm（reset/近闭）～ +84.3 mm（全开）**；
  * PandaGripper 的同口径值只有 **+9.3 mm**（指尖顶点），pad 中心约 -3.6 mm；
  * 因此「把 grip_site 对齐到 Panda 参考位姿」并不等于「把**夹持点**对齐到参考位姿」。

本脚本：
  1. 取 Panda+PandaGripper reset 的 grip_site 6D 位姿作为唯一场景基准；
  2. 实测 ZhiXing 的夹持点相对 grip_site 的局部偏移；
  3. 报告当前落地 init_qpos 下，ZhiXing 夹持点与基准的偏差（应有 ~89 mm）；
  4. 用 DLS IK 重解 6 个臂关节，使 **ZhiXing 的夹持点**（而非声明 grip_site）
     落在基准位姿上；
  5. 复核残差与关节余量，输出候选 init_qpos（**不落地**，留待 N3 验证）。

产物：results/zhixing_init_pose.json
"""
import json
import pathlib
import sys

import numpy as np

RES = 256
TASK_SUITE = "libero_spatial"
DEFAULT_OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"
REF_GRIPPER = ("Panda", "PandaGripper")


def _build(robot, gripper, task_id=0, seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kw = {"bddl_file_name": bddl, "camera_heights": RES, "camera_widths": RES, "robots": [robot]}
    if gripper and gripper != "default":
        kw["gripper_types"] = gripper
    env = OffScreenRenderEnv(**kw)
    env.seed(seed)
    return env


def rot_err_vec(R_target, R_cur):
    """姿态误差向量（世界系，等价于轴角投影）。"""
    R = R_target @ R_cur.T
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    s = np.linalg.norm(v)
    if s < 1e-12:
        return v
    ang = np.arctan2(s, (np.trace(R) - 1.0) / 2.0)
    return v / s * ang


def site_id(sim, suffix):
    import mujoco

    m = sim.model._model
    for i in range(m.nsite):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or ""
        if n.endswith(suffix):
            return i
    raise RuntimeError(f"no site *{suffix}")


def arm_joint_names(sim):
    import mujoco

    m = sim.model._model
    names = []
    for i in range(m.njnt):
        n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if n.startswith("robot0_joint"):
            names.append(n)
    return sorted(names, key=lambda s: int(s.rsplit("_", 1)[-1]))


def pad_verts(sim, gid):
    import mujoco

    m, d = sim.model._model, sim.data._data
    pos = np.array(d.geom_xpos[gid], dtype=float)
    mat = np.array(d.geom_xmat[gid], dtype=float).reshape(3, 3)
    if m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gid]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = np.array(m.mesh_vert[a : a + n], dtype=float)
    else:
        s = np.array(m.geom_size[gid], dtype=float)
        v = s * np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return (mat @ v.T).T + pos


def find_geom(sim, *subs):
    import mujoco

    m = sim.model._model
    for i in range(m.ngeom):
        n = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").lower()
        for s in subs:
            if s.lower() in n:
                return i
    return None


def band_face_center(V, u, band_m=0.005):
    """沿 u 方向取最靠前的一薄层顶点的质心 = 内侧面中心。"""
    proj = V @ u
    return V[proj >= proj.max() - band_m].mean(axis=0)


def grasp_offset_local(sim, gripper):
    """
    夹持点（两指垫**内侧面中点**）在 grip_site 局部系下的偏移，单位 m。

    不用「两网格最近点对」：ZhiXing 是二连平行连杆，两面接近平行时最近点对
    是退化的，会跳到角点（实测同一位形给出 +66 mm，而内面口径是 +82.5 mm）。
    """
    gid = site_id(sim, "grip_site")
    gpos = np.array(sim.data._data.site_xpos[gid], dtype=float)
    gmat = np.array(sim.data._data.site_xmat[gid], dtype=float).reshape(3, 3)
    if gripper == "ZhixingGripper":
        gl, gr = find_geom(sim, "left_fingerpad_col"), find_geom(sim, "right_fingerpad_col")
    else:
        gl, gr = find_geom(sim, "finger1_pad_collision"), find_geom(sim, "finger2_pad_collision")
    Vl, Vr = pad_verts(sim, gl), pad_verts(sim, gr)
    cl, cr = Vl.mean(axis=0), Vr.mean(axis=0)
    u = (cr - cl) / np.linalg.norm(cr - cl)
    fl, fr = band_face_center(Vl, u), band_face_center(Vr, -u)
    mid = (fl + fr) / 2.0
    return gmat.T @ (mid - gpos), float(np.linalg.norm(fl - fr))


def settle_gripper_closed(env, gripper, steps=800):
    """驱动夹爪到闭合位并步进，使 equality/connect 约束真正收敛（同时锁住臂关节）。"""
    import mujoco

    sim = env.env.sim
    m, d = sim.model._model, sim.data._data
    acts = [i for i in range(m.nu)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").startswith("gripper0_")]
    if gripper == "ZhixingGripper":
        targets = {acts[0]: -0.91}
    else:
        targets = {i: 0.0 for i in acts}
    aj = [i for i in range(m.njnt)
          if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or "").startswith("robot0_joint")]
    aadr = [int(m.jnt_qposadr[j]) for j in aj]
    adof = [int(m.jnt_dofadr[j]) for j in aj]
    q0 = np.array([d.qpos[a] for a in aadr], dtype=float)
    d.qvel[:] = 0.0
    for _ in range(steps):
        for a, vv in zip(aadr, q0):
            d.qpos[a] = vv
        for a in adof:
            d.qvel[a] = 0.0
        for i, vv in targets.items():
            d.ctrl[i] = vv
        sim.step()
    mujoco.mj_forward(m, d)


def solve_ik(env, target_pos, target_mat, q_seed, iters=1500, lam=0.05, alpha=0.6, tol=1e-6):
    import mujoco

    sim = env.env.sim
    m, d = sim.model._model, sim.data._data
    gid = site_id(sim, "grip_site")
    jnames = arm_joint_names(sim)
    jids = [sim.model.joint_name2id(n) for n in jnames]
    adr = [int(m.jnt_qposadr[j]) for j in jids]
    dof = [int(m.jnt_dofadr[j]) for j in jids]
    lo = np.array([m.jnt_range[j][0] for j in jids])
    hi = np.array([m.jnt_range[j][1] for j in jids])

    q = np.clip(np.array(q_seed, dtype=float), lo + 1e-9, hi - 1e-9)
    hist = []
    it = 0
    for it in range(iters):
        for a, v in zip(adr, q):
            d.qpos[a] = v
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        cur_p = np.array(d.site_xpos[gid], dtype=float)
        cur_R = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)
        e = np.concatenate([target_pos - cur_p, rot_err_vec(target_mat, cur_R)])
        hist.append(float(np.linalg.norm(e)))
        if hist[-1] < tol:
            break
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, gid)
        J = np.vstack([jacp[:, dof], jacr[:, dof]])
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), e)
        q = np.clip(q + alpha * dq, lo, hi)

    for a, v in zip(adr, q):
        d.qpos[a] = v
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    util = ((q - lo) / (hi - lo)).tolist()
    return {
        "q": q.tolist(),
        "joint_names": jnames,
        "iters": it + 1,
        "limit_util": util,
        "limit_margin_min_rad": float(min(np.minimum(q - lo, hi - q))),
        "err_hist_tail": hist[-5:],
    }


def measure_at(env, gripper, q_arm, off):
    """把臂关节设成 q_arm，返回 grip_site 位姿 + 夹持点世界位姿（off 为夹持点在 grip_site 系的固定偏移）。"""
    import mujoco

    sim = env.env.sim
    m, d = sim.model._model, sim.data._data
    for n, v in zip(arm_joint_names(sim), q_arm):
        d.qpos[int(m.jnt_qposadr[sim.model.joint_name2id(n)])] = float(v)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    gid = site_id(sim, "grip_site")
    gpos = np.array(d.site_xpos[gid], dtype=float)
    gmat = np.array(d.site_xmat[gid], dtype=float).reshape(3, 3)
    return {"grip_pos": gpos, "grip_mat": gmat, "grasp_pos": gpos + gmat @ off,
            "grasp_offset_local": off}


def main():
    import mujoco

    out = {}

    # ---- 1. 基准：Panda + PandaGripper reset 的 grip_site 位姿 ----
    env_r = _build(*REF_GRIPPER)
    env_r.reset()
    sim_r = env_r.env.sim
    rid = site_id(sim_r, "grip_site")
    ref_pos = np.array(sim_r.data._data.site_xpos[rid], dtype=float)
    ref_mat = np.array(sim_r.data._data.site_xmat[rid], dtype=float).reshape(3, 3)
    ref_grasp_off, _ = grasp_offset_local(sim_r, "PandaGripper")
    env_r.close()
    print(f"[ref] Panda grip_site pos={np.round(ref_pos,5).tolist()}")
    print(f"[ref] PandaGripper 夹持点局部偏移={np.round(ref_grasp_off*1000,2).tolist()} mm")

    # ---- 2. ZhiXing：当前 init_qpos 下的夹持点 ----
    env = _build("ECO65", "ZhixingGripper")
    env.reset()
    # 把夹爪驱到闭合位（策略抓取时的构型），并让四连杆约束真正收敛
    settle_gripper_closed(env, "ZhixingGripper")
    sim = env.env.sim
    m = sim.model._model
    q_cur = np.array(
        [sim.data._data.qpos[int(m.jnt_qposadr[sim.model.joint_name2id(n)])]
         for n in arm_joint_names(sim)],
        dtype=float,
    )
    off, clear = grasp_offset_local(sim, "ZhixingGripper")
    cur = measure_at(env, "ZhixingGripper", q_cur, off)

    # 语义基准：Panda 的 grip_site 落在参考位姿时，它的**夹持点**在哪里。
    # 策略学到的语义是「grip_site 到点 → 夹持点即到点」，故要对齐的是夹持点。
    pinch_target = ref_pos + ref_mat @ ref_grasp_off
    d_cur = float(np.linalg.norm(cur["grasp_pos"] - pinch_target))

    print(f"\n[cur] 落地 init_qpos = {np.round(q_cur,6).tolist()}")
    print(f"[cur] ZhiXing 声明 grip_site 与基准距离 = "
          f"{np.linalg.norm(cur['grip_pos']-ref_pos)*1000:.2f} mm")
    print(f"[cur] ZhiXing **夹持点** 与基准夹持点距离 = {d_cur*1000:.2f} mm")
    print(f"[cur] ZhiXing 夹持点局部偏移(闭合位) = "
          f"{np.round(cur['grasp_offset_local']*1000,2).tolist()} mm  面间距={clear*1000:.2f} mm")
    print(f"[ref] Panda 夹持点相对其 grip_site 偏移 = "
          f"{np.round(ref_grasp_off*1000,2).tolist()} mm")

    out["reference"] = {
        "robot": REF_GRIPPER[0], "gripper": REF_GRIPPER[1],
        "grip_site_pos": ref_pos.tolist(), "grip_site_mat": ref_mat.tolist(),
        "panda_grasp_offset_local_m": ref_grasp_off.tolist(),
    }
    out["current_landed"] = {
        "init_qpos": q_cur.tolist(),
        "grip_site_vs_ref_mm": float(np.linalg.norm(cur["grip_pos"] - ref_pos) * 1000),
        "grasp_point_vs_ref_mm": d_cur * 1000,
        "grasp_offset_local_m": off.tolist(),
        "pad_face_clearance_m": clear,
        "semantic_target_note": "基准 = Panda 在其 grip_site 参考位姿下的**夹持点**位置",
        "zhixing_grasp_offset_local_m": cur["grasp_offset_local"].tolist(),
    }

    # ---- 3. IK：让 ZhiXing 的**夹持点**落在基准 grip_site 位姿上 ----
    tgt_pos = pinch_target - ref_mat @ off     # 夹持点 = pinch_target  ⇒  反推 grip_site
    print(f"\n[ik] 目标 grip_site 位置 = {np.round(tgt_pos,5).tolist()} "
          f"(= 基准 − R·offsets，即把夹持点顶到基准上)")

    sol = solve_ik(env, tgt_pos, ref_mat, q_cur)
    chk = measure_at(env, "ZhixingGripper", sol["q"], off)
    print(f"[ik] 收敛 {sol['iters']} 次迭代，末端误差历史尾部={np.round(sol['err_hist_tail'],8).tolist()}")
    print(f"[ik] 新 init_qpos = {np.round(sol['q'],6).tolist()}")
    print(f"[ik] 复核：夹持点与基准夹持点距离 = "
          f"{np.linalg.norm(chk['grasp_pos']-pinch_target)*1000:.4f} mm")
    print(f"[ik] 复核：姿态误差 = "
          f"{np.degrees(np.linalg.norm(rot_err_vec(ref_mat, chk['grip_mat']))):.5f} deg")
    print(f"[ik] 关节余量最小 = {sol['limit_margin_min_rad']:.4f} rad")

    pos_res = float(np.linalg.norm(chk["grasp_pos"] - pinch_target) * 1000)
    rot_res = float(np.degrees(np.linalg.norm(rot_err_vec(ref_mat, chk["grip_mat"]))))
    out["candidate"] = {
        "init_qpos": sol["q"],
        "joint_names": sol["joint_names"],
        "grasp_point_residual_mm": pos_res,
        "grasp_point_rot_residual_deg": rot_res,
        "limit_util": sol["limit_util"],
        "limit_margin_min_rad": sol["limit_margin_min_rad"],
        "iters": sol["iters"],
        "gate_pass": bool(pos_res <= 0.5 and rot_res <= 0.2),
        "landed": False,
    }
    print(f"\n[gate] N2(b) 判据「IK 残差 ≤ 0.5 mm / 0.2°」→ "
          f"{'PASS' if out['candidate']['gate_pass'] else 'FAIL'} "
          f"({pos_res:.4f} mm / {rot_res:.5f} deg)")

    out_dir = pathlib.Path(DEFAULT_OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "zhixing_init_pose.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[out] {out_dir / 'zhixing_init_pose.json'}")
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
