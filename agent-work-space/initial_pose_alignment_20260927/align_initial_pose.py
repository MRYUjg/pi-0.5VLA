#!/usr/bin/env python3
"""ECO65 初始条件对齐：固定物体位姿 + IK 对齐完整末端位姿 + 测量相机↔夹持点变换。

设计要点
--------
1. **不照搬 Panda 的 right_hand 参数**。`right_hand` 是最后一级连杆的坐标系，改它会同时改变
   夹爪安装几何与运动学链。本脚本只改两处：
     * `init_qpos`（机器人初始关节角）—— 用 IK 反解，使 grip_site 的**位置与朝向**与 Panda 一致；
     * `eye_in_hand` 相机的 local pos/quat —— 仅当实测「相机↔夹持点」变换确有差异时才修正。
   夹爪安装（right_hand 的 pos/quat）、夹爪资产、mount 一律不动。
2. **固定物体位姿**：以 Panda 官方 init state 为准，把**非机器人关节**按名字抄到任意场景，
   保证两台机器人看到的是同一个物体摆放。
3. 子命令：measure / solve / apply / verify / report

用法（在 libero 容器内）：
    python align_initial_pose.py measure --robot ECO65 --gripper PandaGripper
    python align_initial_pose.py solve   --task-id 0
    python align_initial_pose.py apply   --solution results/ik_solution.json
    python align_initial_pose.py verify
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import sys

import numpy as np

REPO = pathlib.Path("/app")
OUT = pathlib.Path("/app/agent-work-space/initial_pose_alignment_20260927/results")
RES = 256
TASK_SUITE = "libero_spatial"

MOUNTED_ECO65 = REPO / "third_party/libero/libero/libero/envs/robots/mounted_eco65.py"
ECO65_ROBOT = REPO / "robosuite/models/robots/manipulators/eco65_robot.py"
ECO65_XML = REPO / "robosuite/models/assets/robots/ECO65-B/robot.xml"
PANDA_XML = REPO / "robosuite/models/assets/robots/panda/robot.xml"


# --------------------------------------------------------------------------- #
# 基础数学
# --------------------------------------------------------------------------- #
def T_of(pos, mat):
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = pos
    return T


def rot_err_vec(R_target, R_cur):
    """旋转误差向量（轴角），R_target 相对 R_cur。"""
    R = R_target @ R_cur.T
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    ang = np.arccos(cos)
    if ang < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(ang))
    return axis * ang


def mat2quat_wxyz(R):
    """旋转矩阵 → MuJoCo 约定四元数 (w, x, y, z)。"""
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(9))
    return q


def quat_wxyz2mat(q):
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.ascontiguousarray(q, dtype=float))
    return R.reshape(3, 3)


def parse_nums(text):
    return [float(x) for x in text.replace(",", " ").split()]


# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #
def build_env(robot, gripper, task_id=0, seed=7):
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
    return env, suite, task


def joint_names(sim):
    import mujoco

    raw = sim.model._model
    return [mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_JOINT, i) or f"jnt{i}"
            for i in range(raw.njnt)]


def is_robot_joint(name):
    return name.startswith("robot0_") or name.startswith("gripper0_")


def reference_object_poses(task_id=0, init_idx=0, seed=7):
    """用 Panda 官方 init state 得到物体位姿（按关节名），作为跨机器人的唯一场景基准。"""
    env, suite, _ = build_env("Panda", "PandaGripper", task_id, seed)
    env.reset()
    env.set_init_state(suite.get_task_init_states(task_id)[init_idx])
    sim = env.env.sim
    poses = {}
    for n in joint_names(sim):
        if is_robot_joint(n):
            continue
        v = np.array(sim.data.get_joint_qpos(n), dtype=float)
        poses[n] = v.tolist()
    env.close()
    return poses


def freeze_objects(env, poses):
    sim = env.env.sim
    applied = 0
    for n, v in poses.items():
        try:
            sim.data.set_joint_qpos(n, np.array(v, dtype=float))
            applied += 1
        except Exception:
            pass
    sim.forward()
    return applied


# --------------------------------------------------------------------------- #
# 测量
# --------------------------------------------------------------------------- #
def grip_site_name(env):
    import mujoco

    raw = env.env.sim.model._model
    for i in range(raw.nsite):
        n = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_SITE, i) or ""
        if n.endswith("grip_site"):
            return n
    raise RuntimeError("找不到 grip_site")


def camera_name(env):
    import mujoco

    raw = env.env.sim.model._model
    for i in range(raw.ncam):
        n = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_CAMERA, i) or ""
        if n.endswith("eye_in_hand"):
            return n
    raise RuntimeError("找不到 eye_in_hand 相机")


def render(sim, cam, res=RES):
    img = sim.render(camera_name=cam, width=res, height=res)
    if isinstance(img, tuple):
        img = img[0]
    return np.asarray(img)[::-1]  # robosuite 的 obs 约定


def measure(env, tag, poses, save_png=True):
    from robosuite.utils import transform_utils as TU

    sim = env.env.sim
    freeze_objects(env, poses)

    gname = grip_site_name(env)
    cname = camera_name(env)
    gid = sim.model.site_name2id(gname)
    cid = sim.model.camera_name2id(cname)

    grip_pos = np.array(sim.data.site_xpos[gid], dtype=float)
    grip_mat = np.array(sim.data.site_xmat[gid], dtype=float).reshape(3, 3)
    cam_pos = np.array(sim.data.cam_xpos[cid], dtype=float)
    cam_mat = np.array(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)

    T_grip = T_of(grip_pos, grip_mat)
    T_cam = T_of(cam_pos, cam_mat)

    # 相机 ↔ 夹持点 双向变换
    T_cam2grip = np.linalg.inv(T_cam) @ T_grip    # 夹持点在相机系下
    T_grip2cam = np.linalg.inv(T_grip) @ T_cam    # 相机在夹持点系下

    # right_hand（安装几何）与两者各自在 hand 系下的变换
    rh = [x for x in ("robot0_right_hand", "right_hand") if _has_body(sim, x)]
    hand_info = {}
    if rh:
        hid = sim.model.body_name2id(rh[0])
        hp = np.array(sim.data.body_xpos[hid], dtype=float)
        hm = np.array(sim.data.body_xmat[hid], dtype=float).reshape(3, 3)
        T_hand = T_of(hp, hm)
        hand_info = {
            "name": rh[0],
            "pos": hp.tolist(),
            "T_hand2grip": (np.linalg.inv(T_hand) @ T_grip).tolist(),
            "T_hand2cam": (np.linalg.inv(T_hand) @ T_cam).tolist(),
        }

    img = render(sim, cname)
    if save_png:
        OUT.mkdir(parents=True, exist_ok=True)
        _save_png(OUT / f"wrist_{tag}.png", img)

    # 腕部画面里"物体尺度"的代理量：相机到各物体自由关节的距离
    obj_dist = {}
    for n in poses:
        try:
            bid = sim.model.body_name2id(n.rsplit("_joint", 1)[0])
            obj_dist[n] = float(np.linalg.norm(np.array(sim.data.body_xpos[bid]) - cam_pos))
        except Exception:
            continue

    return {
        "tag": tag,
        "grip_site": gname,
        "camera": cname,
        "eef_pos": grip_pos.tolist(),
        "eef_mat": grip_mat.tolist(),
        "eef_quat_wxyz": mat2quat_wxyz(grip_mat).tolist(),
        "cam_pos": cam_pos.tolist(),
        "cam_mat": cam_mat.tolist(),
        "T_cam2grip": T_cam2grip.tolist(),
        "T_grip2cam": T_grip2cam.tolist(),
        "hand": hand_info,
        "wrist_img_mean": float(img.mean()),
        "cam_to_objects_m": obj_dist,
    }


def _has_body(sim, name):
    try:
        sim.model.body_name2id(name)
        return True
    except Exception:
        return False


def _save_png(path, img):
    try:
        import cv2

        cv2.imwrite(str(path), np.asarray(img)[..., ::-1])
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# IK：阻尼最小二乘，目标为 grip_site 的完整 6D 位姿
# --------------------------------------------------------------------------- #
def arm_joint_names(sim, robot="ECO65"):
    import mujoco

    raw = sim.model._model
    names = []
    for i in range(raw.njnt):
        n = mujoco.mj_id2name(raw, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if n.startswith("robot0_joint"):
            names.append(n)
    return sorted(names, key=lambda s: int(s.rsplit("_", 1)[-1]))


def solve_ik(env, target_pos, target_mat, q_seed, iters=800, lam=0.05, alpha=0.6, tol=1e-5):
    import mujoco

    sim = env.env.sim
    raw_m, raw_d = sim.model._model, sim.data._data
    gid = sim.model.site_name2id(grip_site_name(env))
    jnames = arm_joint_names(sim)
    jids = [sim.model.joint_name2id(n) for n in jnames]
    adr = [int(raw_m.jnt_qposadr[j]) for j in jids]
    dof = [int(raw_m.jnt_dofadr[j]) for j in jids]
    lo = np.array([raw_m.jnt_range[j][0] for j in jids])
    hi = np.array([raw_m.jnt_range[j][1] for j in jids])

    q = np.clip(np.array(q_seed, dtype=float), lo + 1e-6, hi - 1e-6)
    hist = []
    for it in range(iters):
        for a, v in zip(adr, q):
            raw_d.qpos[a] = v
        mujoco.mj_kinematics(raw_m, raw_d)
        mujoco.mj_comPos(raw_m, raw_d)

        cur_p = np.array(raw_d.site_xpos[gid], dtype=float)
        cur_R = np.array(raw_d.site_xmat[gid], dtype=float).reshape(3, 3)
        e = np.concatenate([target_pos - cur_p, rot_err_vec(target_mat, cur_R)])
        hist.append(float(np.linalg.norm(e)))
        if hist[-1] < tol:
            break

        jacp = np.zeros((3, raw_m.nv))
        jacr = np.zeros((3, raw_m.nv))
        mujoco.mj_jacSite(raw_m, raw_d, jacp, jacr, gid)
        J = np.vstack([jacp[:, dof], jacr[:, dof]])
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(6), e)
        q = np.clip(q + alpha * dq, lo, hi)

    for a, v in zip(adr, q):
        raw_d.qpos[a] = v
    mujoco.mj_kinematics(raw_m, raw_d)
    mujoco.mj_comPos(raw_m, raw_d)
    cur_p = np.array(raw_d.site_xpos[gid], dtype=float)
    cur_R = np.array(raw_d.site_xmat[gid], dtype=float).reshape(3, 3)
    pos_err = float(np.linalg.norm(target_pos - cur_p))
    rot_err = float(np.degrees(np.linalg.norm(rot_err_vec(target_mat, cur_R))))
    util = ((q - lo) / (hi - lo)).tolist()
    return {
        "q": q.tolist(),
        "joint_names": jnames,
        "pos_err_mm": pos_err * 1000.0,
        "rot_err_deg": rot_err,
        "iters": it + 1,
        "limit_util": util,
        "limit_margin_min": float(min(np.minimum(q - lo, hi - q))),
        "err_hist_tail": hist[-5:],
    }


# --------------------------------------------------------------------------- #
# 子命令
# --------------------------------------------------------------------------- #
def cmd_measure(a):
    poses = reference_object_poses(a.task_id, a.init_idx, a.seed)
    out = {"task_id": a.task_id, "init_idx": a.init_idx, "seed": a.seed,
           "object_joints": list(poses.keys()), "robots": {}}
    for robot, gripper in (("Panda", "PandaGripper"), ("ECO65", a.gripper)):
        env, _, _ = build_env(robot, gripper, a.task_id, a.seed)
        env.reset()
        out["robots"][f"{robot}+{gripper}"] = measure(env, f"{robot}_{gripper}", poses)
        env.close()
    p = OUT / "initial_condition.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))
    print_report(out)
    print(f"\n[measure] 写入 {p}")


def cmd_solve(a):
    poses = reference_object_poses(a.task_id, a.init_idx, a.seed)

    # 目标 = Panda 的初始 grip_site 位姿
    penv, _, _ = build_env("Panda", "PandaGripper", a.task_id, a.seed)
    penv.reset()
    pm = measure(penv, "Panda_ref", poses, save_png=False)
    penv.close()
    target_pos = np.array(pm["eef_pos"])
    target_mat = np.array(pm["eef_mat"])
    print(f"[solve] 目标(Panda grip_site): pos={np.round(target_pos,4)} "
          f"quat(wxyz)={np.round(pm['eef_quat_wxyz'],4)}")

    env, _, _ = build_env("ECO65", a.gripper, a.task_id, a.seed)
    env.reset()
    sim = env.env.sim
    cur = np.array([sim.data.get_joint_qpos(n) for n in arm_joint_names(sim)], dtype=float)
    print(f"[solve] ECO65 当前 init_qpos = {np.round(cur,4)}")

    seeds = [cur]
    rng = np.random.default_rng(0)
    for s in range(a.multistart):
        seeds.append(np.clip(cur + rng.normal(0, 0.35, size=cur.shape), -3.0, 3.0))

    best = None
    for i, s in enumerate(seeds):
        r = solve_ik(env, target_pos, target_mat, s)
        r["seed_index"] = i
        r["seed"] = s.tolist()
        # 评分：优先误差小，其次关节离限位远（避免贴边姿态）
        r["score"] = r["pos_err_mm"] + 5.0 * r["rot_err_deg"] - 0.5 * r["limit_margin_min"]
        ok = r["pos_err_mm"] < a.pos_tol_mm and r["rot_err_deg"] < a.rot_tol_deg
        print(f"   seed#{i}: pos={r['pos_err_mm']:.3f}mm rot={r['rot_err_deg']:.4f}deg "
              f"margin={r['limit_margin_min']:.3f} {'✅' if ok else ''}")
        if ok and (best is None or r["score"] < best["score"]):
            best = r
    env.close()

    if best is None:
        print("[solve] ❌ 未找到满足容差的解，请放宽 multistart 或检查可达性")
        return 1

    sol = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "robot": "ECO65", "gripper": a.gripper, "task_id": a.task_id, "init_idx": a.init_idx,
        "target_source": "Panda+PandaGripper initial grip_site pose",
        "target_pos": target_pos.tolist(),
        "target_quat_wxyz": pm["eef_quat_wxyz"],
        "old_init_qpos": cur.tolist(),
        "new_init_qpos": best["q"],
        "ik": {k: v for k, v in best.items() if k not in ("q",)},
    }
    p = OUT / "ik_solution.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(sol, indent=2))
    print(f"[solve] ✅ pos={best['pos_err_mm']:.3f}mm rot={best['rot_err_deg']:.4f}deg")
    print(f"[solve] 新 init_qpos(未含夹爪关节) = {np.round(best['q'], 6).tolist()}")
    print(f"[solve] 写入 {p}")
    return 0


def _patch_init_qpos(path, new_q, dry=False):
    import re

    text = path.read_text(encoding="utf-8")
    m = re.search(r"def init_qpos\(self\):\s*\n(?:\s*#[^\n]*\n)*\s*return np\.array\(\[([^\]]*)\]\)", text)
    if not m:
        print(f"  !! 未在 {path.name} 找到 init_qpos 返回语句")
        return None
    old = np.array(parse_nums(m.group(1)))
    if old.shape != np.array(new_q).shape:
        print(f"  !! {path.name} 维度不符：文件 {old.shape} vs IK {np.array(new_q).shape}")
        return None
    arr = "np.array([" + ", ".join(f"{v:.6f}" for v in new_q) + "])"
    new_text = text[:m.start()] + f"def init_qpos(self):\n        return {arr}" + text[m.end():]
    if not dry:
        bak = path.with_suffix(path.suffix + ".pre_init_align.bak")
        if not bak.exists():
            shutil.copy2(path, bak)
        path.write_text(new_text, encoding="utf-8")
    return old.tolist()


def cmd_apply(a):
    sol = json.loads(pathlib.Path(a.solution).read_text())
    q = sol["new_init_qpos"]
    print(f"[apply] 对齐解来自 {a.solution}")
    print(f"[apply] 旧 init_qpos = {np.round(sol['old_init_qpos'],4).tolist()}")
    print(f"[apply] 新 init_qpos = {np.round(q,6).tolist()}")
    for p in (MOUNTED_ECO65, ECO65_ROBOT):
        old = _patch_init_qpos(p, q, dry=a.dry_run)
        print(f"  {'(dry-run) ' if a.dry_run else ''}{p.name}: {old}")

    if a.align_camera:
        print("[apply] --align-camera：检查相机↔夹持点变换")
        _maybe_fix_camera(dry=a.dry_run)
    else:
        print("[apply] 相机未改动（如 verify 显示相机↔夹持点有差异，再带 --align-camera 重跑）")
    return 0


def _maybe_fix_camera(dry=False):
    """仅当 ECO65 与 Panda 的 T_grip2cam 存在实质差异时，修正 eye_in_hand 的 local 参数。

    注意：改的是**相机**，不是 right_hand —— 安装几何与夹爪链路保持不动。
    """
    poses = reference_object_poses(0, 0, 7)
    res = {}
    for robot, gripper in (("Panda", "PandaGripper"), ("ECO65", "PandaGripper")):
        env, _, _ = build_env(robot, gripper, 0, 7)
        env.reset()
        res[robot] = measure(env, f"cam_{robot}", poses, save_png=False)
        env.close()
    Tp = np.array(res["Panda"]["T_grip2cam"])
    Te = np.array(res["ECO65"]["T_grip2cam"])
    dpos = np.linalg.norm(Tp[:3, 3] - Te[:3, 3])
    dang = np.degrees(np.linalg.norm(rot_err_vec(Tp[:3, :3], Te[:3, :3])))
    print(f"  T_grip2cam 差异: 平移 {dpos*1000:.3f} mm, 旋转 {dang:.4f} deg")
    if dpos < 1e-4 and dang < 0.05:
        print("  → 差异可忽略，相机无需修改（保持资产原样）")
        return
    # 需要修正：T_hand2cam_new = T_hand2grip @ T_grip2cam_panda
    T_hand2grip_e = np.array(res["ECO65"]["hand"]["T_hand2grip"])
    T_new = T_hand2grip_e @ Tp
    pos = T_new[:3, 3]
    quat = mat2quat_wxyz(T_new[:3, :3])
    print(f"  → 需要把 eye_in_hand 的 local 参数改为 pos={np.round(pos,6).tolist()} "
          f"quat={np.round(quat,6).tolist()}")
    if dry:
        return
    import re

    text = ECO65_XML.read_text(encoding="utf-8")
    m = re.search(r'<camera mode="fixed" name="eye_in_hand"[^/]*/>', text)
    if not m:
        print("  !! 未找到 eye_in_hand 相机标签")
        return
    new_tag = (f'<camera mode="fixed" name="eye_in_hand" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
               f'quat="{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}" fovy="75"/>')
    bak = ECO65_XML.with_suffix(".xml.pre_cam_align.bak")
    if not bak.exists():
        shutil.copy2(ECO65_XML, bak)
    ECO65_XML.write_text(text[:m.start()] + new_tag + text[m.end():], encoding="utf-8")
    print(f"  [patched] {ECO65_XML.name}  {new_tag}")


def cmd_verify(a):
    ref = json.loads((OUT / "initial_condition.json").read_text()) if (OUT / "initial_condition.json").exists() else None
    poses = reference_object_poses(a.task_id, a.init_idx, a.seed)
    out = {}
    for robot, gripper in (("Panda", "PandaGripper"), ("ECO65", a.gripper)):
        env, _, _ = build_env(robot, gripper, a.task_id, a.seed)
        env.reset()
        out[f"{robot}+{gripper}"] = measure(env, f"verify_{robot}", poses)
        env.close()

    P = out["Panda+PandaGripper"]
    E = out[f"ECO65+{a.gripper}"]
    dpos = np.linalg.norm(np.array(P["eef_pos"]) - np.array(E["eef_pos"]))
    drot = np.degrees(np.linalg.norm(rot_err_vec(np.array(P["eef_mat"]), np.array(E["eef_mat"]))))
    print("=" * 74)
    print(f"初始 grip_site 位姿差异: 位置 {dpos*1000:.3f} mm  姿态 {drot:.4f} deg")
    if ref:
        E0 = ref["robots"].get(f"ECO65+{a.gripper}")
        if E0:
            d0 = np.linalg.norm(np.array(P["eef_pos"]) - np.array(E0["eef_pos"]))
            r0 = np.degrees(np.linalg.norm(rot_err_vec(np.array(P["eef_mat"]), np.array(E0["eef_mat"]))))
            print(f"（对齐前记录值: 位置 {d0*1000:.3f} mm  姿态 {r0:.4f} deg）")
    cg_p = np.array(P["T_cam2grip"])
    cg_e = np.array(E["T_cam2grip"])
    print(f"相机↔夹持点(T_cam2grip) 差异: 平移 "
          f"{np.linalg.norm(cg_p[:3,3]-cg_e[:3,3])*1000:.3f} mm, 旋转 "
          f"{np.degrees(np.linalg.norm(rot_err_vec(cg_p[:3,:3], cg_e[:3,:3]))):.4f} deg")

    # 腕部画面差异（同一物体摆放）
    try:
        import cv2

        ip = cv2.imread(str(OUT / "wrist_verify_Panda.png"))
        ie = cv2.imread(str(OUT / "wrist_verify_ECO65.png"))
        if ip is not None and ie is not None and ip.shape == ie.shape:
            diff = np.abs(ip.astype(float) - ie.astype(float))
            print(f"腕部画面平均像素差: {diff.mean():.2f}/255 ({100*diff.mean()/255:.1f}%)")
            if ref:
                i0 = cv2.imread(str(OUT / f"wrist_ECO65_{a.gripper}.png"))
                if i0 is not None and i0.shape == ip.shape:
                    d0 = np.abs(i0.astype(float) - ip.astype(float)).mean()
                    print(f"（对齐前记录值: {d0:.2f}/255）")
    except Exception as exc:
        print("画面比较跳过:", exc)

    (OUT / "verify.json").write_text(json.dumps(out, indent=2))
    print("=" * 74)
    return 0


def print_report(out):
    print("=" * 74)
    for tag, m in out["robots"].items():
        print(f"### {tag}")
        print(f"  grip_site 位置 : {np.round(m['eef_pos'], 4).tolist()}")
        print(f"  grip_site 姿态 : quat(wxyz) {np.round(m['eef_quat_wxyz'], 4).tolist()}")
        print(f"  相机位置       : {np.round(m['cam_pos'], 4).tolist()}")
        print("  T_cam2grip (夹持点在相机系):")
        for row in np.round(np.array(m["T_cam2grip"]), 5):
            print("     ", row.tolist())
    if len(out["robots"]) == 2:
        p, e = list(out["robots"].values())
        dpos = np.linalg.norm(np.array(p["eef_pos"]) - np.array(e["eef_pos"]))
        drot = np.degrees(np.linalg.norm(rot_err_vec(np.array(p["eef_mat"]), np.array(e["eef_mat"]))))
        cg = np.linalg.norm(np.array(p["T_cam2grip"])[:3, 3] - np.array(e["T_cam2grip"])[:3, 3])
        print("-" * 74)
        print(f"grip_site 位姿差: {dpos*1000:.2f} mm / {drot:.2f} deg")
        print(f"T_cam2grip 平移差: {cg*1000:.3f} mm")
    print("=" * 74)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--gripper", default="PandaGripper")
        p.add_argument("--task-id", type=int, default=0)
        p.add_argument("--init-idx", type=int, default=0)
        p.add_argument("--seed", type=int, default=7)

    m = sub.add_parser("measure", help="测量两台机器人的初始条件（物体位姿固定）"); common(m)
    m.set_defaults(func=cmd_measure)

    s = sub.add_parser("solve", help="IK 反解 ECO65 关节角以匹配 Panda 的 grip_site 位姿"); common(s)
    s.add_argument("--multistart", type=int, default=40)
    s.add_argument("--pos-tol-mm", type=float, default=0.5)
    s.add_argument("--rot-tol-deg", type=float, default=0.2)
    s.set_defaults(func=cmd_solve)

    p = sub.add_parser("apply", help="把 IK 解写入 MountedECO65/ECO65 的 init_qpos"); common(p)
    p.add_argument("--solution", default=str(OUT / "ik_solution.json"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--align-camera", action="store_true", help="必要时修正 eye_in_hand 相机 local 参数")
    p.set_defaults(func=cmd_apply)

    v = sub.add_parser("verify", help="重新测量并对比对齐效果"); common(v)
    v.set_defaults(func=cmd_verify)

    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sys.exit(a.func(a) or 0)


if __name__ == "__main__":
    main()
