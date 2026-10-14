#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2：ZhiXing 的 TCP / 指尖几何 / 相机几何 / 视觉占比 重标定。

回答 N2 的三个核心问题：
  1. ZhiXing 声明的 `grip_site` 与「实际夹持区（两指垫中点）」是否重合？
     差多少？沿接近轴（approach axis）指尖伸出 grip_site 多远？
  2. 装上 ZhiXing 后 `T_cam2grip` 是否仍与 PandaGripper 一致？
     即：**PandaGripper 下的初始位姿对齐结论是否仍然成立**（预期：不成立）。
  3. ZhiXing 在腕部 / 外部视角画面中的占比与 PandaGripper 差多少？
     （这是「视觉 OOD」的直接量化）

三个配置同场景、同物体初态、同分辨率测量：
  ECO65+ZhiXing / ECO65+PandaGripper / Panda+PandaGripper

产物：results/tcp_camera_audit.json、results/renders/*.png
"""
import argparse
import json
import pathlib
import sys

import numpy as np

RES = 256
TASK_SUITE = "libero_spatial"
DEFAULT_OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results"

CONFIGS = [
    ("ECO65", "ZhixingGripper", "eco65_zhixing"),
    ("ECO65", "PandaGripper", "eco65_panda"),
    ("Panda", "PandaGripper", "panda_panda"),
]
JAW_STATES = ["reset", "open", "closed"]


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def T_of(pos, mat):
    T = np.eye(4)
    T[:3, :3] = mat
    T[:3, 3] = pos
    return T


def mat2quat_wxyz(R):
    """旋转矩阵 → 四元数 (w, x, y, z)。"""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    return q / np.linalg.norm(q)


def rot_angle_deg(Ra, Rb):
    R = Ra.T @ Rb
    c = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def build_env(robot, gripper, task_id=0, seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    kw = {
        "bddl_file_name": bddl,
        "camera_heights": RES,
        "camera_widths": RES,
        "robots": [robot],
    }
    if gripper and gripper != "default":
        kw["gripper_types"] = gripper
    env = OffScreenRenderEnv(**kw)
    env.seed(seed)
    return env, suite, task


def sim_names(sim, objtype):
    import mujoco

    raw = sim.model._model
    n = {
        mujoco.mjtObj.mjOBJ_JOINT: raw.njnt,
        mujoco.mjtObj.mjOBJ_BODY: raw.nbody,
        mujoco.mjtObj.mjOBJ_GEOM: raw.ngeom,
        mujoco.mjtObj.mjOBJ_SITE: raw.nsite,
        mujoco.mjtObj.mjOBJ_CAMERA: raw.ncam,
    }[objtype]
    return [mujoco.mj_id2name(raw, objtype, i) or f"id{i}" for i in range(n)]


def find_name(sim, objtype, suffix):
    for n in sim_names(sim, objtype):
        if n.endswith(suffix):
            return n
    raise RuntimeError(f"找不到以 {suffix} 结尾的 {objtype}")


# --------------------------------------------------------------------------- #
# 几何：几何体顶点 / 中心 / 包围
# --------------------------------------------------------------------------- #
def geom_vertices_world(sim, gid):
    """返回该 geom 在世界系下的代表点集合（mesh 用真实顶点，box 用 8 角点）。"""
    import mujoco

    m = sim.model._model
    d = sim.data._data
    pos = np.array(d.geom_xpos[gid], dtype=float)
    mat = np.array(d.geom_xmat[gid], dtype=float).reshape(3, 3)
    gt = m.geom_type[gid]

    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[gid]
        a, n = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
        v = np.array(m.mesh_vert[a : a + n], dtype=float)
    elif gt == mujoco.mjtGeom.mjGEOM_BOX:
        s = np.array(m.geom_size[gid], dtype=float)
        signs = np.array(
            [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
            dtype=float,
        )
        v = s * signs
    elif gt in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
        s = np.array(m.geom_size[gid], dtype=float)
        v = np.array([[0, 0, -s[1]], [0, 0, s[1]]], dtype=float)
    else:
        v = np.zeros((1, 3), dtype=float)

    return (mat @ v.T).T + pos


def geom_center_world(sim, gid):
    return geom_vertices_world(sim, gid).mean(axis=0)


def gripper_geom_ids(sim):
    """属于夹爪子树的 geom（沿 body 父链找 right_gripper）。"""
    import mujoco

    m = sim.model._model
    out = []
    for gid in range(m.ngeom):
        bid = int(m.geom_bodyid[gid])
        chain, guard = [], 0
        while bid != 0 and guard < 64:
            chain.append(bid)
            bid = int(m.body_parentid[bid])
            guard += 1
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in chain]
        if any("right_gripper" in n for n in names):
            out.append(gid)
    return out


def find_geom(sim, *suffixes):
    """按名字后缀找 geom id（大小写不敏感）。"""
    names = sim_names(sim, __import__("mujoco").mjtObj.mjOBJ_GEOM)
    low = [n.lower() for n in names]
    for suf in suffixes:
        s = suf.lower()
        for i, n in enumerate(low):
            if n.endswith(s) or s in n:
                return i
    return None


def pad_geom_ids(sim, gripper):
    """左右指垫的碰撞 geom id。"""
    if gripper == "ZhixingGripper":
        return find_geom(sim, "left_fingerpad_col"), find_geom(sim, "right_fingerpad_col")
    return find_geom(sim, "finger1_pad_collision"), find_geom(sim, "finger2_pad_collision")


def nip_point(sim, gl, gr):
    """
    两指垫之间的**最近点对** —— 物理意义上的「咬合点(nip point)」与真实净开口。

    对 mesh 指垫直接取两片网格表面最近的一对顶点，因此不依赖
    「质心 = 接触面」这类会失真的假设（支撑连杆网格质心并不在接触面上）。
    """
    if gl is None or gr is None:
        return None
    vl = geom_vertices_world(sim, gl)
    vr = geom_vertices_world(sim, gr)
    d2 = ((vl[:, None, :] - vr[None, :, :]) ** 2).sum(-1)
    i, j = np.unravel_index(int(np.argmin(d2)), d2.shape)
    pl, pr = vl[i], vr[j]
    return {
        "closest_left_world": pl.tolist(),
        "closest_right_world": pr.tolist(),
        "nip_center_world": ((pl + pr) / 2.0).tolist(),
        "nip_clearance_m": float(np.linalg.norm(pl - pr)),
    }


def gripper_geom_table(sim, ggids, gpos, gmat):
    """逐个夹爪 geom 在 grip_site 局部系下的位置与沿接近轴的伸出量。"""
    import mujoco

    m = sim.model._model
    rows = []
    for gid in ggids:
        v = geom_vertices_world(sim, gid)
        loc = (gmat.T @ (v - gpos).T).T
        rows.append(
            {
                "geom": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom{gid}",
                "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[gid])) or "",
                "extent_plus_z_m": float(loc[:, 2].max()),
                "extent_minus_z_m": float(loc[:, 2].min()),
                "center_local_m": loc.mean(axis=0).tolist(),
            }
        )
    rows.sort(key=lambda r: -r["extent_plus_z_m"])
    return rows


# --------------------------------------------------------------------------- #
# 夹爪状态控制
# --------------------------------------------------------------------------- #
def set_jaw_state(env, gripper, state):
    """直接写 gripper 关节 qpos 到指定状态（用 equality 关系精确求解）。"""
    sim = env.env.sim
    robot = env.env.robots[0]

    if gripper == "ZhixingGripper":
        # Left_1 = v；equality 要求 Left_2=v, Right_1=-v, Right_2=-v,
        # Left_Support=-v, Right_Support=-v（声明序见 XML）
        v = {"open": 0.0, "closed": -0.91, "reset": None}[state]
        if v is None:
            return None
        q = np.array([v, -v, v, -v, -v, -v], dtype=float)
        names = [
            "gripper0_Left_1_Joint",
            "gripper0_Left_Support_Joint",
            "gripper0_Left_2_Joint",
            "gripper0_Right_2_Joint",
            "gripper0_Right_1_Joint",
            "gripper0_Right_Support_Joint",
        ]
        for n, val in zip(names, q):
            sim.data.set_joint_qpos(n, float(val))
        sim.forward()
        return q.tolist()

    if gripper == "PandaGripper":
        v = {"open": 0.04, "closed": 0.0, "reset": None}[state]
        if v is None:
            return None
        sim.data.set_joint_qpos("gripper0_finger_joint1", float(v))
        sim.data.set_joint_qpos("gripper0_finger_joint2", float(-v))
        sim.forward()
        return [v, -v]

    return None


def jaw_opening_m(sim, gripper):
    """两指垫中心间距（夹爪开口的工程定义）。"""
    if gripper == "ZhixingGripper":
        gl = find_geom(sim, "left_fingerpad_col")
        gr = find_geom(sim, "right_fingerpad_col")
    else:
        gl = find_geom(sim, "finger1_pad_collision")
        gr = find_geom(sim, "finger2_pad_collision")
    if gl is None or gr is None:
        return None, None, None
    cl = geom_center_world(sim, gl)
    cr = geom_center_world(sim, gr)
    return float(np.linalg.norm(cl - cr)), cl.tolist(), cr.tolist()


# --------------------------------------------------------------------------- #
# 视觉占比
# --------------------------------------------------------------------------- #
def view_stats(sim, cam_name, gripper_gids, res=RES):
    """渲染 RGB + 分割，统计夹爪像素占比。"""
    import mujoco

    seg = sim.render(camera_name=cam_name, width=res, height=res, segmentation=True)
    rgb = sim.render(camera_name=cam_name, width=res, height=res)
    if isinstance(rgb, tuple):
        rgb = rgb[0]
    rgb = np.asarray(rgb, dtype=np.uint8)[::-1]
    seg = np.asarray(seg)[::-1]

    # segmentation 像素 = [objtype, objid]；geom objtype == mjOBJ_GEOM
    is_geom = seg[:, :, 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    gid_map = np.where(is_geom, seg[:, :, 1], -1)

    total = gid_map.size
    gset = set(int(g) for g in gripper_gids)
    mask = np.isin(gid_map, list(gset))
    n_grip = int(mask.sum())

    # 底部 40% 行内的夹爪占比（腕部视角里夹爪通常占据画面下方）
    h = gid_map.shape[0]
    bottom = slice(int(h * 0.6), h)
    n_bottom_total = gid_map[bottom, :].size
    n_grip_bottom = int(mask[bottom, :].sum())

    # 夹爪像素的平均颜色（用于「外观」对比）
    mean_color = rgb[mask].mean(axis=0).tolist() if n_grip > 0 else None
    mean_color_all = rgb.reshape(-1, 3).mean(axis=0).tolist()

    return {
        "camera": cam_name,
        "resolution": res,
        "gripper_pixels": n_grip,
        "total_pixels": int(total),
        "gripper_pixel_fraction": float(n_grip / total),
        "gripper_pixel_fraction_bottom40": float(n_grip_bottom / n_bottom_total),
        "mean_color_gripper_rgb": mean_color,
        "mean_color_frame_rgb": mean_color_all,
    }, rgb, mask


def save_png(path, rgb, mask=None):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    if mask is not None:
        vis = rgb.copy()
        vis[mask] = (0.45 * vis[mask] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
        Image.fromarray(vis).save(path.with_name(path.stem + "_mask.png"))


# --------------------------------------------------------------------------- #
# 单配置测量
# --------------------------------------------------------------------------- #
def measure_config(robot, gripper, tag, poses, out_dir, save_png_flag=True):
    import mujoco

    env, suite, task = build_env(robot, gripper)
    env.reset()
    # N1 技术坑：env.reset() 会重建 robot 对象，必须重新取
    rb = env.env.robots[0]
    sim = env.env.sim

    # 冻结物体到统一初态，保证三配置场景一致
    applied = 0
    for n, v in poses.items():
        try:
            sim.data.set_joint_qpos(n, np.array(v, dtype=float))
            applied += 1
        except Exception:
            pass
    sim.forward()

    # 还原 reset 时的关节角记录
    grip_name = find_name(sim, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
    cam_name = find_name(sim, mujoco.mjtObj.mjOBJ_CAMERA, "eye_in_hand")
    agent_name = "agentview"
    gid = sim.model.site_name2id(grip_name)
    cid = sim.model.camera_name2id(cam_name)
    aid = sim.model.camera_name2id(agent_name)

    ggids = gripper_geom_ids(sim)
    res = {"tag": tag, "robot": robot, "gripper": gripper,
           "task": task.language, "objects_frozen": applied,
           "gripper_geoms": len(ggids),
           "jaw": {}, "views": {}, "geometry": {}, "camera": {}}

    # ---- reset 态的夹爪几何 ----
    def snapshot(state_name):
        gpos = np.array(sim.data.site_xpos[gid], dtype=float)
        gmat = np.array(sim.data.site_xmat[gid], dtype=float).reshape(3, 3)
        cpos = np.array(sim.data.cam_xpos[cid], dtype=float)
        cmat = np.array(sim.data.cam_xmat[cid], dtype=float).reshape(3, 3)
        apos = np.array(sim.data.cam_xpos[aid], dtype=float)

        T_grip, T_cam = T_of(gpos, gmat), T_of(cpos, cmat)
        opening, pad_l, pad_r = jaw_opening_m(sim, gripper)
        center = None
        if pad_l is not None:
            center = ((np.array(pad_l) + np.array(pad_r)) / 2.0)

        # 夹爪全部顶点，投影到 grip_site 的 z 轴（接近轴）
        allv = np.vstack([geom_vertices_world(sim, g_) for g_ in ggids]) if ggids else np.zeros((1, 3))
        rel = allv - gpos
        proj = rel @ gmat[:, 2]  # 沿 grip_site 的 z

        rec = {
            "grip_site_world_pos": gpos.tolist(),
            "grip_site_world_quat_wxyz": mat2quat_wxyz(gmat).tolist(),
            "grasp_center_world": center.tolist() if center is not None else None,
            "tcp_offset_world_m": (center - gpos).tolist() if center is not None else None,
            "tcp_offset_local_m": (gmat.T @ (center - gpos)).tolist() if center is not None else None,
            "tcp_offset_norm_mm": float(np.linalg.norm(center - gpos) * 1000) if center is not None else None,
            "jaw_opening_m": opening,
            "gripper_extent_plus_z_m": float(proj.max()),
            "gripper_extent_minus_z_m": float(proj.min()),
            "gripper_lowest_below_gripsite_mm": float(-proj.min() * 1000),
        }
        # 咬合点（两指垫真实最近点对）——不依赖「质心=接触面」的失真假设
        gl, gr = pad_geom_ids(sim, gripper)
        nip = nip_point(sim, gl, gr)
        if nip is not None:
            nc = np.array(nip["nip_center_world"], dtype=float)
            loc = gmat.T @ (nc - gpos)
            rec["nip_center_world"] = nip["nip_center_world"]
            rec["nip_clearance_m"] = nip["nip_clearance_m"]
            rec["nip_offset_world_m"] = (nc - gpos).tolist()
            rec["nip_offset_local_m"] = loc.tolist()
            rec["nip_offset_norm_mm"] = float(np.linalg.norm(nc - gpos) * 1000)
            rec["nip_offset_approach_mm"] = float(loc[2] * 1000)
            rec["nip_offset_lateral_mm"] = float(np.linalg.norm(loc[:2]) * 1000)
        rec["geom_table"] = gripper_geom_table(sim, ggids, gpos, gmat)

        rec["T_cam2grip"] = (np.linalg.inv(T_cam) @ T_grip).tolist()
        rec["T_grip2cam"] = (np.linalg.inv(T_grip) @ T_cam).tolist()
        rec["cam_world_pos"] = cpos.tolist()
        rec["agentview_world_pos"] = apos.tolist()
        rec["dist_cam_to_gripsite_mm"] = float(np.linalg.norm(cpos - gpos) * 1000)

        # right_hand 安装几何
        rh = None
        for cand in ("robot0_right_hand", "right_hand"):
            try:
                rh = sim.model.body_name2id(cand)
                break
            except Exception:
                continue
        if rh is not None:
            hid = int(rh)
            hp = np.array(sim.data.body_xpos[hid], dtype=float)
            hm = np.array(sim.data.body_xmat[hid], dtype=float).reshape(3, 3)
            T_hand = T_of(hp, hm)
            rec["T_hand2grip"] = (np.linalg.inv(T_hand) @ T_grip).tolist()
            rec["T_hand2cam"] = (np.linalg.inv(T_hand) @ T_cam).tolist()
        return rec

    # reset 态
    res["jaw"]["reset"] = None
    res["geometry"]["reset"] = snapshot("reset")

    # 渲染三态视图
    for state in JAW_STATES:
        if state != "reset":
            res["jaw"][state] = set_jaw_state(env, gripper, state)
        for view, cname in (("wrist", cam_name), ("agentview", agent_name)):
            st, rgb, mask = view_stats(sim, cname, ggids)
            key = f"{view}_{state}"
            res["views"][key] = st
            if save_png_flag:
                save_png(out_dir / "renders" / f"{view}_{tag}_{state}.png", rgb, mask)
        if state == "reset":
            res["geometry"]["reset"] = snapshot("reset")
        else:
            res["geometry"][state] = snapshot(state)

    # 关键相机变换（用 reset 态）
    res["camera"] = {
        "T_cam2grip": res["geometry"]["reset"]["T_cam2grip"],
        "T_grip2cam": res["geometry"]["reset"]["T_grip2cam"],
        "T_hand2grip": res["geometry"]["reset"].get("T_hand2grip"),
        "T_hand2cam": res["geometry"]["reset"].get("T_hand2cam"),
        "dist_cam_to_gripsite_mm": res["geometry"]["reset"]["dist_cam_to_gripsite_mm"],
    }

    env.close()
    return res


# --------------------------------------------------------------------------- #
# 对比
# --------------------------------------------------------------------------- #
def compare(results):
    by = {r["tag"]: r for r in results}
    ref = by.get("panda_panda")
    cmp_out = {"reference": "panda_panda", "vs_reference": {}, "eco65_zhixing_vs_eco65_panda": {}}
    if not ref:
        return cmp_out

    for tag, r in by.items():
        if tag == "panda_panda":
            continue
        g, rg = r["geometry"]["reset"], ref["geometry"]["reset"]
        d = {}
        # TCP 位置
        d["grip_site_pos_delta_mm"] = float(
            np.linalg.norm(np.array(g["grip_site_world_pos"]) - np.array(rg["grip_site_world_pos"])) * 1000
        )
        R_a = quat_wxyz2mat(np.array(g["grip_site_world_quat_wxyz"]))
        R_b = quat_wxyz2mat(np.array(rg["grip_site_world_quat_wxyz"]))
        d["grip_site_rot_delta_deg"] = rot_angle_deg(R_a, R_b)
        # 实际夹持区
        if g["grasp_center_world"] and rg["grasp_center_world"]:
            d["grasp_center_delta_mm"] = float(
                np.linalg.norm(np.array(g["grasp_center_world"]) - np.array(rg["grasp_center_world"])) * 1000
            )
        d["tcp_offset_norm_mm"] = g["tcp_offset_norm_mm"]
        d["ref_tcp_offset_norm_mm"] = rg["tcp_offset_norm_mm"]
        d["jaw_opening_reset_m"] = g["jaw_opening_m"]
        d["ref_jaw_opening_reset_m"] = rg["jaw_opening_m"]
        d["gripper_lowest_below_gripsite_mm"] = g["gripper_lowest_below_gripsite_mm"]
        d["ref_gripper_lowest_below_gripsite_mm"] = rg["gripper_lowest_below_gripsite_mm"]
        # 咬合点：沿接近轴伸出 grip_site 多远（决定「下降到位」语义）
        for k in ("nip_offset_approach_mm", "nip_offset_lateral_mm", "nip_offset_norm_mm",
                  "nip_clearance_m"):
            d[k] = g.get(k)
            d[f"ref_{k}"] = rg.get(k)
        # 相机
        cg = np.array(r["camera"]["T_cam2grip"])
        cr = np.array(ref["camera"]["T_cam2grip"])
        d["T_cam2grip_trans_delta_mm"] = float(np.linalg.norm(cg[:3, 3] - cr[:3, 3]) * 1000)
        d["T_cam2grip_rot_delta_deg"] = rot_angle_deg(cg[:3, :3], cr[:3, :3])
        # 视觉占比
        for k in ("wrist_open", "wrist_reset", "wrist_closed", "agentview_open"):
            a = r["views"].get(k, {}).get("gripper_pixel_fraction")
            b = ref["views"].get(k, {}).get("gripper_pixel_fraction")
            if a is not None and b is not None:
                d[f"{k}_grip_frac"] = a
                d[f"{k}_ref_grip_frac"] = b
                d[f"{k}_frac_ratio"] = float(a / b) if b > 0 else None
        cmp_out["vs_reference"][tag] = d

    z = by.get("eco65_zhixing")
    p = by.get("eco65_panda")
    if z and p:
        a = np.array(z["camera"]["T_cam2grip"])
        b = np.array(p["camera"]["T_cam2grip"])
        cmp_out["eco65_zhixing_vs_eco65_panda"] = {
            "T_cam2grip_trans_delta_mm": float(np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000),
            "T_cam2grip_rot_delta_deg": rot_angle_deg(a[:3, :3], b[:3, :3]),
            "conclusion_holds": bool(
                np.linalg.norm(a[:3, 3] - b[:3, 3]) * 1000 < 0.5
                and rot_angle_deg(a[:3, :3], b[:3, :3]) < 0.05
            ),
        }
    return cmp_out


def quat_wxyz2mat(q):
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--init-idx", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-png", action="store_true")
    a = ap.parse_args()

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 统一物体初态：用 Panda 官方 init state 作为跨本体场景基准
    env0, suite0, _ = build_env("Panda", "PandaGripper", a.task_id, a.seed)
    env0.reset()
    env0.set_init_state(suite0.get_task_init_states(a.task_id)[a.init_idx])
    sim0 = env0.env.sim
    import mujoco

    poses = {}
    for i in range(sim0.model._model.njnt):
        n = mujoco.mj_id2name(sim0.model._model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if n.startswith("robot0_") or n.startswith("gripper0_"):
            continue
        poses[n] = np.array(sim0.data.get_joint_qpos(n), dtype=float).tolist()
    env0.close()
    print(f"[setup] 冻结物体关节 {len(poses)} 个（task_id={a.task_id}, init_idx={a.init_idx}）")

    results = []
    for robot, gripper, tag in CONFIGS:
        print(f"[measure] {tag}: {robot} + {gripper}")
        try:
            r = measure_config(robot, gripper, tag, poses, out_dir, not a.no_png)
            results.append(r)
            g = r["geometry"]["reset"]
            print(
                f"    grip_site={np.round(g['grip_site_world_pos'],4).tolist()} "
                f"nip_approach={g.get('nip_offset_approach_mm')}mm "
                f"nip_lateral={g.get('nip_offset_lateral_mm')}mm "
                f"nip_clear={g.get('nip_clearance_m')} "
                f"wrist_frac={r['views']['wrist_reset']['gripper_pixel_fraction']:.4f}"
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            results.append({"tag": tag, "robot": robot, "gripper": gripper, "error": str(e)})

    cmp_out = compare(results)
    blob = {"configs": results, "comparison": cmp_out}
    (out_dir / "tcp_camera_audit.json").write_text(json.dumps(blob, indent=2), encoding="utf-8")
    print(f"\n[out] {out_dir / 'tcp_camera_audit.json'}")
    print(json.dumps(cmp_out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
