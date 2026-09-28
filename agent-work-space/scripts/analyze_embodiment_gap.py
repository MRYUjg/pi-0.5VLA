"""量化「换机械臂为什么变差」：对同一任务/同一随机初始化，测三件事。

1) 几何：机器人基座、初始 eef、目标物体的世界坐标与距离（可达性）
2) 相机：agentview / eye_in_hand 的世界位姿与视线方向（视觉分布差异）
3) 控制：关节限位占用率、指令位移 vs 实际位移（跟踪误差）

用法（容器内）：python analyze_embodiment_gap.py --robot ECO65 --gripper PandaGripper
"""
import collections
import dataclasses
import json
import logging
import math
import pathlib

import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

DUMMY = [0.0] * 6 + [-1.0]
OSC_POS_SCALE = 0.05    # osc_pose.json output_max 前三维
OSC_ROT_SCALE = 0.5     # 后三维


@dataclasses.dataclass
class Args:
    robot: str = "ECO65"
    gripper: str = "PandaGripper"
    host: str = "127.0.0.1"
    port: int = 8000
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_steps_wait: int = 10
    max_steps: int = 120
    replan_steps: int = 5
    seed: int = 7


def quat2axisangle(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[3] = min(q[3], 1.0)
    den = math.sqrt(1.0 - q[3] * q[3])
    return np.zeros(3) if math.isclose(den, 0.0) else (q[:3] * 2.0 * math.acos(q[3])) / den


def build_state(obs):
    gq = np.asarray(obs["robot0_gripper_qpos"])
    grip = gq[[1, 5]] if gq.shape[0] >= 6 else gq[:2]
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), grip))


def prep(img):
    return image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(img[::-1, ::-1]), 224, 224))


def cam_info(sim, name):
    mid = sim.model.camera_name2id(name)
    return {
        "name": name,
        "pos": np.round(sim.model.cam_pos[mid], 4).tolist(),
        "quat": np.round(sim.model.cam_quat[mid], 4).tolist(),
        "fovy": float(sim.model.cam_fovy[mid]),
    }


def main(a: Args) -> None:
    logging.basicConfig(level=logging.WARNING)
    np.random.seed(a.seed)
    suite = benchmark.get_benchmark_dict()[a.task_suite_name]()
    task = suite.get_task(a.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    kw = {"bddl_file_name": bddl, "camera_heights": 256, "camera_widths": 256,
          "robots": [a.robot], "camera_names": ["agentview", "robot0_eye_in_hand", "robot0_robotview"]}
    if a.gripper and a.gripper != "default":
        kw["gripper_types"] = a.gripper
    env = OffScreenRenderEnv(**kw)
    env.seed(a.seed)

    # 注意：hard_reset=True 时 env.reset() 会重建 sim/robot 对象，必须在 reset 之后再取引用
    obs = env.reset()
    robot = env.env.robots[0]
    sim = env.env.sim
    model = robot.robot_model
    out = {
        "robot": a.robot,
        "gripper": type(robot.gripper).__name__,
        "robot_model_class": type(model).__name__,
        "n_arm_joints": len(model.joints),
        "action_dim": robot.action_dim,
        "init_qpos": np.round(model.init_qpos, 4).tolist(),
        "cameras": {},
    }

    # --- 几何 ---
    obj_keys = [k for k in obs if k.endswith("_pos") and not k.startswith("robot0")]
    out["robot_base_pos"] = np.round(robot.base_pos, 4).tolist()
    out["eef_pos_t10"] = np.round(obs["robot0_eef_pos"], 4).tolist()
    dists = {k: float(np.linalg.norm(np.asarray(obs["robot0_eef_pos"]) - np.asarray(obs[k])))
             for k in obj_keys}
    out["obj_pos"] = {k: np.round(obs[k], 4).tolist() for k in obj_keys}
    out["eef_to_obj_dist"] = {k: round(v, 4) for k, v in sorted(dists.items(), key=lambda x: x[1])}
    base = np.asarray(robot.base_pos)
    out["obj_dist_from_base"] = {k: round(float(np.linalg.norm(np.asarray(obs[k]) - base)), 4)
                                for k in obj_keys}
    # --- 相机 ---
    for cam in ("agentview", "robot0_eye_in_hand"):
        if cam in [sim.model.camera_id2name(i) for i in range(sim.model.ncam)]:
            out["cameras"][cam] = cam_info(sim, cam)
    # 腕部相机到 eef 的距离（视觉尺度代理量）
    if "robot0_eye_in_hand" in out["cameras"]:
        cp = np.asarray(out["cameras"]["robot0_eye_in_hand"]["pos"])
        out["wrist_cam_to_eef"] = round(float(np.linalg.norm(cp - np.asarray(obs["robot0_eef_pos"]))), 4)

    # --- 关节限位 ---
    lo = sim.model.jnt_range[:, 0]
    hi = sim.model.jnt_range[:, 1]
    jidx = np.asarray(robot._ref_joint_pos_indexes)          # noqa: SLF001
    span = lo[jidx] * 0 + (hi[jidx] - lo[jidx])
    out["joint_range"] = np.round(np.stack([lo[jidx], hi[jidx]], 1), 3).tolist()
    # 执行器力矩上限与关节阻尼（判断是否"力气不够"）

    # --- 闭环跑：记录限位占用与跟踪误差 ---
    client = _websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
    plan = collections.deque()
    t = 0
    limit_util = []          # 每步各关节在行程中的位置(0=下限,1=上限)
    track_err = []           # 每个动作块的「指令位移 vs 实际位移」
    while t < a.max_steps + a.num_steps_wait:
        if t < a.num_steps_wait:
            obs, _, _, _ = env.step(DUMMY)
            t += 1
            continue
        q = sim.data.qpos[jidx]
        limit_util.append((q - lo[jidx]) / np.maximum(span, 1e-9))
        if not plan:
            el = {"observation/image": prep(obs["agentview_image"]),
                  "observation/wrist_image": prep(obs["robot0_eye_in_hand_image"]),
                  "observation/state": build_state(obs), "prompt": str(task.language)}
            chunk = np.asarray(client.infer(el)["actions"])
            plan.extend(chunk[: a.replan_steps])
            eef_before = np.asarray(obs["robot0_eef_pos"]).copy()
            cmd_disp = np.zeros(3)
            n_cmd = 0
        act = np.asarray(plan.popleft())
        cmd_disp = cmd_disp + act[:3] * OSC_POS_SCALE
        n_cmd += 1
        obs, _, done, _ = env.step(act.tolist())
        if n_cmd == a.replan_steps or not plan:
            achieved = np.asarray(obs["robot0_eef_pos"]) - eef_before
            track_err.append({"cmd": cmd_disp.copy(), "got": achieved.copy(),
                              "err": float(np.linalg.norm(cmd_disp - achieved))})
        if done:
            break
        t += 1
    env.close()

    lu = np.stack(limit_util) if limit_util else np.zeros((1, len(jidx)))
    out["joint_limit_util_min"] = np.round(lu.min(0), 3).tolist()
    out["joint_limit_util_max"] = np.round(lu.max(0), 3).tolist()
    out["joint_near_limit_frac"] = np.round(((lu < 0.05) | (lu > 0.95)).mean(0), 3).tolist()
    if track_err:
        errs = np.array([r["err"] for r in track_err])
        cmdn = np.array([np.linalg.norm(r["cmd"]) for r in track_err])
        gotn = np.array([np.linalg.norm(r["got"]) for r in track_err])
        out["track_err_mean_m"] = round(float(errs.mean()), 4)
        out["track_err_max_m"] = round(float(errs.max()), 4)
        out["cmd_disp_mean_m"] = round(float(cmdn.mean()), 4)
        out["got_disp_mean_m"] = round(float(gotn.mean()), 4)
        out["track_ratio"] = round(float(gotn.sum() / max(cmdn.sum(), 1e-9)), 3)
        out["n_blocks"] = len(track_err)

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(tyro.cli(Args))
