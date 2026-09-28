"""轨迹诊断：同一任务、同一随机初始化协议下，对比 ECO65 与 Panda 的
   (a) 末端轨迹 (b) 策略输出的动作统计 (c) 与目标的距离变化。

用来区分两种失败模式：
  * 策略输出本身就是乱动（动作幅度/方向异常）→ 观测分布不匹配（相机/本体）
  * 策略输出合理但末端到不了目标 → 机械臂运动学/工作空间/控制层问题
"""
import collections
import dataclasses
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


@dataclasses.dataclass
class Args:
    robot: str = "ECO65"
    gripper: str = "ZhixingGripper"
    host: str = "127.0.0.1"
    port: int = 8000
    task_suite_name: str = "libero_spatial"
    task_id: int = 0
    num_steps_wait: int = 10
    max_steps: int = 120          # 只看前 120 步，够判断趋势
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
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(img[::-1, ::-1]), 224, 224))


def main(a: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    np.random.seed(a.seed)
    suite = benchmark.get_benchmark_dict()[a.task_suite_name]()
    task = suite.get_task(a.task_id)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    kw = {"bddl_file_name": bddl, "camera_heights": 256, "camera_widths": 256, "robots": [a.robot]}
    if a.gripper and a.gripper != "default":
        kw["gripper_types"] = a.gripper
    env = OffScreenRenderEnv(**kw)
    env.seed(a.seed)
    r0 = env.env.robots[0]
    print(f"\n##### {a.robot} + {type(r0.gripper).__name__} | task={task.language[:60]} #####")
    print(f"robot_model={type(r0.robot_model).__name__} action_dim={r0.action_dim} "
          f"init_qpos={np.round(r0.robot_model.init_qpos, 3)}")

    obs = env.reset()
    client = _websocket_client_policy.WebsocketClientPolicy(a.host, a.port)
    plan = collections.deque()
    t = 0
    rows = []
    eef0 = np.asarray(obs["robot0_eef_pos"]).copy()
    while t < a.max_steps + a.num_steps_wait:
        if t < a.num_steps_wait:
            obs, _, _, _ = env.step(DUMMY)
            t += 1
            continue
        if not plan:
            el = {"observation/image": prep(obs["agentview_image"]),
                  "observation/wrist_image": prep(obs["robot0_eye_in_hand_image"]),
                  "observation/state": build_state(obs), "prompt": str(task.language)}
            chunk = np.asarray(client.infer(el)["actions"])
            plan.extend(chunk[: a.replan_steps])
            rows.append({
                "t": t,
                "eef": np.asarray(obs["robot0_eef_pos"]).copy(),
                "grip": np.asarray(obs["robot0_gripper_qpos"]).copy(),
                "a0": chunk[0].copy(),
                "chunk_absmean": float(np.abs(chunk[:, :6]).mean()),
                "chunk_max": float(np.abs(chunk[:, :6]).max()),
                "grip_cmd": float(chunk[0][6]),
            })
        act = plan.popleft()
        obs, _, done, _ = env.step(np.asarray(act).tolist())
        if done:
            break
        t += 1
    env.close()

    print(f"{'t':>4} {'eef_x':>7} {'eef_y':>7} {'eef_z':>7} {'|Δeef|':>7} "
          f"{'a0_dx':>7} {'a0_dy':>7} {'a0_dz':>7} {'|a|mean':>8} {'grip':>6}")
    for r in rows:
        d = np.linalg.norm(r["eef"] - eef0)
        print(f"{r['t']:>4} {r['eef'][0]:>7.3f} {r['eef'][1]:>7.3f} {r['eef'][2]:>7.3f} {d:>7.3f} "
              f"{r['a0'][0]:>7.3f} {r['a0'][1]:>7.3f} {r['a0'][2]:>7.3f} "
              f"{r['chunk_absmean']:>8.3f} {r['grip_cmd']:>6.2f}")
    if rows:
        eefs = np.stack([r["eef"] for r in rows])
        print(f"-- eef 位移: 起点 {np.round(rows[0]['eef'],3)} 终点 {np.round(eefs[-1],3)} "
              f"总位移 {np.linalg.norm(eefs[-1]-eefs[0]):.3f} m, "
              f"轨迹范围 {np.round(eefs.max(0)-eefs.min(0),3)}")
        print(f"-- 动作幅度: |a[:, :6]| 均值 {np.mean([r['chunk_absmean'] for r in rows]):.3f}, "
              f"最大 {np.max([r['chunk_max'] for r in rows]):.3f}")
        print(f"-- 夹爪指令范围: {min(r['grip_cmd'] for r in rows):.2f} ~ {max(r['grip_cmd'] for r in rows):.2f}")
    print("DIAG_OK")


if __name__ == "__main__":
    main(tyro.cli(Args))
