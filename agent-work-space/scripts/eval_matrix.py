"""参数化 LIBERO 评测（examples/libero/main.py 的可对照版本）。

目的：把「机器人 / 夹爪」作为唯一变量做消融，回答
  * 评测效果差是不是夹爪造成的？
  * 这条链路本身能不能跑出正常成功率（阳性对照 Panda + PandaGripper）？

三格对照（同一份代码、同一 seed、同一任务集、同样 replan_steps）：
  A) --robot ECO65 --gripper ZhixingGripper   （当前适配方案，基线）
  B) --robot ECO65 --gripper PandaGripper     （只换夹爪）
  C) --robot Panda --gripper PandaGripper     （训练时的原始配置，阳性对照）

用法（在 libero 容器内）：
  python eval_matrix.py --robot ECO65 --gripper PandaGripper --tag B_eco65_panda
"""
import collections
import dataclasses
import json
import logging
import math
import pathlib
import time

import imageio
import numpy as np
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    # --- 消融变量 ---
    robot: str = "ECO65"            # ECO65 | Panda
    gripper: str = "ZhixingGripper"  # ZhixingGripper | PandaGripper | RethinkGripper | default
    # auto: ECO65 用随机 reset（与原 main.py 一致），其余用 set_init_state
    init_mode: str = "auto"          # auto | random | init_state
    tag: str = "run"                 # 结果目录/文件名标签

    # --- 模型服务 ---
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # --- LIBERO ---
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 1
    seed: int = 7
    max_tasks: int = 0  # 0 = 跑完整任务集；>0 用于冒烟

    # --- 输出 ---
    out_root: str = "/app/agent-work-space/eval"


# 夹爪 state 选择器（与 examples/libero/main.py 保持同一约定）
#   ZhixingGripper: 6 个联动关节，取两个 Support 关节 [1, 5]
#   PandaGripper / RethinkGripper: 2 个平动指关节 slice(0, 2)
GRIPPER_STATE_SELECTOR = {
    "ZhixingGripper": [1, 5],
    "PandaGripper": slice(0, 2),
    "RethinkGripper": slice(0, 2),
}


def resolve_gripper_selector(env):
    """按实际挂载的夹爪返回 (类名, state 选择器)。"""
    gripper = env.env.robots[0].gripper
    name = type(gripper).__name__
    n = len(env.env.robots[0]._ref_gripper_joint_pos_indexes)
    if name not in GRIPPER_STATE_SELECTOR:
        raise ValueError(f"未登记的夹爪类型 {name}（{n} 关节）：请补充 GRIPPER_STATE_SELECTOR")
    sel = GRIPPER_STATE_SELECTOR[name]
    need = (max(sel) if isinstance(sel, list) else sel.stop - 1)
    if need >= n:
        raise ValueError(f"{name} 的选择器 {sel} 需要 >= {need+1} 个关节，实际 {n}")
    return name, sel


def quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def build_state(obs, selector):
    """8 维 state：eef_pos(3) + axisangle(3) + 夹爪两位，夹爪取法由 selector 决定。"""
    gq = np.asarray(obs["robot0_gripper_qpos"])
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), gq[selector]))


def preprocess(img):
    img = np.ascontiguousarray(img[::-1, ::-1])  # 与训练一致：旋转 180°
    return image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)

    suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    n_tasks = min(suite.n_tasks, args.max_tasks) if args.max_tasks else suite.n_tasks
    max_steps = MAX_STEPS[args.task_suite_name]

    out_dir = pathlib.Path(args.out_root) / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    results = []
    total_eps = total_ok = 0
    t_start = time.time()

    for task_id in tqdm.tqdm(range(n_tasks)):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        env_kwargs = {
            "bddl_file_name": bddl,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
            "robots": [args.robot],
        }
        if args.gripper and args.gripper != "default":
            env_kwargs["gripper_types"] = args.gripper
        env = OffScreenRenderEnv(**env_kwargs)
        env.seed(args.seed)

        robot0 = env.env.robots[0]
        gname, gsel = resolve_gripper_selector(env)
        if task_id == 0:
            logging.info("gripper selector: %s -> %s", gname, gsel)
            logging.info("=" * 70)
            logging.info("robot_model = %s | gripper = %s | dof = %s | action_dim = %s",
                         type(robot0.robot_model).__name__, type(robot0.gripper).__name__,
                         getattr(robot0.gripper, "dof", "?"), robot0.action_dim)
            logging.info("controller  = %s", type(robot0.controller).__name__)
            logging.info("=" * 70)

        task_eps = task_ok = 0
        for ep in range(args.num_trials_per_task):
            env.reset()
            action_plan = collections.deque()
            done = False
            obs = None

            mode = args.init_mode
            if mode == "auto":
                mode = "random" if "ECO65" in type(env.env.robots[0].robot_model).__name__ else "init_state"
            try:
                obs = env.reset() if mode == "random" else env.set_init_state(init_states[ep])
            except Exception as exc:  # noqa: BLE001
                logging.error("init failed (%s): %s", mode, exc)
                results.append({"task": task.language, "episode": ep, "success": False,
                                "reason": f"init_failure: {exc}", "steps": 0})
                task_eps += 1
                total_eps += 1
                continue

            replay_main, replay_wrist = [], []
            t0 = time.time()
            t = 0
            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = preprocess(obs["agentview_image"])
                    wrist = preprocess(obs["robot0_eye_in_hand_image"])
                    replay_main.append(img)
                    replay_wrist.append(wrist)

                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": build_state(obs, gsel),
                            "prompt": str(task.language),
                        }
                        chunk = client.infer(element)["actions"]
                        assert len(chunk) >= args.replan_steps, f"policy returned {len(chunk)} steps"
                        action_plan.extend(chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    obs, _, done, _ = env.step(np.asarray(action).tolist())
                    if done:
                        break
                    t += 1
                except Exception as exc:  # noqa: BLE001
                    logging.error("rollout exception: %s", exc)
                    break

            dt = time.time() - t0
            task_eps += 1
            total_eps += 1
            if done:
                task_ok += 1
                total_ok += 1

            suffix = "success" if done else "failure"
            desc = task.language.replace(" ", "_")
            stem = f"{desc}__ep{ep:02d}"
            if replay_main:
                imageio.mimwrite(video_dir / f"{stem}__{suffix}.mp4",
                                 [np.asarray(x) for x in replay_main], fps=10)
                imageio.mimwrite(video_dir / f"wrist__{stem}__{suffix}.mp4",
                                 [np.asarray(x) for x in replay_wrist], fps=10)
            results.append({"task": task.language, "episode": ep, "success": bool(done),
                            "steps": t, "seconds": round(dt, 1), "init_mode": mode})
            logging.info("[%s] task %d/%d ep %d | success=%s | steps=%d | %.1fs | 累计 %d/%d",
                         args.tag, task_id + 1, n_tasks, ep + 1, done, t, dt, total_ok, total_eps)

        logging.info("[%s] task %d/%d 完成：%d/%d", args.tag, task_id + 1, n_tasks, task_ok, task_eps)
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass

    summary = {
        "tag": args.tag,
        "robot": args.robot,
        "gripper": args.gripper,
        "init_mode": args.init_mode,
        "task_suite": args.task_suite_name,
        "num_trials_per_task": args.num_trials_per_task,
        "seed": args.seed,
        "replan_steps": args.replan_steps,
        "total_episodes": total_eps,
        "total_successes": total_ok,
        "success_rate": (total_ok / total_eps) if total_eps else 0.0,
        "wall_seconds": round(time.time() - t_start, 1),
        "episodes": results,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logging.info("=" * 70)
    logging.info("[%s] robot=%s gripper=%s  ==>  %d/%d = %.1f%%  (%.0fs)",
                 args.tag, args.robot, args.gripper, total_ok, total_eps,
                 100.0 * summary["success_rate"], summary["wall_seconds"])
    logging.info("=" * 70)


if __name__ == "__main__":
    main(tyro.cli(Args))
