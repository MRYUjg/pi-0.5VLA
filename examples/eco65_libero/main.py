"""ECO65 + RethinkGripper LIBERO evaluation script.

Uses monkey-patching (register_eco65.patch()) to intercept
SingleArmEnv.__init__ and force ECO65 instead of Panda.
Follows the same pattern as examples/eco65_sim/main.py.
"""

import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _ws
import register_eco65
import tqdm
import tyro

# ---- Monkey-patch before any LIBERO env is created ----
register_eco65.patch()

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    video_out_path: str = "data/eco65_libero/videos"
    seed: int = 7


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle (from robosuite)."""
    quat = np.asarray(quat)
    if quat[3] > 1.0:
        quat = quat / np.linalg.norm(quat)
    elif quat[3] < -1.0:
        quat = quat / np.linalg.norm(quat)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def eval_libero_eco65(args: Args) -> None:
    np.random.seed(args.seed)

    print("=" * 60)
    print("ECO65 + RethinkGripper  LIBERO Evaluation")
    print("=" * 60)
    print(f"  Server:      ws://{args.host}:{args.port}")
    print(f"  Task suite:  {args.task_suite_name}")
    print(f"  Trials/task: {args.num_trials_per_task}")
    print(f"  Video dir:   {args.video_out_path}")
    print()

    # Load LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks = task_suite.n_tasks
    logging.info("Task suite: %s (%d tasks)", args.task_suite_name, num_tasks)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    max_steps = _MAX_STEPS[args.task_suite_name]

    # Connect to pi05 server
    print("[1] Connecting to model server...")
    client = _ws.WebsocketClientPolicy(args.host, args.port)
    metadata = client.get_server_metadata()
    print(f"    Connected. Server metadata: {list(metadata.keys())}")
    print()

    # Warmup inference
    print("[2] Warmup inference...")
    img = np.random.randint(0, 256, (args.resize_size, args.resize_size, 3), dtype=np.uint8)
    warmup = client.infer({
        "observation/image": img,
        "observation/wrist_image": img,
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": "warmup",
    })
    print(f"    Action shape: {warmup['actions'].shape}, "
          f"range: [{warmup['actions'].min():.4f}, {warmup['actions'].max():.4f}]")
    if "policy_timing" in warmup:
        print(f"    Infer time: {warmup['policy_timing'].get('infer_ms', 'N/A'):.1f} ms")
    print()

    # Main evaluation loop
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        env, task_description = _get_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}", leave=False):
            logging.info("\nTask: %s", task_description)
            episode_done = False
            replay_images = []

            try:
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(initial_states[episode_idx])
            except Exception as e:
                logging.error("Failed to initialize: %s", e)
                _save_video(replay_images, args.video_out_path,
                            task_description, episode_idx, "init_failure")
                task_episodes += 1
                total_episodes += 1
                continue

            t = 0
            logging.info("Starting episode %d...", task_episodes + 1)

            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects to settle
                    if t < args.num_steps_wait:
                        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                        # Include wait frames in replay
                        wimg = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wimg = image_tools.convert_to_uint8(
                            image_tools.resize_with_pad(wimg, args.resize_size, args.resize_size)
                        )
                        replay_images.append(wimg)
                        t += 1
                        continue

                    # Preprocess images
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)

                    if not action_plan:
                        # ECO65 state: eef_pos(3) + eef_rot(3) + gripper(1) = 7 dims
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"Need {args.replan_steps} steps, got {len(action_chunk)}"
                        )
                        action_plan.extend(action_chunk[:args.replan_steps])

                    action = action_plan.popleft()
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        episode_done = True
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error("Exception at step %d: %s", t, e)
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if episode_done else "failure"
            _save_video(replay_images, args.video_out_path,
                        task_description, episode_idx, suffix)

            sr = total_successes / total_episodes * 100
            logging.info("Done: %s | frames: %d | total: %d/%d (%.1f%%)",
                         episode_done, len(replay_images),
                         total_successes, total_episodes, sr)

        logging.info("Task %d SR: %d/%d = %.1f%%",
                     task_id, task_successes, task_episodes,
                     task_successes / task_episodes * 100 if task_episodes else 0)

    logging.info("\n===== FINAL: %d/%d = %.1f%% =====",
                 total_successes, total_episodes,
                 total_successes / total_episodes * 100 if total_episodes else 0)


def _save_video(frames, out_dir, task_description, episode_idx, suffix):
    """Save episode replay video."""
    task_segment = task_description.replace(" ", "_")
    video_path = pathlib.Path(out_dir) / f"rollout_{task_segment}_{episode_idx:03d}_{suffix}.mp4"
    try:
        if frames:
            imageio.mimwrite(video_path, [np.asarray(x) for x in frames], fps=10)
            logging.info("Saved video: %s (%d frames)", video_path.name, len(frames))
        else:
            logging.warning("No frames for %s episode %d", task_segment, episode_idx)
    except Exception as e:
        logging.error("Failed to save video %s: %s", video_path, e)


def _get_env(task, resolution, seed):
    """Create LIBERO environment (monkey-patched to use ECO65)."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task_description


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(eval_libero_eco65)
