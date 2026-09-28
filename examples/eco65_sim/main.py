# examples/eco65_sim/main.py
# Launch LIBERO task suite with ECO65 + RethinkGripper via Pi0.5 VLA policy.

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
from openpi_client import websocket_client_policy as _websocket_client_policy
import register_eco65
import tqdm
import tyro

# ---- Monkey-patch before any LIBERO env is created ----
register_eco65.patch()

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

# Save first N episodes per task unconditionally, then only successes
UNCONDITIONAL_SAVE_COUNT = 3


@dataclasses.dataclass
class Args:
    # Model server
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO tasks
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Output
    video_out_path: str = "data/eco65_sim/videos"
    seed: int = 7


def _should_save_video(episode_idx: int, episode_done: bool) -> bool:
    """Decide whether to save video for this episode.

    First UNCONDITIONAL_SAVE_COUNT episodes are always saved.
    After that, only successful episodes are saved.
    """
    if episode_idx < UNCONDITIONAL_SAVE_COUNT:
        return True
    return episode_done


def eval_libero_eco65(args: Args) -> None:
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s (%d tasks)", args.task_suite_name, num_tasks_in_suite)

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = max_steps_map.get(args.task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)

        # OffScreenRenderEnv now creates ECO65 + RethinkGripper (patched)
        env, task_description = _get_eco65_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info("\nTask: %s", task_description)
            episode_done = False
            replay_images = []

            try:
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(initial_states[episode_idx])
            except Exception as e:
                logging.error("Failed to initialize environment: %s", e)
                # Always save init-failure for first N episodes
                if episode_idx < UNCONDITIONAL_SAVE_COUNT:
                    _save_video_safe(replay_images, args.video_out_path,
                                     task_description, episode_idx, "init_failure")
                task_episodes += 1
                total_episodes += 1
                continue

            t = 0
            logging.info("Starting episode %d...", task_episodes + 1)

            while t < max_steps + args.num_steps_wait:
                try:
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        # Record wait-step frames only if this episode will be saved
                        if _should_save_video(episode_idx, False):
                            wait_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                            wait_img = image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wait_img, args.resize_size, args.resize_size)
                            )
                            replay_images.append(wait_img)
                        t += 1
                        continue

                    # Preprocess images (rotate 180 to match train preprocessing)
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Collect frames only if this episode will be saved
                    if _should_save_video(episode_idx, episode_done):
                        replay_images.append(img)

                    if not action_plan:
                        # ECO65 + RethinkGripper: state = eef_pos(3) + eef_rot(3) + gripper_qpos(1) = 7 dims
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"Replan steps {args.replan_steps} exceeds action chunk {len(action_chunk)}"
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

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

            # Save video based on policy: first N always, then only successes
            if _should_save_video(episode_idx, episode_done):
                suffix = "success" if episode_done else "failure"
                _save_video_safe(replay_images, args.video_out_path,
                                 task_description, episode_idx, suffix)
            else:
                logging.info("Skipping video for episode %d (not in first %d and not success)",
                             episode_idx + 1, UNCONDITIONAL_SAVE_COUNT)

            logging.info("Task: %s | Episode: %d | Success: %s | Frames: %d",
                         task_description, episode_idx + 1, episode_done, len(replay_images))
            logging.info("# episodes: %d  # successes: %d (%.1f%%)",
                         total_episodes, total_successes,
                         total_successes / total_episodes * 100 if total_episodes > 0 else 0)

        logging.info("Task %s success rate: %.1f%%",
                     task_description, float(task_successes) / float(task_episodes) * 100 if task_episodes > 0 else 0)

    logging.info("=== Total: %d/%d (%.1f%%) ===",
                 total_successes, total_episodes,
                 total_successes / total_episodes * 100 if total_episodes > 0 else 0)


def _save_video_safe(frames, out_dir, task_description, episode_idx, suffix):
    """Save video for an episode, handling edge cases safely."""
    task_segment = task_description.replace(" ", "_")
    video_path = pathlib.Path(out_dir) / f"rollout_{task_segment}_{episode_idx:03d}_{suffix}.mp4"
    try:
        if frames:
            imageio.mimwrite(video_path, [np.asarray(x) for x in frames], fps=10)
            logging.info("Saved video: %s (%d frames)", video_path.name, len(frames))
        else:
            logging.warning("No frames for %s episode %d, skipping video", task_segment, episode_idx)
    except Exception as e:
        logging.error("Failed to save video %s: %s", video_path, e)


def _get_eco65_libero_env(task, resolution, seed):
    """Create LIBERO environment (patched to use ECO65)."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat = quat / np.linalg.norm(quat)
    elif quat[3] < -1.0:
        quat = quat / np.linalg.norm(quat)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(eval_libero_eco65)
