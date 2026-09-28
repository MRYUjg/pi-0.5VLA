# examples/eco65_sim/main.py
# Launch ECO65 + RethinkGripper simulation with Pi0.5 VLA policy.
import dataclasses
import logging
import pathlib

import env as _env
import imageio
import numpy as np
from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy
import tyro


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    task: str = "Lift"
    seed: int = 0
    action_horizon: int = 10
    num_episodes: int = 5
    max_episode_steps: int = 300
    video_dir: str = "data/eco65_sim/videos"


def main(args: Args) -> None:
    env = _env.ECO65SimEnvironment(task=args.task, seed=args.seed)
    policy = action_chunk_broker.ActionChunkBroker(
        policy=websocket_client_policy.WebsocketClientPolicy(
            host=args.host,
            port=args.port,
        ),
        action_horizon=args.action_horizon,
    )

    out_dir = pathlib.Path(args.video_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for episode in range(args.num_episodes):
        logging.info("=== Episode %d ===", episode + 1)
        env.reset()
        policy.reset()

        frames = []
        for step in range(args.max_episode_steps):
            obs = env.get_observation()
            action = policy.infer(obs)
            env.apply_action(action)

            # Save frame from agentview camera
            frames.append(np.asarray(obs["observation/image"]))

            if env.is_episode_complete():
                logging.info("Episode done at step %d", step + 1)
                break
        else:
            logging.info("Episode reached max steps (%d)", args.max_episode_steps)

        # Save video
        video_path = out_dir / f"episode_{episode + 1:03d}.mp4"
        imageio.mimwrite(video_path, frames, fps=10)
        logging.info("Saved video: %s", video_path)

    logging.info("All episodes complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
