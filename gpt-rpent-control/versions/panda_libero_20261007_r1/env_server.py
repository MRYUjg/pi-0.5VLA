"""RPent Env RPC facade over the project's unchanged LIBERO/robosuite stack."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite import macros
from robosuite.utils.camera_utils import get_camera_extrinsic_matrix, get_camera_intrinsic_matrix
from robosuite.utils.transform_utils import quat2axisangle
from rpent.robots.components.env_facade_base import BaseEnvFacade
from rpent.utils.rpc.main_thread_serve import MainThreadServeMixin


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


class ProjectLiberoFacade(MainThreadServeMixin, BaseEnvFacade):
    """Own one Panda episode; stop at native success or the registered budget."""

    def __init__(self, suite: str, task: int, seed: int, output: Path, max_steps: int = 220):
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.meta = dict(suite=suite, task=task, seed=seed, max_episode_steps=max_steps,
                         robot="Panda", gripper="PandaGripper", action_dim=7)
        self.max_steps = max_steps
        self.steps = 0
        self.resets = 0
        self.closed = False
        self.writer = None
        self.log = (output / "transitions.jsonl").open("x", buffering=1)
        np.random.seed(seed)
        task_suite = benchmark.get_benchmark_dict()[suite]()
        spec = task_suite.get_task(task)
        self.language = str(spec.language)
        self.initial_states = task_suite.get_task_init_states(task)
        self.init_index = seed % len(self.initial_states)
        bddl = Path(get_libero_path("bddl_files")) / spec.problem_folder / spec.bddl_file
        self.bddl_sha = hashlib.sha256(bddl.read_bytes()).hexdigest()
        self.env = OffScreenRenderEnv(bddl_file_name=str(bddl), robots=["Panda"],
                                      gripper_types="PandaGripper", camera_heights=256,
                                      camera_widths=256, camera_depths=True,
                                      control_freq=20, horizon=1000)
        self.env.seed(seed)
        super().__init__()
        self._rpc["env.raw_obs"] = self.raw_obs
        self._rpc["env.summary"] = self.summary

    def get_env_meta(self) -> dict:
        return dict(self.meta)

    def get_task_language(self) -> str:
        return self.language

    def raw_obs(self) -> dict:
        return self.env.env._get_observations(force_update=True)

    def policy_obs(self, raw: dict) -> dict:
        state = np.concatenate((raw["robot0_eef_pos"],
                                quat2axisangle(raw["robot0_eef_quat"].copy()),
                                raw["robot0_gripper_qpos"])).astype(np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Panda policy requires eight finite state values")
        return {"main_images": np.ascontiguousarray(raw["agentview_image"][::-1, ::-1]),
                "wrist_images": np.ascontiguousarray(raw["robot0_eye_in_hand_image"][::-1, ::-1]),
                "extra_view_images": None, "states": state, "task_descriptions": self.language}

    def reset(self) -> tuple[dict, dict]:
        if self.resets:
            if self.steps:
                raise RuntimeError("evaluation episode cannot reset after actions; use a fresh run")
            # BaseEnvClient and LiberoPrimitives both initialize via reset.
            # Before any action, return the same settled initial scene without advancing physics.
            return self.policy_obs(self.raw_obs()), {"initialization": json.loads(
                (self.output / "registration.json").read_text()), "initial_scene_reused": True}
        self.resets += 1
        self.env.reset()
        initial = np.asarray(self.initial_states[self.init_index])
        np.save(self.output / "official_initial.npy", initial, allow_pickle=False)
        self.env.set_init_state(initial)
        for _ in range(10):
            self.env.step([0.0] * 6 + [-1.0])
        if self.env.check_success():
            raise RuntimeError("initial episode is already successful")
        self.steps = 0
        robot = self.env.robots[0]
        model_name = type(robot.robot_model).__name__
        gripper_name = type(robot.gripper).__name__
        if model_name not in {"Panda", "MountedPanda", "OnTheGroundPanda"} or gripper_name != "PandaGripper":
            raise RuntimeError(f"unexpected embodiment: {model_name}/{gripper_name}")
        c = robot.controller
        config = {k: np.asarray(getattr(c, k)).tolist() for k in
                  ["input_min", "input_max", "output_min", "output_max", "kp", "kd", "damping_ratio"]
                  if hasattr(c, k)}
        config.update(robot_model=model_name, gripper_model=gripper_name,
                      controller=type(c).__name__, control_freq=self.env.env.control_freq,
                      control_delta=bool(c.use_delta), extra_joint_pd_profile="legacy",
                      model_timestep=float(self.env.sim.model.opt.timestep))
        raw = self.raw_obs()
        obs = self.policy_obs(raw)
        np.savez_compressed(self.output / "initial_policy_obs.npz", **{k: v for k, v in obs.items() if v is not None})
        registration = dict(self.meta, language=self.language, init_index=self.init_index,
                            initial_state_sha256=digest(initial), bddl_sha256=self.bddl_sha,
                            controller=config, image_convention=macros.IMAGE_CONVENTION,
                            XML_sha256=hashlib.sha256(self.env.sim.model.get_xml().encode()).hexdigest(),
                            initial_main_sha256=digest(obs["main_images"]),
                            initial_wrist_sha256=digest(obs["wrist_images"]),
                            settled_state_sha256=digest(self.env.get_sim_state()))
        (self.output / "registration.json").write_text(json.dumps(registration, indent=2))
        self.writer = imageio.get_writer(self.output / "dual_camera.mp4", fps=20,
                                         codec="libx264", quality=7,
                                         ffmpeg_params=["-threads", "1"])
        self.writer.append_data(np.concatenate([obs["main_images"], obs["wrist_images"]], axis=1))
        return obs, {"initialization": registration}

    def summary(self) -> dict:
        return dict(self.meta, steps=self.steps, native_success=bool(self.env.check_success()),
                    truncated=self.steps >= self.max_steps, init_index=self.init_index,
                    simulation_time=float(self.env.sim.data.time))

    def step(self, action) -> tuple:
        if self.steps >= self.max_steps or self.env.check_success():
            raise RuntimeError("action requested after terminal episode")
        proposed = np.asarray(action, dtype=np.float32)
        if proposed.shape != (7,) or not np.isfinite(proposed).all():
            raise ValueError("actions must contain seven finite values")
        executed = np.clip(proposed, -1, 1)
        before = float(self.env.sim.data.time)
        raw, reward, done, info = self.env.step(executed)
        self.steps += 1
        native = bool(self.env.check_success())
        truncated = self.steps >= self.max_steps and not native
        after = float(self.env.sim.data.time)
        if not np.isclose(after - before, .05, atol=1e-8):
            raise RuntimeError("unexpected simulation control interval")
        obs = self.policy_obs(raw)
        entry = dict(step=self.steps, proposed_action=proposed.tolist(),
                     executed_action=executed.tolist(), clipped_dimensions=(proposed != executed).tolist(),
                     time_before=before, time_after=after, native_success=native, env_done=bool(done),
                     truncated=truncated, state8=obs["states"].tolist(),
                     qpos=self.env.sim.data.qpos.tolist(), qvel=self.env.sim.data.qvel.tolist(),
                     ctrl=self.env.sim.data.ctrl.tolist(), main_sha256=digest(obs["main_images"]),
                     wrist_sha256=digest(obs["wrist_images"]))
        self.log.write(json.dumps(entry) + "\n")
        self.writer.append_data(np.concatenate([obs["main_images"], obs["wrist_images"]], axis=1))
        return obs, float(reward), native, truncated, {"env_done": bool(done), "control_step": self.steps}

    def chunk_step(self, actions, *, return_all_frames: bool = False) -> tuple:
        arr = np.asarray(actions, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 7 or not 1 <= len(arr) <= 5 or not np.isfinite(arr).all():
            raise ValueError("VLA chunks must contain one to five seven-dimensional finite actions")
        frames, rewards, terminated, truncated = [], [], [], []
        info = {}
        for action in arr:
            obs, reward, term, trunc, info = self.step(action)
            frames.append(obs); rewards.append(reward); terminated.append(term); truncated.append(trunc)
            if term or trunc:
                break
        return (frames if return_all_frames else frames[-1], np.asarray(rewards),
                np.asarray(terminated), np.asarray(truncated), info)

    def get_camera_meta(self, camera_name="agentview", height=256, width=256) -> dict:
        if camera_name not in {"agentview", "robot0_eye_in_hand"}:
            raise ValueError("unsupported camera")
        sim = self.env.sim
        return {"camera_name": camera_name, "height": height, "width": width,
                "intrinsic_K": get_camera_intrinsic_matrix(sim, camera_name, height, width).tolist(),
                "extrinsic_cam2world": get_camera_extrinsic_matrix(sim, camera_name).tolist(),
                "depth_near": float(sim.model.vis.map.znear * sim.model.stat.extent),
                "depth_far": float(sim.model.vis.map.zfar * sim.model.stat.extent)}

    def render_camera(self, camera_name="agentview", height=1024, width=1024, depth=False):
        if camera_name not in {"agentview", "robot0_eye_in_hand"} or not 1 <= height <= 1024 or not 1 <= width <= 1024:
            raise ValueError("unsupported camera/resolution")
        result = self.env.sim.render(camera_name=camera_name, height=height, width=width, depth=depth)
        # Match robosuite's raw observation convention; RPent applies the calibration flip.
        stride = {"opengl": 1, "opencv": -1}[macros.IMAGE_CONVENTION]
        if depth:
            return result[0][::stride], result[1][::stride]
        return result[::stride]

    def close(self) -> None:
        if self.closed:
            return
        if self.resets:
            (self.output / "summary.json").write_text(json.dumps(self.summary(), indent=2))
        if self.writer:
            self.writer.close()
        self.log.close()
        self.env.close()
        self.closed = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    ProjectLiberoFacade(args.suite, args.task, args.seed, args.output).serve(
        transport="http", host="127.0.0.1", port=args.port, parent_watch=args.parent_watch)
