"""Expose the frozen original pi0.5 service through RPent's VLA contract."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import urllib.request

import numpy as np
from openpi_client import image_tools
from rpent.robots.components.vla_facade_base import BaseVLAFacade


class ProjectPi05Facade(BaseVLAFacade):
    def __init__(self, policy_url: str, output: Path, seed: int):
        self.url = policy_url
        self.output = output
        output.mkdir(parents=True, exist_ok=False)
        self.log = (output / "predictions.jsonl").open("x", buffering=1)
        self.seed = seed
        self.requests = 0
        with urllib.request.urlopen(policy_url + "/health", timeout=5) as response:
            registration = json.load(response)
        if registration["variant"] != "original_torch" or registration["denoising_steps"] != 10:
            raise ValueError("first Panda verification requires the original ten-step pi0.5 policy")
        (output / "policy_registration.json").write_text(json.dumps(registration, indent=2))
        super().__init__()

    def predict(self, obs: dict, options: dict | None = None) -> np.ndarray:
        main = np.asarray(obs["main_images"])
        wrist = np.asarray(obs["wrist_images"])
        state = np.asarray(obs["states"], np.float32)
        if main.shape != (1, 256, 256, 3) or wrist.shape != main.shape or state.shape != (1, 8):
            raise ValueError("unexpected RPent batched Panda observation")
        if main.dtype != np.uint8 or wrist.dtype != np.uint8 or not np.isfinite(state).all():
            raise ValueError("invalid image/state values")
        prompt = obs["task_descriptions"][0]
        if not isinstance(prompt, str) or not prompt or len(prompt) > 1024:
            raise ValueError("invalid policy instruction")
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(main[0], 224, 224))
        wrist_image = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist[0], 224, 224))
        noise_seed = self.seed * 1000 + self.requests
        payload = {"image": base64.b64encode(image.tobytes()).decode(),
                   "wrist_image": base64.b64encode(wrist_image.tobytes()).decode(),
                   "state": state[0].tolist(), "prompt": prompt, "noise_seed": noise_seed,
                   "client_id": f"rpent_s{self.seed}_prediction{self.requests}"}
        request = urllib.request.Request(self.url + "/infer", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            reply = json.load(response)
        actions = np.asarray(reply["actions"], np.float32)
        if reply["variant"] != "original_torch" or actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError("invalid policy response")
        np.savez_compressed(self.output / f"prediction{self.requests:04d}.npz", image=image,
                            wrist_image=wrist_image, state=state[0], prompt=prompt, actions=actions)
        self.log.write(json.dumps({"prediction": self.requests, "noise_seed": noise_seed,
                                   "prompt": prompt, "state8": state[0].tolist(),
                                   "image_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                                   "wrist_sha256": hashlib.sha256(wrist_image.tobytes()).hexdigest(),
                                   "raw_actions": actions.tolist(), "returned_actions": 5,
                                   "server_request": reply["request"], "timing": reply["timing"]}) + "\n")
        self.requests += 1
        return actions[None, :5]

    def close(self) -> None:
        self.log.close()
        (self.output / "shutdown.json").write_text(json.dumps({"requests": self.requests, "clean_shutdown": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--policy-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()
    ProjectPi05Facade(args.policy_url, args.output, args.seed).serve(
        transport="http", host="127.0.0.1", port=args.port, parent_watch=args.parent_watch)
