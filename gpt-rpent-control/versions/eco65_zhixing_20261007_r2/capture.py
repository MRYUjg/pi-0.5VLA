"""Fingerprint reused framework/controller source and verify frozen policy files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from launch import REPO, ROOT, RPENT


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture() -> dict:
    model_root = REPO / "agent-work-space/6-model-inference"
    frozen = json.loads((model_root / "releases/trt_t4_20261007_r1/manifest.json").read_text())
    selected = {name: entry["sha256"] for name, entry in frozen["files_relative_to_6_model_inference"].items()
                if name.startswith("versions/trt_t4_20261007_r1/")}
    verified = {name: sha(model_root / name) == expected for name, expected in selected.items()}
    if not verified or not all(verified.values()):
        raise RuntimeError("frozen T4 source mismatch")
    revision = subprocess.check_output(["git", "-C", str(RPENT), "rev-parse", "HEAD"], text=True).strip()
    clean = subprocess.run(["git", "-C", str(RPENT), "diff", "--quiet"]).returncode == 0
    if revision != "068cd64f4f175179a13c67d429a4d00ca56fb196" or not clean:
        raise RuntimeError("RPent checkout changed from the read baseline")
    files = {}
    for base in [RPENT / "rpent", RPENT / "robots/libero", REPO / "robosuite",
                 REPO / "third_party/libero/libero/libero/envs", REPO / "src/openpi",
                 REPO / "packages/openpi-client/src"]:
        for path in base.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".xml", ".json"} and "__pycache__" not in path.parts:
                files[str(path)] = sha(path)
    return {"RPent_revision": revision, "RPent_tracked_source_clean": clean,
            "OpenPI_revision": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
            "T4_frozen_source_verified": verified, "reused_files_sha256": files,
            "policy_image": subprocess.check_output(["docker", "image", "inspect", "pi05_trt:20261007-r1", "--format", "{{.Id}}"], text=True).strip(),
            "simulation_image": subprocess.check_output(["docker", "image", "inspect", "libero:latest", "--format", "{{.Id}}"], text=True).strip()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.output.open("x") as file:
        json.dump(capture(), file, indent=2)
    print("framework, controller and frozen policy source fingerprints captured")
