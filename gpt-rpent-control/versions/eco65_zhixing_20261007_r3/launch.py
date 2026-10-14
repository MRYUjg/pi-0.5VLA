"""Host runner: isolated real simulation, stdin credential, immutable outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parent
RPENT = Path("/mnt/data/vla-policy/RPent")


def run_task(task: int, seed: int, run_id: str, mode: str,
             policy_url: str = "http://127.0.0.1:18981") -> dict:
    output = ROOT / "runs" / run_id
    output.mkdir(parents=True, exist_ok=True)
    label = f"task{task:02d}_seed{seed}"
    command = ["docker", "run", "--rm", "-i", "--name", "rpent-" + run_id.replace("_", "-") + "-" + label,
               "--network", "host", "--cpus", "2", "--memory", "5g",
               "-e", "PYTHONPATH=/rpent:/app:/app/packages/openpi-client/src:/app/third_party/libero",
               "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "MUJOCO_GL=osmesa",
               "-e", "OPENBLAS_NUM_THREADS=1", "-e", "OMP_NUM_THREADS=1", "-e", "LP_NUM_THREADS=1",
               "-e", "EXTRA_JOINT_PD_PROFILE=legacy", "-e", "EXTRA_JOINT_PD_KP=",
               "-v", str(REPO) + ":/app:ro", "-v", str(RPENT) + ":/rpent:ro",
               "-v", str(ROOT) + ":/control:ro", "-v", str(output) + ":/output",
               "-v", "/dev/null:/app/gpt-6-control/deepseek_api_key:ro",
               "-w", "/control/versions/eco65_zhixing_20261007_r3",
               "--entrypoint", "/control/runtime/py310/bin/python", "libero:latest",
               "/control/versions/eco65_zhixing_20261007_r3/run_episode.py",
               "--output", "/output/" + label, "--task", str(task), "--seed", str(seed), "--mode", mode,
               "--policy-url", policy_url]
    credential = (REPO / "gpt-6-control/deepseek_api_key").read_text().strip() if mode == "planner" else ""
    start = time.monotonic()
    with (output / (label + ".log")).open("x") as log:
        proc = subprocess.run(command, input=credential + "\n", text=True, stdout=log, stderr=subprocess.STDOUT)
    evidence = dict(task=task, seed=seed, mode=mode, command=command, exit_code=proc.returncode,
                    seconds=time.monotonic() - start)
    (output / (label + "_exit.json")).write_text(json.dumps(evidence, indent=2))
    print(json.dumps({k: v for k, v in evidence.items() if k != "command"}), flush=True)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=["chain", "planner"], default="planner")
    args = parser.parse_args()
    records = [run_task(task, args.seed, args.run_id, args.mode) for task in args.tasks]
    sys.exit(0 if all(r["exit_code"] == 0 for r in records) else 1)
