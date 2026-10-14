"""Own original policy service plus a bounded Spatial task validation campaign."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from launch import REPO, ROOT, run_task

MODEL_ROOT = REPO / "agent-work-space/6-model-inference"
POLICY_IMAGE = "sha256:cf96556ab8b55eb3f80770d2a684d6961c0128c8100660a8cc4c7e61ecc60369"


def write(path: Path, data) -> None:
    with path.open("x") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main(args) -> int:
    output = ROOT / "runs" / args.run_id
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    name = "rpent-policy-" + args.run_id.replace("_", "-")
    policy_output = output / "policy"
    policy_output.mkdir()
    command = ["docker", "run", "-d", "--name", name, "--gpus", "all", "--ipc", "host",
               "-p", f"127.0.0.1:{port}:{port}",
               "-e", "PYTHONPATH=/app/src:/app/packages/openpi-client/src",
               "-e", "OPENPI_DATA_HOME=/openpi_assets", "-e", "HF_HUB_OFFLINE=1", "-e", "JAX_PLATFORMS=cpu",
               "-v", str(REPO) + ":/app:ro", "-v", str(MODEL_ROOT) + ":/work:ro",
               "-v", str(policy_output) + ":/out",
               "-v", "/dev/null:/app/gpt-6-control/deepseek_api_key:ro", "-w", "/app",
               "--entrypoint", "/.venv/bin/python", POLICY_IMAGE,
               "/work/versions/trt_t4_20261007_r1/server.py", "--variant", "original_torch",
               "--out", "/out/server", "--port", str(port)]
    write(policy_output / "command.json", command)
    results = []
    started = time.monotonic()
    subprocess.run(command, check=True, stdout=subprocess.PIPE)
    try:
        while True:
            try:
                with urllib.request.urlopen(url + "/health", timeout=2) as response:
                    ready = json.load(response)
                assert ready["variant"] == "original_torch"
                write(policy_output / "ready.json", ready)
                break
            except Exception:
                state = json.loads(subprocess.check_output(["docker", "inspect", name]))[0]["State"]
                if not state["Running"] or time.monotonic() - started > 600:
                    raise RuntimeError("policy startup failed")
                time.sleep(1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda task: run_task(task, args.seed, args.run_id, "planner", url), args.tasks))
        write(output / "task_process_summary.json", results)
    finally:
        try:
            urllib.request.urlopen(urllib.request.Request(url + "/shutdown", data=b"{}"), timeout=5).read()
        except Exception:
            pass
        try:
            subprocess.run(["docker", "wait", name], timeout=30, check=True, stdout=subprocess.PIPE)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "stop", "-t", "5", name], check=True, stdout=subprocess.PIPE)
        state = json.loads(subprocess.check_output(["docker", "inspect", name]))[0]
        (policy_output / "server.log").write_bytes(subprocess.check_output(["docker", "logs", name], stderr=subprocess.STDOUT))
        write(policy_output / "container_exit.json", state["State"])
        write(output / "workflow_summary.json", {"tasks": args.tasks, "seed": args.seed,
              "workers": args.workers, "seconds": time.monotonic() - started,
              "all_task_processes_exit_zero": bool(results) and all(r["exit_code"] == 0 for r in results),
              "policy_exited_zero": state["State"]["ExitCode"] == 0,
              "policy_running_after_cleanup": state["State"]["Running"]})
    return 0 if results and all(r["exit_code"] == 0 for r in results) and state["State"]["ExitCode"] == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--run-id", required=True)
    raise SystemExit(main(parser.parse_args()))
