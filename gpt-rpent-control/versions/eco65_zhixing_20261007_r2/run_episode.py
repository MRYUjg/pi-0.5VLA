"""Run RPent's actual API planner and LIBERO toolkit against project adapters."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import traceback

import httpx
import numpy as np
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from robots.libero.env_client import LiberoEnvClient
from robots.libero.toolkit import LiberoToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory.manager import MemoryManager
from rpent.planner.api_loop import ApiAgentLoop
from rpent.robots.components.pi05_vla_client import Pi05VLAClient
from rpent.utils.daemon import ProcessDaemon, pick_free_port
from rpent.utils.logging import init_output_dir
from rpent.utils.rpc import RpcError, make_rpc_client, wait_for_ready

VERSION = Path(__file__).resolve().parent


class EmbodimentToolkit(LiberoToolkit):
    """Expose native VLA skills and feedback for the first policy baseline."""

    def _register_libero_tools(self) -> None:
        super()._register_libero_tools()
        allowed = {"view_env_state", "pi0_pick", "pi0_doubled", "finish"}
        self._tools = {name: item for name, item in self._tools.items() if name in allowed}


def run(args) -> int:
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    source_lock = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in VERSION.iterdir() if p.suffix in {".py", ".txt", ".json", ".md"}}
    (output / "source_lock.json").write_text(json.dumps(source_lock, indent=2))
    (output / "executed_source").mkdir()
    for name in source_lock:
        shutil.copy2(VERSION / name, output / "executed_source" / name)
    init_output_dir(output / "rpent")
    events = NullDashboardEventSink()
    daemons, clients = {}, {}
    toolkit = None
    result = dict(task=args.task, seed=args.seed, suite="libero_spatial", mode=args.mode,
                  native_success=False, runtime_chain_passed=False, planner_error=None)
    # Credential arrives once through stdin, never argv/environment/a bind mount.
    key = sys.stdin.readline().strip() if args.mode == "planner" else None
    if args.mode == "planner" and not key:
        raise RuntimeError("planner credential was not supplied on stdin")
    api_log = (output / "api_responses.jsonl").open("x", buffering=1)

    async def record_response(response: httpx.Response) -> None:
        await response.aread()
        if response.request.url.host != "api.deepseek.com":
            return
        if "text/event-stream" in response.headers.get("content-type", ""):
            chunks = [json.loads(line[5:].strip()) for line in response.text.splitlines()
                      if line.startswith("data:") and line[5:].strip() not in {"", "[DONE]"}]
            body = {"id": next((c.get("id") for c in chunks if c.get("id")), None),
                    "model": next((c.get("model") for c in chunks if c.get("model")), None),
                    "usage": next((c.get("usage") for c in reversed(chunks) if c.get("usage")), None),
                    "choices": [c.get("choices") for c in chunks]}
        else:
            body = response.json()
        api_log.write(json.dumps({"http_status": response.status_code, "id": body.get("id"),
                                 "model": body.get("model"), "usage": body.get("usage"),
                                 "choices": body.get("choices")}, ensure_ascii=False) + "\n")

    try:
        for component, script, extra in [
            ("env", "env_server.py", ["--task", str(args.task), "--seed", str(args.seed)]),
            ("vla", "vla_bridge.py", ["--policy-url", args.policy_url,
                                      "--seed", str(args.seed * 10 + args.task)])]:
            port = pick_free_port()
            command = [sys.executable, str(VERSION / script), "--port", str(port),
                       "--output", str(output / component), "--parent-watch", *extra]
            daemon = ProcessDaemon(component, command, log_path=str(output / (component + ".log")),
                                   cwd=str(VERSION))
            daemon.start()
            daemons[component] = daemon
            client = make_rpc_client(f"http://127.0.0.1:{port}")
            clients[component] = client
            wait_for_ready(client, daemon=daemon, timeout_s=180)
        env = LiberoEnvClient(clients["env"], expected_meta={"suite": "libero_spatial",
                              "task": args.task, "seed": args.seed, "max_episode_steps": 220,
                              "robot": "ECO65", "gripper": "ZhixingGripper", "action_dim": 7},
                              return_all_frames=True)
        vla = Pi05VLAClient(clients["vla"], embodiment="libero")
        toolkit = EmbodimentToolkit(runtime_kwargs={"env": env, "model": vla, "sam3_client": None},
                               dashboard_events=events, memory=MemoryManager(output / "memory"),
                               state_output_dir=output / "artifacts")
        before = clients["env"].call("env.summary")
        guards = {}
        for name, method, arguments in [
            ("invalid_dimension_rejected", "env.step", ([0.0, 0.0],)),
            ("oversized_chunk_rejected", "env.chunk_step", (np.zeros((6, 7)),)),
            ("nonfinite_action_rejected", "env.step", ([float("nan")] * 7,))]:
            try:
                clients["env"].call(method, args=arguments)
            except RpcError:
                guards[name] = True
            else:
                guards[name] = False
        after = clients["env"].call("env.summary")
        guards["rejected_calls_do_not_advance_physics"] = before == after
        raw = env.raw_obs()
        guards["policy_RGB_orientation_exact"] = np.array_equal(env.last_obs["main_images"], raw["agentview_image"][::-1, ::-1])
        guards["policy_wrist_orientation_exact"] = np.array_equal(env.last_obs["wrist_images"], raw["robot0_eye_in_hand_image"][::-1, ::-1])
        image, depth = env.render_camera("agentview", height=256, width=256, depth=True)
        guards["RPC_camera_matches_raw_observation"] = np.array_equal(image, raw["agentview_image"])
        guards["RGBD_shape_and_finite"] = image.shape == (256, 256, 3) and depth.shape[:2] == (256, 256) and bool(np.isfinite(depth).all())
        (output / "component_checks.json").write_text(json.dumps(guards, indent=2))
        if not all(guards.values()):
            raise RuntimeError("real component guard/observation checks failed")
        (output / "tool_schemas.json").write_text(json.dumps(toolkit.get_tools_spec(), indent=2))
        initial = toolkit.execute_tool("view_env_state", {})
        # The API planner itself strips image bytes in no_images mode.
        status = {k: v for k, v in initial.result.items() if not k.startswith("_image")}
        if args.mode == "chain":
            skill = toolkit.execute_tool("pi0_doubled", {"prompt": env.get_task_language(), "max_chunks": 2})
            (output / "chain_tool_result.json").write_text(json.dumps(
                {k: v for k, v in skill.result.items() if not k.startswith("_image")}, indent=2))
            completed = toolkit.execute_tool("finish", {"status": "success" if env.terminated else "failure",
                                                       "summary": "Bounded real policy-chain diagnostic completed."})
            result["finish"] = completed.result
        else:
            client = httpx.AsyncClient(event_hooks={"response": [record_response]}, timeout=180)
            provider = OpenAIProvider(base_url="https://api.deepseek.com", api_key=key, http_client=client)
            key = None
            model = OpenAIChatModel("deepseek-flash", provider=provider,
                                    settings={"temperature": 0, "extra_body": {"thinking": {"type": "disabled"}}})
            planner = ApiAgentLoop(model=model, dashboard_events=events, max_tokens=2048,
                                   no_images=True, timeout_s=900, reasoning_effort="none")
            outcome = planner.solve(system_prompt=(VERSION / "planner_prompt.txt").read_text(),
                                    user_message=json.dumps(status, ensure_ascii=False),
                                    toolkit=toolkit, max_turns=8)
            result.update(finish=outcome.finish_result, planner_stats=outcome.stats,
                          planner_error=outcome.error, planner_model="deepseek-flash",
                          modality="text_tools_no_images")
            (output / "conversation.json").write_text(json.dumps(outcome.messages, ensure_ascii=False, indent=2))
        actual = clients["env"].call("env.summary")
        result.update(actual)
        result["runtime_chain_passed"] = bool(actual["steps"] > 0 and result.get("finish") is not None
                                               and not result["planner_error"])
        result["toolkit_solved"] = toolkit.solved()
        result["finish_claim_matches_native"] = ((result.get("finish") or {}).get("status") == "success") == actual["native_success"]
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        # Adapters do not carry secrets; mask any occurrence before recording diagnostics.
        detail = traceback.format_exc()
        if key:
            detail = detail.replace(key, "[redacted]")
        (output / "exception.txt").write_text(detail)
        result["runtime_chain_passed"] = False
    finally:
        if toolkit is not None:
            toolkit.close()
        for client in clients.values():
            client.close()
        for daemon in reversed(list(daemons.values())):
            daemon.stop()
        result["component_exit_codes"] = {k: d.poll() for k, d in daemons.items()}
        result["owned_components_exited_cleanly"] = len(daemons) == 2 and all(d.poll() == 0 for d in daemons.values())
        api_log.close()
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["runtime_chain_passed"] and result["owned_components_exited_cleanly"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy-url", default="http://127.0.0.1:18981")
    parser.add_argument("--mode", choices=["chain", "planner"], default="planner")
    sys.exit(run(parser.parse_args()))
