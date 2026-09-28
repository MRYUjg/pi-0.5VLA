#!/usr/bin/env python3
"""N0 基线指纹冻结：产出 manifests/source_fingerprint.json。

用途：本仓库不是 git 仓库，回退只能依赖 *.bak。本脚本把「9/10 可用版本」的
源码哈希、备份文件、镜像、依赖版本、默认夹爪真实值、GPU 环境固化成一份 JSON，
后续任何改动都能据此判定「是否偏离基线、偏离在哪」。

用法：
    python3 agent-work-space/zhixing_adaptation_20260927/scripts/n0_fingerprint.py
"""
import datetime
import hashlib
import json
import os
import re
import subprocess

REPO = "/mnt/data/vla-policy/openpi"
WORK = os.path.join(REPO, "agent-work-space/zhixing_adaptation_20260927")
OUT = os.path.join(WORK, "manifests/source_fingerprint.json")

# 8 类关键文件：机器人/夹爪/控制器/入口/评测/推理服务/训练配置
TARGETS = {
    "robot_xml": "robosuite/models/assets/robots/ECO65-B/robot.xml",
    "zhixing_gripper_py": "robosuite/models/grippers/zhixing_gripper.py",
    "panda_gripper_py": "robosuite/models/grippers/panda_gripper.py",
    "eco65_robot_py": "robosuite/models/robots/manipulators/eco65_robot.py",
    "mounted_eco65_py": "third_party/libero/libero/libero/envs/robots/mounted_eco65.py",
    "libero_main_py": "examples/libero/main.py",
    "eval_matrix_py": "agent-work-space/scripts/eval_matrix.py",
    "serve_policy_py": "scripts/serve_policy.py",
    "training_config_py": "src/openpi/training/config.py",
}

# 每一次改动对应的备份文件（回退路径）；新增改动必须在此登记
BAK_FILES = [
    "robosuite/models/assets/robots/ECO65-B/robot.xml.pre_contact_fix.bak",
    "robosuite/models/robots/manipulators/eco65_robot.py.pre_init_align.bak",
    "third_party/libero/libero/libero/envs/robots/mounted_eco65.py.pre_init_align.bak",
    "examples/libero/main.py.pre_gripper_entry.bak",
    "agent-work-space/scripts/eval_matrix.py.pre_gripper_entry.bak",
    # N1 修复（开合极性 + init_qpos 自洽性）
    "robosuite/models/grippers/zhixing_gripper.py.pre_polarity_switch.bak",
    "robosuite/models/grippers/zhixing_gripper.py.pre_fix_landing.bak",
    "docs/ECO65_ZhiXing_项目说明.md.pre_n1_correction.bak",
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, timeout=120):
    """best-effort 执行外部命令，失败不中断指纹采集。"""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()[:500]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "stdout": "", "stderr": f"{type(e).__name__}: {e}"}


def parse_property(src, cls_marker, prop):
    """从源码里抽出某个 @property 的 return 值（本仓不是 git 仓库，直接读源码取证）。"""
    try:
        text = open(src, encoding="utf-8-sig").read()
    except Exception:  # noqa: BLE001
        return None
    # 定位 def <prop>(self) 之后的第一个 return
    m = re.search(rf"def\s+{prop}\s*\(self\)\s*:(.*?)(?=\n\s*(?:@|def\s))", text, re.S)
    if not m:
        return None
    body = m.group(1)
    rets = re.findall(r"^\s*return\s+(.+?)\s*$", body, re.M)
    active = [r for r in rets if not r.lstrip().startswith("#")]
    return active[0] if active else (rets[0] if rets else None)


def main():
    fp = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "purpose": "N0 基线冻结：ECO65+PandaGripper 9/10 可用版本的源码/环境指纹",
        "repo": REPO,
        "vcs": "本仓库不是 git 仓库，无 commit 可引；回退依赖 *.bak",
        "sources": {},
        "backups": {},
        "default_gripper_actual": {},
        "init_qpos_actual": {},
        "docker_images": [],
        "gpu": {},
        "policy_server": {},
        "container_versions": {},
    }

    for name, rel in TARGETS.items():
        p = os.path.join(REPO, rel)
        if os.path.isfile(p):
            fp["sources"][name] = {"path": rel, "sha256": sha256(p), "bytes": os.path.getsize(p)}
        else:
            fp["sources"][name] = {"path": rel, "sha256": None, "error": "MISSING"}

    for rel in BAK_FILES:
        p = os.path.join(REPO, rel)
        if os.path.isfile(p):
            fp["backups"][rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p), "present": True}
        else:
            fp["backups"][rel] = {"present": False, "error": "MISSING — 回退路径不完整"}

    # R5：两份文档对 default_gripper 的记载相互矛盾，此处以源码实测为准
    for name in ("eco65_robot_py", "mounted_eco65_py"):
        rel = TARGETS[name]
        fp["default_gripper_actual"][name] = parse_property(os.path.join(REPO, rel), None, "default_gripper")
        fp["init_qpos_actual"][name] = parse_property(os.path.join(REPO, rel), None, "init_qpos")

    r = run("docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'")
    if r["ok"]:
        fp["docker_images"] = r["stdout"].splitlines()

    r = run("nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader")
    if r["ok"]:
        fp["gpu"]["raw"] = r["stdout"].splitlines()
    r = run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    if r["ok"]:
        fp["gpu"]["compute_apps"] = r["stdout"].splitlines()

    r = run("ps -eo pid,etime,cmd | grep -E 'serve_policy' | grep -v grep")
    if r["ok"]:
        fp["policy_server"]["process"] = r["stdout"].splitlines()

    # 容器内依赖版本（best-effort；需要 GPU 预留，失败不影响主指纹）
    probe = (
        "source /.venv/bin/activate 2>/dev/null; "
        "python -c \"import sys,robosuite,mujoco,numpy;"
        "print('python',sys.version.split()[0]);"
        "print('robosuite',robosuite.__version__);"
        "print('mujoco',mujoco.__version__);"
        "print('numpy',numpy.__version__)\""
    )
    r = run(
        "cd %s && docker compose -f examples/libero/compose.yml run --rm --no-deps runtime "
        "/bin/bash -c '%s'" % (REPO, probe),
        timeout=300,
    )
    fp["container_versions"] = {
        "ok": r["ok"],
        "stdout": r["stdout"].splitlines(),
        "stderr": r["stderr"] if not r["ok"] else "",
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fp, f, ensure_ascii=False, indent=2)

    print(f"WROTE {OUT}")
    print("\n=== default_gripper 实测（R5 判定）===")
    for k, v in fp["default_gripper_actual"].items():
        print(f"  {k}: {v}")
    print("\n=== 源码指纹 ===")
    for k, v in fp["sources"].items():
        print(f"  {v['sha256'][:16] if v['sha256'] else 'MISSING':16}  {k}")
    print("\n=== 备份完整性 ===")
    for k, v in fp["backups"].items():
        print(f"  {'OK ' if v['present'] else 'MISSING'}  {k}")


if __name__ == "__main__":
    main()
