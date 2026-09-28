#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N0（返工）：产出**当前候选版**的独立指纹与冻结参数。

审核记录 §3.1 的问题
--------------------
* 旧 manifest 的 `sources` 与当前源码不符（重跑原验收脚本得 N0: 5/6，
  `eco65_robot_py` / `mounted_eco65_py` 哈希不匹配），因为 R1 又改过这两个文件；
* 旧 manifest **不包含**关键 `zhixing_gripper.xml`、地面机器人类、OSC 配置
  与采集/回放脚本；
* `container_versions.ok=false`（嵌套引号 bug）；
* `configs/` 为空，"参数冻结"没有落地。

本脚本因此：
  1. **不覆盖**历史 `source_fingerprint*.json`，另写
     `manifests/source_fingerprint_current_candidate.json`；
  2. 把资产/控制器/采集/回放脚本一并纳入指纹；
  3. 用独立的 `dep_probe.sh` 取容器依赖版本；
  4. 把最终验收参数冻结到 `configs/frozen_protocol.json`。

用法（仓库根目录）：
    python3 agent-work-space/zhixing_adaptation_20260927/scripts/n0_fingerprint_current.py
"""
import datetime
import hashlib
import json
import os
import re
import subprocess

REPO = "/mnt/data/vla-policy/openpi"
WORK = os.path.join(REPO, "agent-work-space/zhixing_adaptation_20260927")
OUT = os.path.join(WORK, "manifests/source_fingerprint_current_candidate.json")
CFG = os.path.join(WORK, "configs/frozen_protocol.json")

# 当前候选版的完整资产/代码集（比旧版多：夹爪 XML、地面机器人类、OSC 配置、
# R1 修复涉及的 robosuite 基类、采集/回放/审计脚本）
TARGETS = {
    # --- 机器人 / 夹爪 资产与模型 ---
    "robot_xml": "robosuite/models/assets/robots/ECO65-B/robot.xml",
    "zhixing_gripper_xml": "robosuite/models/assets/grippers/zhixing_gripper.xml",
    "zhixing_gripper_py": "robosuite/models/grippers/zhixing_gripper.py",
    "panda_gripper_py": "robosuite/models/grippers/panda_gripper.py",
    "eco65_robot_py": "robosuite/models/robots/manipulators/eco65_robot.py",
    "mounted_eco65_py": "third_party/libero/libero/libero/envs/robots/mounted_eco65.py",
    "on_the_ground_eco65_py": "third_party/libero/libero/libero/envs/robots/on_the_ground_eco65.py",
    "libero_robots_init_py": "third_party/libero/libero/libero/envs/robots/__init__.py",
    # --- R1 修复涉及的 robosuite 基类 ---
    "robosuite_robot_py": "robosuite/robots/robot.py",
    "robosuite_single_arm_py": "robosuite/robots/single_arm.py",
    # --- 控制器配置 ---
    "osc_pose_json": "robosuite/controllers/config/osc_pose.json",
    # --- 入口 / 推理 / 训练 ---
    "libero_main_py": "examples/libero/main.py",
    "eval_matrix_py": "agent-work-space/scripts/eval_matrix.py",
    "serve_policy_py": "scripts/serve_policy.py",
    "training_config_py": "src/openpi/training/config.py",
    # --- N4 采集 / 回放 / 审计 链路 ---
    "n4_collect_py": "agent-work-space/zhixing_adaptation_20260927/scripts/n4_collect.py",
    "n4_audit_py": "agent-work-space/zhixing_adaptation_20260927/scripts/n4_audit.py",
    "run_n4_collect_sh": "agent-work-space/zhixing_adaptation_20260927/scripts/run_n4_collect.sh",
    "clean_n4_data_sh": "agent-work-space/zhixing_adaptation_20260927/scripts/clean_n4_data.sh",
    "verify_r1_gripper_init_py": "agent-work-space/zhixing_adaptation_20260927/scripts/verify_r1_gripper_init.py",
    "dep_probe_sh": "agent-work-space/zhixing_adaptation_20260927/scripts/dep_probe.sh",
    # --- N2 契约重建 与 R7/N3 全程 OSC 验收 ---
    "make_tcp_contract_v2_py": "agent-work-space/zhixing_adaptation_20260927/scripts/make_tcp_contract_v2.py",
    "n2v2_settled_probe_py": "agent-work-space/zhixing_adaptation_20260927/scripts/n2v2_settled_probe.py",
    "n3_osc_full_chain_py": "agent-work-space/zhixing_adaptation_20260927/scripts/n3_osc_full_chain.py",
    "make_acceptance_protocol_py": "agent-work-space/zhixing_adaptation_20260927/scripts/make_acceptance_protocol.py",
}

# 回退路径：每条改动都必须有对应 .bak 才算"可恢复资产集"完整
BAK_FILES = [
    "robosuite/models/assets/robots/ECO65-B/robot.xml.pre_contact_fix.bak",
    "robosuite/models/robots/manipulators/eco65_robot.py.pre_init_align.bak",
    "third_party/libero/libero/libero/envs/robots/mounted_eco65.py.pre_init_align.bak",
    "examples/libero/main.py.pre_gripper_entry.bak",
    "agent-work-space/scripts/eval_matrix.py.pre_gripper_entry.bak",
    "robosuite/models/grippers/zhixing_gripper.py.pre_polarity_switch.bak",
    "robosuite/models/grippers/zhixing_gripper.py.pre_fix_landing.bak",
    "docs/ECO65_ZhiXing_项目说明.md.pre_n1_correction.bak",
    # R1 改动前先备份（本脚本只登记；缺哪个就说明回退路径不完整）
    "robosuite/models/robots/manipulators/eco65_robot.py.pre_r1_gripper_qpos.bak",
    "third_party/libero/libero/libero/envs/robots/mounted_eco65.py.pre_r1_gripper_qpos.bak",
    "robosuite/robots/robot.py.pre_r1_gripper_qpos.bak",
    "robosuite/robots/single_arm.py.pre_r1_gripper_qpos.bak",
    "agent-work-space/zhixing_adaptation_20260927/scripts/run_n4_collect.sh.pre_r5_guard.bak",
]

# 审核时点的修订版本（**只能按哈希识别，无法逐字节还原**）。
# 原因：本仓库不是 git 仓库，R2/R3/R4/R6 是对脚本的就地修改；当时未先做 .bak。
# 这是审核记录已指出的"无版本控制"风险的具体后果，此处如实登记而不是伪造备份。
# sha256 取自 review_20260928/review_source_hashes.json。
REVIEWED_REVISIONS_NOT_RESTORABLE = {
    "agent-work-space/zhixing_adaptation_20260927/scripts/n4_collect.py": {
        "reviewed_sha256": "ec54546a729dbac0853a2ba778895eb2576e9f542cc1e2aa150744832290b594",
        "reviewed_bytes": 34705,
        "changed_by": ["R2 首帧观测", "R4 state 关节名", "R6 动作尺度元数据",
                       "R3 夹爪累积状态落盘"],
        "rollback_note": "无字节备份；回退需按上述四项反向修改，或从本文件之前的"
                         "任何完整仓库副本恢复。当前版本已通过 n4_audit 全项契约检查。",
    },
    "agent-work-space/zhixing_adaptation_20260927/scripts/n4_audit.py": {
        "reviewed_sha256": "3f4d7ed7a26078a6e4083536ab98a22894c66fc068e22ada085970968b5080fe",
        "reviewed_bytes": 14489,
        "changed_by": ["R3 回放严格恢复与硬判据", "配对判别力自检"],
        "rollback_note": "同上。",
    },
}

FROZEN = {
    "note": "N0 冻结参数：后续任何偏离都必须新建版本，不得就地修改本文件所描述的口径",
    "suite": "libero_object",
    "task_id": 0,
    "collect": {
        "control_hz": 20,
        "control_timestep_s": 0.05,
        "image_res": 256,
        "eval_resize": 224,
        "image_flip": "180 ([::-1,::-1])，与 examples/libero/main.py 一致",
        "gripper_speed": 0.01,
        "gripper_speed_asset_default": 0.2,
        "teacher_gain_pos_m": 0.02,
        "teacher_gain_rot_rad": 0.05,
        "prep_hold_steps": 45,
        "action_space": "OSC_POSE 7 维，归一化 [-1,1]",
        "controller_output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
        "max_frames": 900,
    },
    "state_contract": {
        "dims": 8,
        "layout": ["grip_site_x", "grip_site_y", "grip_site_z",
                   "axisangle_x", "axisangle_y", "axisangle_z",
                   "Left_1_Joint", "Right_1_Joint"],
        "gripper_joint_selection": "按 joint name 后缀定位主动关节（禁止硬编码索引）",
    },
    "gripper": {
        "name": "ZhixingGripper",
        "dof": 1,
        "joint_count": 6,
        "polarity": "-1 = 张开，+1 = 闭合（已在 format_action 中翻转以对齐约定）",
        "tcp_eef_pos_m": 0.20254,
    },
    "replay_tolerances": {
        "arm_free_space_rad": 0.01,
        "gripper_free_space_rad": 0.01,
        "render_vs_stored_255": 3.0,
        "obs_state_median_m": 1e-4,
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd, timeout=300):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"ok": p.returncode == 0, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()[:500]}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "stdout": "", "stderr": "%s: %s" % (type(e).__name__, e)}


def parse_property(src, prop):
    """从源码抽某个 @property 的 return（本仓无 git，直接读源码取证）。"""
    try:
        text = open(src, encoding="utf-8-sig").read()
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"def\s+%s\s*\(self\)\s*:(.*?)(?=\n\s*(?:@|def\s))" % prop, text, re.S)
    if not m:
        return None
    rets = re.findall(r"^\s*return\s+(.+?)\s*$", m.group(1), re.M)
    active = [r for r in rets if not r.lstrip().startswith("#")]
    return active[0] if active else (rets[0] if rets else None)


def main():
    fp = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "purpose": "N0（返工）当前候选版指纹：ECO65 + ZhiXing 在 R1–R6 修复后的完整资产/代码/环境快照",
        "supersedes_note": ("历史 source_fingerprint.json / _pre_switch / _switch_era 作为"
                            "历史证据保留，本文件是**当前候选版**，不代表仍是同一个基线"),
        "repo": REPO,
        "vcs": "本仓库不是 git 仓库，无 commit 可引；回退依赖 *.bak 与本指纹",
        "sources": {}, "backups": {}, "default_gripper_actual": {},
        "init_qpos_actual": {}, "docker_images": [], "gpu": {},
        "policy_server": {}, "container_versions": {},
    }

    for name, rel in TARGETS.items():
        p = os.path.join(REPO, rel)
        if os.path.isfile(p):
            fp["sources"][name] = {"path": rel, "sha256": sha256(p),
                                   "bytes": os.path.getsize(p)}
        else:
            fp["sources"][name] = {"path": rel, "sha256": None, "error": "MISSING"}

    missing_bak = []
    for rel in BAK_FILES:
        p = os.path.join(REPO, rel)
        if os.path.isfile(p):
            fp["backups"][rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p),
                                  "present": True}
        else:
            fp["backups"][rel] = {"present": False, "error": "MISSING"}
            missing_bak.append(rel)
    fp["recovery_set_complete"] = not missing_bak
    fp["missing_backups"] = missing_bak
    fp["reviewed_revisions_not_restorable"] = REVIEWED_REVISIONS_NOT_RESTORABLE
    fp["recovery_set_caveat"] = (
        "生产资产（robot.xml / 夹爪 XML / 两个机器人类 / robosuite 基类 / 采集入口）"
        "均有逐字节 .bak 可回退；两个分析脚本（n4_collect.py / n4_audit.py）"
        "只能按审核时点的 sha256 识别，无法逐字节还原 —— 见 "
        "reviewed_revisions_not_restorable。")

    for name in ("eco65_robot_py", "mounted_eco65_py"):
        rel = TARGETS[name]
        src = os.path.join(REPO, rel)
        fp["default_gripper_actual"][name] = parse_property(src, "default_gripper")
        fp["init_qpos_actual"][name] = parse_property(src, "init_qpos")

    r = run("docker images --format '{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}'")
    if r["ok"]:
        fp["docker_images"] = r["stdout"].splitlines()
    r = run("nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,"
            "utilization.gpu --format=csv,noheader")
    if r["ok"]:
        fp["gpu"]["raw"] = r["stdout"].splitlines()
    r = run("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader")
    if r["ok"]:
        fp["gpu"]["compute_apps"] = r["stdout"].splitlines()
    r = run("ps -eo pid,etime,cmd | grep -E 'serve_policy' | grep -v grep")
    fp["policy_server"]["process"] = r["stdout"].splitlines() if r["ok"] else []

    # 容器依赖版本：走独立脚本，避免嵌套引号 bug（§3.1）
    r = run("cd %s && docker compose -f examples/libero/compose.yml run --rm --no-deps "
            "runtime /bin/bash /app/agent-work-space/zhixing_adaptation_20260927/scripts/dep_probe.sh"
            % REPO, timeout=420)
    deps = {}
    for line in r["stdout"].splitlines():
        if line.startswith("DEP_PROBE_JSON "):
            try:
                deps = json.loads(line[len("DEP_PROBE_JSON "):])
            except Exception:                           # noqa: BLE001
                deps = {}
    fp["container_versions"] = {"ok": bool(deps), "parsed": deps,
                                "stderr": r["stderr"] if not deps else ""}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(fp, f, ensure_ascii=False, indent=2)

    cfg = dict(FROZEN)
    cfg["generated_at"] = fp["generated_at"]
    cfg["fingerprint_file"] = os.path.relpath(OUT, REPO)
    cfg["fingerprint_sha256_of_itself_omitted"] = True
    os.makedirs(os.path.dirname(CFG), exist_ok=True)
    with open(CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("WROTE %s" % OUT)
    print("WROTE %s" % CFG)
    print("\n=== default_gripper 实测 ===")
    for k, v in fp["default_gripper_actual"].items():
        print("  %-18s %s" % (k, v))
    print("\n=== init_qpos 实测 ===")
    for k, v in fp["init_qpos_actual"].items():
        print("  %-18s %s" % (k, v))
    print("\n=== 源码指纹 ===")
    for k, v in fp["sources"].items():
        print("  %-18s %s" % (v["sha256"][:16] if v["sha256"] else "MISSING", k))
    print("\n=== 容器依赖 ===")
    print("  ok=%s  %s" % (fp["container_versions"]["ok"], fp["container_versions"]["parsed"]))
    print("\n=== 回退资产集完整性 ===")
    print("  complete=%s  缺失=%d" % (fp["recovery_set_complete"], len(missing_bak)))
    for m in missing_bak:
        print("    MISSING %s" % m)


if __name__ == "__main__":
    main()
