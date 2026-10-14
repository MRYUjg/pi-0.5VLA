#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""措施 B2：把运行时参数收敛到**一个**冻结入口（复核 §5）。

复核 §5.1 的问题：N3/N4 显式用 `speed=0.01`，但 `ZhixingGripper` 类默认仍是 **0.2**，
任何未显式覆盖的评测/部署入口都会静默使用 0.2；N1 的 speed=0.2 消融里最终 tip 间距
39.09 mm、0.01 时 22.93 mm —— 两者**不是同一个物理配置**。

§5 的措施要求："用一个配置/环境工厂贯穿采集、回放、策略评测；保存运行时真实参数与
哈希并断言一致。"

本模块就是那个入口：
  * `apply()`      —— 在任何 env 构建**之前**调用，把冻结参数写进环境变量；
  * `effective()`  —— 返回当前**真实生效**的参数（从运行对象读，不读声明值）；
  * `fingerprint()`—— 生效参数 + 关键资产哈希，写进 episode 元数据/审计结果，
                      供事后判定"这条数据是哪套配置跑出来的"。
"""
import hashlib
import os
import pathlib

REPO = pathlib.Path("/app")
NODE = REPO / "agent-work-space" / "zhixing_adaptation_20260927"

# --------------------------------------------------------------------------- #
# 冻结的部署参数（Deployment-freeze）。改这里必须同时升版本号并重跑受影响节点。
# --------------------------------------------------------------------------- #
FREEZE_VERSION = "b2-1"
DEPLOY_GRIPPER_SPEED = "0.01"        # 部署速度；资产原值 0.2 会导致 ~2 步内闭死
# 额外关节回拉 PD：`legacy`(3.0) = 历史行为；`eco65_no_extra_pd`(0.0) =
# 措施 B3 已验证的候选配置。**默认仍是 legacy**，把候选改为默认属措施 C/D 的决定，
# 必须先在同一版本下重跑完整 N3 全矩阵。
DEPLOY_EXTRA_JOINT_PD_PROFILE = "legacy"

ASSETS = {
    "zhixing_gripper_xml": "robosuite/models/assets/grippers/zhixing_gripper.xml",
    "zhixing_gripper_py": "robosuite/models/grippers/zhixing_gripper.py",
    "eco65_robot_xml": "robosuite/models/assets/robots/ECO65-B/robot.xml",
    "eco65_robot_py": "robosuite/models/robots/manipulators/eco65_robot.py",
    "mounted_eco65_py": "third_party/libero/libero/libero/envs/robots/mounted_eco65.py",
    "on_the_ground_eco65_py": "third_party/libero/libero/libero/envs/robots/on_the_ground_eco65.py",
    "robot_model_py": "robosuite/models/robots/robot_model.py",
    "single_arm_py": "robosuite/robots/single_arm.py",
    "n4_collect_py": "agent-work-space/zhixing_adaptation_20260927/scripts/n4_collect.py",
    "n4_audit_py": "agent-work-space/zhixing_adaptation_20260927/scripts/n4_audit.py",
}


def apply(gripper_speed=None, extra_joint_pd_profile=None, force=False):
    """把冻结参数写进环境变量。已显式设置的环境变量默认**不覆盖**（除非 force）。"""
    if force or not os.environ.get("ZHIXING_GRIPPER_SPEED"):
        os.environ["ZHIXING_GRIPPER_SPEED"] = str(
            gripper_speed if gripper_speed is not None else DEPLOY_GRIPPER_SPEED)
    if force or not os.environ.get("EXTRA_JOINT_PD_PROFILE"):
        os.environ["EXTRA_JOINT_PD_PROFILE"] = str(
            extra_joint_pd_profile if extra_joint_pd_profile is not None
            else DEPLOY_EXTRA_JOINT_PD_PROFILE)
    return effective()


def effective(env=None):
    """当前**真实生效**的参数。传入 env 时从运行对象读夹爪速度（推荐）。"""
    speed = None
    if env is not None:
        try:
            speed = float(env.env.robots[0].gripper.speed)
        except Exception:                                 # noqa: BLE001
            speed = None
    if speed is None:
        speed = float(os.environ.get("ZHIXING_GRIPPER_SPEED", "0.2"))
    return {
        "freeze_version": FREEZE_VERSION,
        "gripper_speed": speed,
        "gripper_speed_from_env": os.environ.get("ZHIXING_GRIPPER_SPEED"),
        "extra_joint_pd_profile": os.environ.get("EXTRA_JOINT_PD_PROFILE", "legacy"),
        "extra_joint_pd_kp_override": os.environ.get("EXTRA_JOINT_PD_KP") or None,
    }


def assert_matches_freeze(env=None, gripper_speed=None):
    """断言运行参数与冻结值一致；不一致就抛错（不允许静默跑错配置）。"""
    want = float(gripper_speed if gripper_speed is not None else DEPLOY_GRIPPER_SPEED)
    got = effective(env)["gripper_speed"]
    if abs(got - want) > 1e-12:
        raise RuntimeError(
            "[B2] 夹爪速度与冻结值不一致：生效 %.6f，冻结 %.6f。"
            "所有正式入口必须在建 env 之前调用 runtime_config.apply()。"
            % (got, want))
    return True


def _sha16(p):
    try:
        return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
    except Exception:                                     # noqa: BLE001
        return None


def fingerprint(env=None):
    """生效参数 + 关键资产哈希；用于把数据/结果绑定到具体配置版本。"""
    fp = effective(env)
    fp["assets_sha16"] = {k: _sha16(REPO / v) for k, v in ASSETS.items()}
    return fp
