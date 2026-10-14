#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""N0/协议（审核记录 §3.3）：把最终验收口径固化成**版本化协议**。

审核记录 §3.3 的原话要点：
  * 用户此前确认的是 **Spatial 10 任务 × 20 次、总成功率 ≥90%**；
  * N4 写"D1 决策"把试采改成了 `libero_object`，而 `阶段进展.md` 仍写 Spatial；
  * "无论变更是否曾获同意，工程验收口径都应统一写入版本化协议"；
  * Object 单任务试采**不能**算作 Spatial 的高成功率目标已推进到相同验收阶段；
  * 当前成功示范 447–459 步，而 Object 评测入口上限只有 280 步；
  * "黑碗外径大于开口"只排除了包住全宽的特定抓法。

本脚本只**登记口径**，不做测量。它把"已确认 / 偏离 / 未决"三类事实分开写清楚，
未决项一律标 `UNRESOLVED` 并给出可选方案，避免用默认值冒充已授权结论。

用法：python3 agent-work-space/zhixing_adaptation_20260927/scripts/make_acceptance_protocol.py
"""
import datetime
import json
import os
import re

REPO = "/mnt/data/vla-policy/openpi"
NODE = os.path.join(REPO, "agent-work-space/zhixing_adaptation_20260927")
OUT_JSON = os.path.join(NODE, "configs/final_acceptance_protocol.json")
OUT_MD = os.path.join(NODE, "configs/final_acceptance_protocol.md")

# 从源码读取评测步数上限，避免手抄漂移
def read_max_steps():
    p = os.path.join(REPO, "agent-work-space/scripts/eval_matrix.py")
    txt = open(p, encoding="utf-8").read()
    m = re.search(r"MAX_STEPS\s*=\s*\{(.*?)\}", txt, re.S)
    out = {}
    if m:
        for k, v in re.findall(r'"(\w+)"\s*:\s*(\d+)', m.group(1)):
            out[k] = int(v)
    return out


def main():
    max_steps = read_max_steps()
    proto = {
        "protocol_version": "1.1",
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "authority_note": (
            "本文件是审核记录 §3.3 要求的**版本化验收协议**。"
            "任何口径变更都必须新建版本号，不得就地改写已发布版本。"),

        "confirmed_scope": {
            "status": "用户已确认（历史）",
            "source": "任务方案.md 目标与验收口径；阶段进展.md 已确认段",
            "suite": "libero_spatial",
            "tasks": 10,
            "rollouts_per_task": 20,
            "total_rollouts": 200,
            "macro_average_success_target": 0.90,
            "reporting_requirements": [
                "逐任务成功率", "失败类型分布", "95% 置信区间",
                "完成步数", "推理延迟", "异常终止数", "Panda 参考"],
            "caveat": (
                "这是**有限测试集上的验收指标**，不是对未知环境真实成功率 ≥90% 的统计保证；"
                "单任务低成功率不得被宏平均掩盖。"),
        },

        "deviation_in_progress": {
            "status": "已裁决（2026-09-28 用户确认）",
            "what": "N4 试采与评测改用了 libero_object 的**单任务（task 0）**",
            "implication": (
                "Object 单任务试采**不能**算作 Spatial 10×20 的验收目标已推进到同一阶段；"
                "它可以作为**链路打通与数据契约验证**的试点，但不能对外表述为"
                "「成功率目标已推进」。"),
            "decision": "RESOLVED",
            "decision_detail": (
                "用户选择 **(A)：最终验收保持 libero_spatial，10 任务 × 20 次 = 200 次，"
                "宏平均 ≥90%**。Object 单任务试采仅作为链路与数据契约的试点，"
                "不计入 Spatial 的验收进度，也不得与 Spatial 数字混比。"),
        },

        "step_limit_conflict": {
            "status": "UNRESOLVED（阻塞跨配置成功率比较）",
            "measured": {
                "teacher_success_frames_observed": [517, 521, 521, 523, 529],
                "note": "N4 已采集的成功示范为 517–529 步（含新增的释放后观察期 70 步）",
            },
            "eval_entry_max_steps": max_steps,
            "problem": (
                "评测入口上限（Object 280 步 / Spatial **220 步**）小于教师示范长度。"
                "这不证明微调后策略必然失败，但说明**教师示范尚未证明能在目标时限内完成**。"),
            "is_cap_hardcoded": (
                "**不是硬约束**：`MAX_STEPS` 只是 "
                "`agent-work-space/scripts/eval_matrix.py:36-42` 与 `examples/libero/main.py:92-100` "
                "里的两个字面量字典，取值沿用官方 LIBERO 约定"
                "（注释写 longest training demo has N steps）。可以改，"
                "但改了就不再与官方协议及历史 Panda 参考同口径。"),
            "teacher_timing_diagnosis": {
                "finding": (
                    "逐 episode 的 `phase_end` 显示：**transport 恒为 220 步、place 恒为 160 步**"
                    "（5/5 完全一致），即两段 `servo()` 都返回了 `max_steps` —— "
                    "**从未满足其 8 mm 容差**，属超时退出而非收敛。"),
                "measured_phase_end_examples": [
                    {"approach": 22, "descend": 18, "close_steps": 10,
                     "transport": 220, "place": 160},
                    {"approach": 21, "descend": 18, "close_steps": 10,
                     "transport": 220, "place": 160},
                    {"approach": 27, "descend": 19, "close_steps": 10,
                     "transport": 220, "place": 160},
                ],
                "implication": (
                    "约 521 帧里有 **380 帧（73%）** 花在两段未收敛的伺服上。"
                    "所以时限冲突的首要原因是**教师控制质量**，而不是任务本身需要 520 步。"),
                "not_yet_isolated": (
                    "尚未隔离「为何不收敛」，候选：① 刻意降低的 gain=0.55（为防惯性滑落）；"
                    "② `robosuite/robots/single_arm.py` 中 joint_kp=3.0 的关节 PD 回拉"
                    "在**带载**时造成稳态误差；③ 目标点不可达。"
                    "需要一次带载稳态误差实测才能定因。"),
            },
            "options": [
                "(A) 缩短教师：先定位上面三个候选原因，让 transport/place 真正收敛、"
                "或在达标后提前退出；预期可把示范压到 220 步以内"
                "（**首选，审核记录也要求「缩短无效等待/优化控制」**）；",
                "(B) 放宽时限：把该 suite 的 max_steps 提高并重新冻结；",
            ],
            "rule": (
                "无论选哪个，**同一时限必须同时施加于全部被比较配置**"
                "（微调前后 ECO65+ZhiXing、ECO65+Panda、Panda 参考），"
                "否则成功率不可比。选定值写入 configs/frozen_protocol.json。"),
        },

        "randomness_control": {
            "init_protocol_must_be_explicit": (
                "同一次比较内固定同一种初始场景口径（random 或 init_state），"
                "并把该口径写入结果 JSON。历史记录显示 `auto→init_state` 复现的是"
                "8/10 协议，而 10/10 阳性对照用的是 random —— 两者不可混比。"),
            "inference_rng": (
                "推理 RNG 随请求历史推进；不得把全部波动都归因于 RNG，"
                "也不得用 RNG 解释跨口径差异。"),
            "seeds": "训练 / 验证 / 测试 seed 分别命名并保存 manifest；重复初始化按对象初态哈希去重。",
        },

        "control_comparisons": {
            "panda_reference": (
                "Panda+PandaGripper 必须用与目标配置**完全相同**的 suite、任务、seed、"
                "时限、动作块长度与相机预处理重跑，不得引用历史数字代替。"),
            "eco65_panda": "作为本体（机械臂）对照，用于把「夹爪因素」与「机械臂因素」分开。",
            "replan_steps": 5,
            "replan_change_rule": "若验证集表明需改为 1/2，视为**新的部署配置**并完整重评。",
            "eval_hygiene": [
                "不使用脚本专家干预", "不读特权物体状态生成动作",
                "不进行中途人工接管", "固定并记录 max_steps 与动作块执行长度"],
        },

        "scope_caveats": {
            "black_bowl_reachability": (
                "「黑碗外径大于开口」只排除了**包住全宽**这一种抓法，"
                "不能证明夹边/侧抓等所有抓法都不可行；单个固定姿态、有限初值 IK 求解失败"
                "也不能证明目标在所有姿态下物理不可达。"
                "相关结论必须限定为「在当前抓取方式 / 求解条件下失败」。"),
            "visual_ood": (
                "「视觉 OOD 是主因、微调是必经之路」尚未被严格隔离证明："
                "后续又发现 TCP、碰撞/力学与控制时序问题。当前结论宜为"
                "「视觉与策略分布适配仍待验证」，需在**最终物理/控制配置**下复评后，"
                "再做保持其他条件一致的视觉消融或自数据微调实验。"),
        },

        "n5_entry_conditions": [
            "N1–N3 在当前资产/控制器/速度下重验通过（开合时序、TCP/相机、平移+旋转、全程 OSC 抓放）",
            "N4 数据链路返工通过：首帧新鲜、state 关节正确、动作尺度元数据正确、"
            "回放严格恢复且失败接入验收",
            "≥5 条不同初始化的真实成功示范，逐条回放通过",
            "suite（Spatial 或 Object）、任务覆盖、每任务测试数、时限、随机性控制、"
            "Panda 对照**全部写入本协议**",
        ],
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(proto, f, ensure_ascii=False, indent=2)

    md = ["# 最终验收协议（版本化）", "",
          "版本 **%s**，生成于 %s。" % (proto["protocol_version"], proto["generated_at"]), "",
          "> %s" % proto["authority_note"], "",
          "## 1. 已确认的验收范围", "",
          "| 项 | 值 |", "|---|---|",
          "| suite | `%s` |" % proto["confirmed_scope"]["suite"],
          "| 任务数 | %d |" % proto["confirmed_scope"]["tasks"],
          "| 每任务测试次数 | %d |" % proto["confirmed_scope"]["rollouts_per_task"],
          "| 总测试次数 | **%d** |" % proto["confirmed_scope"]["total_rollouts"],
          "| 宏平均成功率目标 | **≥%.0f%%** |" % (proto["confirmed_scope"]["macro_average_success_target"] * 100),
          "", "必报项：%s。" % "、".join(proto["confirmed_scope"]["reporting_requirements"]),
          "", "> %s" % proto["confirmed_scope"]["caveat"], "",
          "## 2. 当前偏离（未获书面确认）", "",
          "- **事实**：%s" % proto["deviation_in_progress"]["what"],
          "- **影响**：%s" % proto["deviation_in_progress"]["implication"],
          "- **裁决**：`%s`" % proto["deviation_in_progress"]["decision"],
          "- %s" % proto["deviation_in_progress"]["decision_detail"], "",
          "## 3. 时限冲突（阻塞跨配置比较）", "",
          "- 教师成功示范实测帧数：%s" % proto["step_limit_conflict"]["measured"]["teacher_success_frames_observed"],
          "- 评测入口上限：`%s`（libero_object=%d，libero_spatial=%d）"
          % (json.dumps(max_steps), max_steps.get("libero_object", -1), max_steps.get("libero_spatial", -1)),
          "- **这个上限是不是写死的**：%s" % proto["step_limit_conflict"]["is_cap_hardcoded"],
          "- **问题**：%s" % proto["step_limit_conflict"]["problem"], "",
          "### 教师耗时诊断（决定该不该动上限）", "",
          "- %s" % proto["step_limit_conflict"]["teacher_timing_diagnosis"]["finding"],
          "- %s" % proto["step_limit_conflict"]["teacher_timing_diagnosis"]["implication"],
          "- **尚未隔离**：%s" % proto["step_limit_conflict"]["teacher_timing_diagnosis"]["not_yet_isolated"], "",
          "可选方案：", ""]
    for o in proto["step_limit_conflict"]["options"]:
        md.append("- %s" % o)
    md += ["", "> **规则**：%s" % proto["step_limit_conflict"]["rule"], "",
           "## 4. 随机性与对照", "",
           "- %s" % proto["randomness_control"]["init_protocol_must_be_explicit"],
           "- %s" % proto["randomness_control"]["inference_rng"],
           "- %s" % proto["randomness_control"]["seeds"],
           "- Panda 参考：%s" % proto["control_comparisons"]["panda_reference"],
           "- `replan_steps = %d`；%s" % (proto["control_comparisons"]["replan_steps"],
                                          proto["control_comparisons"]["replan_change_rule"]),
           "- 评测纪律：%s" % "；".join(proto["control_comparisons"]["eval_hygiene"]), "",
           "## 5. 结论边界（必须随结果一起引用）", "",
           "- **黑碗可达性**：%s" % proto["scope_caveats"]["black_bowl_reachability"],
           "- **视觉 OOD**：%s" % proto["scope_caveats"]["visual_ood"], "",
           "## 6. 进入 N5 的条件", ""]
    for c in proto["n5_entry_conditions"]:
        md.append("- [ ] %s" % c)
    md.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print("WROTE %s" % OUT_JSON)
    print("WROTE %s" % OUT_MD)
    print("\n已确认: %s %d×%d = %d 次, 宏平均 ≥%.0f%%"
          % (proto["confirmed_scope"]["suite"], proto["confirmed_scope"]["tasks"],
             proto["confirmed_scope"]["rollouts_per_task"],
             proto["confirmed_scope"]["total_rollouts"],
             proto["confirmed_scope"]["macro_average_success_target"] * 100))
    print("未决  : suite 决策=%s ；时限冲突=%s（教师 %s 步 vs Object 出口上限 %d / Spatial %d 步）"
          % (proto["deviation_in_progress"]["decision"],
             proto["step_limit_conflict"]["status"],
             proto["step_limit_conflict"]["measured"]["teacher_success_frames_observed"],
             max_steps.get("libero_object", -1), max_steps.get("libero_spatial", -1)))


if __name__ == "__main__":
    main()
