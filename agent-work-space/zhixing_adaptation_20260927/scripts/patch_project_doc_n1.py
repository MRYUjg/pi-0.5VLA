#!/usr/bin/env python3
"""把 docs/ECO65_ZhiXing_项目说明.md 里已被 N1 实测推翻的两处结论更新掉。

  §6.3 动作映射：原文称 u=-1 打开、u=+1 闭合 —— 实测相反。
  §6.5 初始化：原文把 init_qpos 广播列为「待核验」—— 已核实并修复。

幂等：已更新过则跳过。
"""
import hashlib
import shutil
import sys

DOC = "/mnt/data/vla-policy/openpi/docs/ECO65_ZhiXing_项目说明.md"
BAK = DOC + ".pre_n1_correction.bak"

OLD_63 = ("按代码约定，`u=-1` 对应 `q_target=-0.91`，即打开；`u=+1` 对应 `q_target=0`，"
          "即闭合。零输入保持当前累加目标。非零输入主要提供方向，其绝对值大小被 `sign()` 消去。")

NEW_63 = (
    "**（2026-09-27 实测修正）** `q_target = -0.455 + 0.455 × u_next` 这一步与原文一致："
    "`u=-1` → `q_target=-0.91`、`u=+1` → `q_target=0`。但**原文把物理效果写反了**：\n\n"
    "| u | q_target | 实测指尖开口 | 实际动作 |\n|---|---|---|---|\n"
    "| `+1` | `0.00` | **11.26 cm** | **张开** |\n"
    "| `-1` | `-0.91` | **2.28 cm** | **闭合（近闭）** |\n\n"
    "即 `u=+1` 实际**张开**、`u=-1` 实际**闭合**，与 robosuite/LIBERO「+1=闭合」的统一约定相反"
    "（PandaGripper 在同一机器人、同一协议下实测 `+1` → 0.10 cm 闭合、`-1` → 7.94 cm 张开）。\n\n"
    "该极性反转已在 `ZhixingGripper.format_action` 中修正（累积项取负），"
    "修正后 `+1=闭合`、`-1=张开`。证据："
    "`agent-work-space/zhixing_adaptation_20260927/results/gripper_polarity.json`。"
    "零输入保持当前累加目标；非零输入主要提供方向，其绝对值大小被 `sign()` 消去。")

OLD_65 = ("这一初始状态并不直接满足全部主从关节的正负关系，仿真约束求解器随后需要消除偏差。"
          "这是静态代码可见的初始化一致性问题，其是否造成明显跳动或接触异常需要运行验证。")

NEW_65 = (
    "**（2026-09-27 已核实并修复）** 运行验证确认这**不是无害的**：XML 的 5 条 joint equality 要求 "
    "Left_Support / Right_1 / Right_2 / Right_Support = -Left_1 = **+0.91**，"
    "因此 4 个关节符号错误，**最大 equality 残差 1.82 rad（≈104°）**，"
    "且**左侧四连杆 connect 残差 34.73 mm**（reset 瞬间即被撑开 3.5 cm）。\n\n"
    "现已改为自洽的 6 元素初始化 `[-0.91, 0.91, -0.91, 0.91, 0.91, 0.91]`，"
    "实测全部 joint equality 残差 **0.0**、四连杆 connect 残差 **7e-8 m**。\n\n"
    "注意：`-0.91` 是执行器行程的**接近闭合**端（并非全开）；全开对应 `Left_1 = 0`。证据："
    "`agent-work-space/zhixing_adaptation_20260927/results/gripper_audit.json`。")


def main():
    src = open(DOC, encoding="utf-8").read()
    if "2026-09-27 实测修正" in src:
        print("已更新过，跳过。")
        return 0
    if OLD_63 not in src:
        print("FATAL: 未找到 §6.3 目标段落")
        return 1
    if OLD_65 not in src:
        print("FATAL: 未找到 §6.5 目标段落")
        return 1
    shutil.copy2(DOC, BAK)
    src = src.replace(OLD_63, NEW_63, 1).replace(OLD_65, NEW_65, 1)
    open(DOC, "w", encoding="utf-8").write(src)
    print("PATCHED", DOC)
    print("  backup sha256 =", hashlib.sha256(open(BAK, "rb").read()).hexdigest())
    print("  new    sha256 =", hashlib.sha256(open(DOC, "rb").read()).hexdigest())
    return 0


if __name__ == "__main__":
    sys.exit(main())
