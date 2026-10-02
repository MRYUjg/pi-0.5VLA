"""
Gripper for Zhixing CRT-CTAG2F90 two-finger adaptive gripper.
Single position actuator drives Left_1_Joint; five driven joints follow
via equality constraints, with four-bar linkage keeping fingertips parallel.

动作语义（N1 体检后修正，2026-09-27）
====================================
本夹爪遵循 robosuite / LIBERO 的统一约定：**action +1 = 闭合，-1 = 张开**。

历史问题：最初实现的语义是反的（+1 张开、-1 闭合）。原因是执行器
ctrlrange 为 [-0.91, 0]，而 robosuite 用 `ctrl = bias + weight * u` 做归一化
映射（bias = -0.455, weight = 0.455），于是

    u = +1  ->  ctrl =  0.00  ->  Left_1 =  0.00  ->  开口 11.26 cm（全开）
    u = -1  ->  ctrl = -0.91  ->  Left_1 = -0.91  ->  开口  2.28 cm（近闭）

实测证据：agent-work-space/zhixing_adaptation_20260927/results/gripper_polarity.json
（双路径交叉验证），并与 PandaGripper 在同一机器人、同一协议下对照
（Panda: +1 -> 0.10 cm 闭合、-1 -> 7.94 cm 张开）。本夹爪与约定相反，
故在 `format_action` 中翻转累积方向。
"""
import os

import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class ZhixingGripperBase(GripperModel):
    """
    Zhixing CRT-CTAG2F90 is a two-finger adaptive gripper with a single
    position actuator and passive four-bar linkage.

    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/zhixing_gripper.xml"), idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        """
        6 个联动关节的自洽初始化。

        历史问题：原实现返回单元素 `[-0.91]`，而 `SingleArm.reset()`
        (robosuite/robots/single_arm.py:181) 会把它赋给全部 6 个夹爪关节索引，
        按 NumPy 广播规则 6 个关节全部变成 -0.91。但 XML 的 5 条 joint equality
        要求 Left_Support / Right_1 / Right_2 / Right_Support = -Left_1 = +0.91，
        因此 4 个关节符号错误：最大 equality 残差 1.82 rad（≈104°），
        左侧四连杆 connect 残差 34.73 mm（reset 瞬间即被撑开 3.5 cm）。

        下面按 Left_1 = -0.91（执行器行程端点，实测开口约 2.3 cm 的近闭位）
        反解出其余 5 个关节，实测残差全部为 0.0。

        注意：**-0.91 是「接近闭合」端，不是原注释所说的 fully open**；
        全开对应 Left_1 = 0（开口 11.26 cm）。若希望 reset 时全开，
        把下面的 -0.91 改成 0.0 即可（得到 [0]*6，同样自洽，且是 XML 装配零位）。
        """
        return np.array([-0.91, 0.91, -0.91, 0.91, 0.91, 0.91])

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": ["left_fingerpad_col"],
            "right_fingerpad": ["right_fingerpad_col"],
        }


class ZhixingGripper(ZhixingGripperBase):
    """
    Single-action variant: maps VLA gripper output [-1,1] to the
    position actuator with speed-limited accumulation.
    -1 => open, 1 => close
    """

    def format_action(self, action):
        assert len(action) == self.dof
        # 符号取负以对齐 +1=闭合 / -1=张开 的统一约定（理由见模块 docstring）。
        # 修正前此处为 `+ self.speed * np.sign(action)`，实际得到 +1=张开、-1=闭合。
        self.current_action = np.clip(
            self.current_action - self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action

    @property
    def speed(self):
        # [server-local] 可通过环境变量覆盖，用于夹爪开合速度消融实验。
        #
        # 2026-09-30 变更：默认值由 0.2 改为 **0.01**（部署值）。
        # 理由（实测，见 agent-work-space/outputs/ZhiXing夹爪控制问题清单_20260930.md）：
        #   speed=0.2 时闭合只需 5 个控制步（0.25 s），指垫会**拨飞/扎穿**物体——
        #   同一 30 mm 试块、同一链路：0.2 → 指垫穿透 17–23 mm、抬升 0、0/15 全败；
        #   0.01 → 抬升 74 mm。0.01 亦与 PandaGripper/RethinkGripper 的 speed 一致。
        # 环境变量 ZHIXING_GRIPPER_SPEED 仍然优先，消融实验照旧可覆盖。
        return float(os.environ.get("ZHIXING_GRIPPER_SPEED", "0.01"))

    @property
    def dof(self):
        return 1
