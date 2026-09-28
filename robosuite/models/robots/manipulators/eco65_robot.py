import warnings

import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


# --------------------------------------------------------------------------- #
# [R1] ECO65 初始关节角：按**实际挂载的夹爪**选择（唯一真源）
# --------------------------------------------------------------------------- #
# ECO65 的初始关节角必须与挂载的夹爪匹配，两套解不可互换：
#
#   PandaGripper   ：N2 阶段用 DLS IK 对齐 Panda 初始 grip_site 6D 位姿得到的解，
#                    对应 ECO65+PandaGripper 的历史 9/10 基线。
#   ZhixingGripper ：按 ZhiXing 的 TCP（eef 由 0.120 前移到 0.20254）重新求解的解。
#
# 审核记录 R1 实测：ECO65+PandaGripper 误用 ZhiXing 解后，初始 grip_site 偏离
# Panda 参考 109.1435 mm；换回 Panda 解仅 0.0012 mm。原先两个机器人类都无条件
# 返回 ZhiXing 解，导致 9/10 基线的初始条件被悄悄改掉。
#
# 解析时序（重要）：`Robot.load_model()` 在 `add_gripper()` **之前**就读取一次
# `robot_model.init_qpos`，那时 `self.grippers` 仍为空。因此
# `SingleArm.load_model()` 在挂载夹爪后会**重新解析**一次，见
# `robosuite/robots/single_arm.py` 中的 [R1] 注释与 `Robot._init_qpos_explicit`。
ECO65_INIT_QPOS_BY_GRIPPER = {
    "PandaGripper": np.array(
        [-0.128227, -0.080175, -1.482052, -0.064915, 1.563529, -1.699308]
    ),
    "ZhixingGripper": np.array(
        [-0.12639, -0.222035, -0.841572, -0.563545, 1.563635, -1.697476]
    ),
}

# 夹爪尚未挂载（首次解析）或类型未登记时的兜底：取 Panda 解，与历史 9/10 基线一致。
ECO65_INIT_QPOS_FALLBACK = ECO65_INIT_QPOS_BY_GRIPPER["PandaGripper"]


def attached_gripper_name(model):
    """返回模型上已挂载夹爪的类名；尚未挂载时返回 None。"""
    grippers = getattr(model, "grippers", None)
    if not grippers:
        return None
    return type(list(grippers.values())[-1]).__name__


def eco65_init_qpos_for_model(model):
    """按 model 上**实际挂载的夹爪**返回 ECO65 初始关节角。

    `robosuite` 与 LIBERO（`MountedECO65`）共用本函数，避免两处约定漂移。
    """
    name = attached_gripper_name(model)
    if name is None:
        # 发生在 add_gripper() 之前；SingleArm.load_model() 随后会重新解析并纠正。
        return ECO65_INIT_QPOS_FALLBACK.copy()
    if name in ECO65_INIT_QPOS_BY_GRIPPER:
        return ECO65_INIT_QPOS_BY_GRIPPER[name].copy()
    warnings.warn(
        "ECO65 未登记夹爪 %r 的初始关节角，已回退到 PandaGripper 解；"
        "请在 ECO65_INIT_QPOS_BY_GRIPPER 中补充该夹爪的实测解。" % (name,),
        RuntimeWarning,
        stacklevel=2,
    )
    return ECO65_INIT_QPOS_FALLBACK.copy()


class ECO65(ManipulatorModel):
    """
    ECO65-B is a 6-DOF collaborative robot arm.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/ECO65-B/robot.xml"), idn=idn)

        # Set joint damping (6 joints)
        self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1)))

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "ZhixingGripper"

    @property
    def default_controller_config(self):
        return "default_eco65"

    @property
    def init_qpos(self):
        # [R1] 按实际挂载的夹爪选择，不再无条件返回 ZhiXing 解。
        return eco65_init_qpos_for_model(self)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        return "single"
