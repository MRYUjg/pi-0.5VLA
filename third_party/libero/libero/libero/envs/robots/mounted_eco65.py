import numpy as np

from robosuite.models.robots.manipulators.eco65_robot import eco65_init_qpos_for_model
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class MountedECO65(ManipulatorModel):
    """ECO65-B 6-DOF arm mounted on a table, LIBERO-compatible."""

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/ECO65-B/robot.xml"), idn=idn)
        self.set_joint_attribute(attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1)))

    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "ZhixingGripper"
        # return "RethinkGripper"

    @property
    def default_controller_config(self):
        return "default_eco65"

    @property
    def init_qpos(self):
        # [R1] 与 robosuite/models/robots/manipulators/eco65_robot.py 共用同一张
        # ECO65_INIT_QPOS_BY_GRIPPER 表（Panda 解 / ZhiXing 解），避免两处约定漂移。
        return eco65_init_qpos_for_model(self)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "study_table": lambda table_length: (-0.25 - table_length / 2, 0, 0),
            "kitchen_table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
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
