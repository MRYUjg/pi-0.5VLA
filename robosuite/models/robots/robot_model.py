import os

import numpy as np

from robosuite.models.base import MujocoXMLModel
from robosuite.utils.mjcf_utils import ROBOT_COLLISION_COLOR, array_to_string, string_to_array
from robosuite.utils.transform_utils import euler2mat, mat2quat

REGISTERED_ROBOTS = {}


# --------------------------------------------------------------------------- #
# [复核 §3.2] 额外关节回拉 PD 配置化
# --------------------------------------------------------------------------- #
# `SingleArm.control()` 在 `controller.run_controller()` 之后额外叠加了一个把关节
# 拉回 `init_qpos` 的 PD 项（历史硬编码 `joint_kp = 3.0`）。它不经过任务零空间投影，
# 当目标姿态需要明显偏离初始关节角时会与末端控制目标竞争。
#
# 复核实测（`rework_review_20260928/control_probe.json`）：
#   kp=3.0 → +ry 0.15 rad 姿态误差 2.257°、±90° 重定向 3.96°/3.91°，三者**均不收敛**；
#   kp=0   → 同样三个动作变为 0.212°/0.241°/0.240°，**全部收敛**。
#
# 故本项必须可配置而非写死：默认保持历史 3.0（Panda 对照与历史基线不变），
# ECO65 候选配置通过 profile 或环境变量关闭。
EXTRA_JOINT_PD_DEFAULT_KP = 3.0

# 命名 profile：robot 类名 → 该项 kp。未列出的类取默认值 3.0。
EXTRA_JOINT_PD_PROFILES = {
    # 历史行为：所有机器人都按默认 3.0（勿改）
    "legacy": {},
    # ECO65 候选：关闭额外回拉 PD，保留 OSC 控制器自身的补偿机制；
    # Panda 不在表内 → 仍为 3.0，作为对照。
    "eco65_no_extra_pd": {
        "ECO65": 0.0,
        "MountedECO65": 0.0,
        "OnTheGroundECO65": 0.0,
    },
}


def extra_joint_pd_profile():
    """当前生效的 profile 名（环境变量 `EXTRA_JOINT_PD_PROFILE`，默认 legacy）。"""
    return os.environ.get("EXTRA_JOINT_PD_PROFILE", "legacy")


def extra_joint_pd_kp_for(robot_class_name):
    """返回某机器人类当前应使用的额外关节回拉 PD 的 kp。

    优先级：`EXTRA_JOINT_PD_KP`（显式数值覆盖）> profile 表 > 默认 3.0。
    """
    profile = extra_joint_pd_profile()
    table = EXTRA_JOINT_PD_PROFILES.get(profile)
    if table is None:
        raise ValueError(
            "未知 EXTRA_JOINT_PD_PROFILE=%r；可选 %s"
            % (profile, sorted(EXTRA_JOINT_PD_PROFILES)))
    kp = table.get(robot_class_name, EXTRA_JOINT_PD_DEFAULT_KP)
    override = os.environ.get("EXTRA_JOINT_PD_KP", "")
    if override.strip():
        kp = float(override)
    return float(kp)


def register_robot(target_class):
    REGISTERED_ROBOTS[target_class.__name__] = target_class


def create_robot(robot_name, *args, **kwargs):
    """
    Instantiates a Robot object.

    Args:
        robot_name (str): Name of the robot to initialize
        *args: Additional arguments to pass to the specific Robot class initializer
        **kwargs: Additional arguments to pass to the specific Robot class initializer

    Returns:
        Robot: Desired robot

    Raises:
        Exception: [Invalid robot name]
    """
    if robot_name not in REGISTERED_ROBOTS:
        raise Exception(
            "Robot {} not found. Make sure it is a registered robot among: {}".format(
                robot_name, ", ".join(REGISTERED_ROBOTS)
            )
        )
    return REGISTERED_ROBOTS[robot_name](*args, **kwargs)


class RobotModelMeta(type):
    """Metaclass for registering robot arms"""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)

        # List all environments that should not be registered here.
        _unregistered_envs = ["RobotModel", "ManipulatorModel"]

        if cls.__name__ not in _unregistered_envs:
            register_robot(cls)
        return cls


class RobotModel(MujocoXMLModel, metaclass=RobotModelMeta):
    """
    Base class for all robot models.

    Args:
        fname (str): Path to relevant xml file from which to create this robot instance
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    def __init__(self, fname, idn=0):
        super().__init__(fname, idn=idn)

        # Define other variables that get filled later
        self.mount = None

        # Get camera names for this robot
        self.cameras = self.get_element_names(self.worldbody, "camera")

        # By default, set small frictionloss and armature values
        self.set_joint_attribute(attrib="frictionloss", values=0.1 * np.ones(self.dof), force=False)
        self.set_joint_attribute(attrib="damping", values=0.1 * np.ones(self.dof), force=False)
        self.set_joint_attribute(
            attrib="armature", values=np.array([5.0 / (i + 1) for i in range(self.dof)]), force=False
        )

    def set_base_xpos(self, pos):
        """
        Places the robot on position @pos.

        Args:
            pos (3-array): (x,y,z) position to place robot base
        """
        self._elements["root_body"].set("pos", array_to_string(pos - self.bottom_offset))

    def set_base_ori(self, rot):
        """
        Rotates robot by rotation @rot from its original orientation.

        Args:
            rot (3-array): (r,p,y) euler angles specifying the orientation for the robot base
        """
        # xml quat assumes w,x,y,z so we need to convert to this format from outputted x,y,z,w format from fcn
        rot = mat2quat(euler2mat(rot))[[3, 0, 1, 2]]
        self._elements["root_body"].set("quat", array_to_string(rot))

    def set_joint_attribute(self, attrib, values, force=True):
        """
        Sets joint attributes, e.g.: friction loss, damping, etc.

        Args:
            attrib (str): Attribute to set for all joints
            values (n-array): Values to set for each joint
            force (bool): If True, will automatically override any pre-existing value. Otherwise, if a value already
                exists for this value, it will be skipped

        Raises:
            AssertionError: [Inconsistent dimension sizes]
        """
        assert values.size == len(self._elements["joints"]), (
            "Error setting joint attributes: "
            + "Values must be same size as joint dimension. Got {}, expected {}!".format(values.size, self.dof)
        )
        for i, joint in enumerate(self._elements["joints"]):
            if force or joint.get(attrib, None) is None:
                joint.set(attrib, array_to_string(np.array([values[i]])))

    def add_mount(self, mount):
        """
        Mounts @mount to arm.

        Throws error if robot already has a mount or if mount type is incorrect.

        Args:
            mount (MountModel): mount MJCF model

        Raises:
            ValueError: [mount already added]
        """
        if self.mount is not None:
            raise ValueError("Mount already added for this robot!")

        # First adjust mount's base position
        offset = self.base_offset - mount.top_offset
        mount._elements["root_body"].set("pos", array_to_string(offset))

        self.merge(mount, merge_body=self.root_body)

        self.mount = mount

        # Update cameras in this model
        self.cameras = self.get_element_names(self.worldbody, "camera")

    # -------------------------------------------------------------------------------------- #
    # Public Properties: In general, these are the name-adjusted versions from the private   #
    #                    attributes pulled from their respective raw xml files               #
    # -------------------------------------------------------------------------------------- #

    @property
    def naming_prefix(self):
        return "robot{}_".format(self.idn)

    @property
    def dof(self):
        """
        Defines the number of DOF of the robot

        Returns:
            int: robot DOF
        """
        return len(self._joints)

    @property
    def bottom_offset(self):
        """
        Returns vector from model root body to model bottom.
        By default, this is equivalent to this robot's mount's (bottom_offset - top_offset) + this robot's base offset

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        return (
            (self.mount.bottom_offset - self.mount.top_offset) + self._base_offset
            if self.mount is not None
            else self._base_offset
        )

    @property
    def horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model. This method takes into
        account the mount horizontal radius as well

        Returns:
            float: radius
        """
        return max(self._horizontal_radius, self.mount.horizontal_radius)

    @property
    def models(self):
        """
        Returns a list of all m(sub-)models owned by this robot model. By default, this includes the mount model,
        if specified

        Returns:
            list: models owned by this object
        """
        return [self.mount] if self.mount is not None else []

    @property
    def contact_geom_rgba(self):
        return ROBOT_COLLISION_COLOR

    # -------------------------------------------------------------------------------------- #
    # All subclasses must implement the following properties                                 #
    # -------------------------------------------------------------------------------------- #

    @property
    def default_mount(self):
        """
        Defines the default mount type for this robot that gets added to root body (base)

        Returns:
            str: Default mount name to add to this robot
        """
        raise NotImplementedError

    @property
    def default_controller_config(self):
        """
        Defines the name of default controller config file in the controllers/config directory for this robot.

        Returns:
            str: filename of default controller config for this robot
        """
        raise NotImplementedError

    @property
    def init_qpos(self):
        """
        Defines the default rest qpos of this robot

        Returns:
            np.array: Default init qpos of this robot
        """
        raise NotImplementedError

    @property
    def base_xpos_offset(self):
        """
        Defines the dict of various (x,y,z) tuple offsets relative to specific arenas placed at (0,0,0)
        Assumes robot is facing forwards (in the +x direction) when determining offset. Should have entries for each
        arena case; i.e.: "bins", "empty", and "table")

        Returns:
            dict: Dict mapping arena names to robot offsets from the global origin (dict entries may also be lambdas
                for variable offsets)
        """
        raise NotImplementedError

    @property
    def top_offset(self):
        """
        Returns vector from model root body to model top.
        Useful for, e.g. placing models on a surface.
        Must be defined by subclass.

        Returns:
            np.array: (dx, dy, dz) offset vector
        """
        raise NotImplementedError

    @property
    def _horizontal_radius(self):
        """
        Returns maximum distance from model root body to any radial point of the model.

        Helps us put models programmatically without them flying away due to a huge initial contact force.
        Must be defined by subclass.

        Returns:
            float: radius
        """
        raise NotImplementedError

    @property
    def _important_sites(self):
        """
        Returns:
            dict: (Default is no important sites; i.e.: empty dict)
        """
        return {}

    @property
    def _important_geoms(self):
        """
        Returns:
             dict: (Default is no important geoms; i.e.: empty dict)
        """
        return {}

    @property
    def _important_sensors(self):
        """
        Returns:
            dict: (Default is no sensors; i.e.: empty dict)
        """
        return {}
