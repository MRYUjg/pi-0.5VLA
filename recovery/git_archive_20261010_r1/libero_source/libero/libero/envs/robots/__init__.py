from .mounted_panda import MountedPanda
from .on_the_ground_panda import OnTheGroundPanda
from .mounted_eco65 import MountedECO65
from .on_the_ground_eco65 import OnTheGroundECO65

from robosuite.robots.single_arm import SingleArm
from robosuite.robots import ROBOT_CLASS_MAPPING

ROBOT_CLASS_MAPPING.update(
    {
        "MountedPanda": SingleArm,
        "OnTheGroundPanda": SingleArm,
        "MountedECO65": SingleArm,
        "OnTheGroundECO65": SingleArm,
    }
)

