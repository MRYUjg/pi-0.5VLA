"""D7 验证：libero 镜像内 ECO65 / 知行夹爪 注册情况 + EGL 无头渲染 + torch。"""
import robosuite
import mujoco

from robosuite.models.grippers import GRIPPER_MAPPING
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.models.robots.manipulators.eco65_robot import ECO65
from robosuite.models.grippers.zhixing_gripper import ZhixingGripper

print("robosuite      :", robosuite.__version__, "| mujoco", mujoco.__version__)
print("robosuite path :", robosuite.__file__)
print("ECO65 in REGISTERED_ROBOTS  :", "ECO65" in REGISTERED_ROBOTS)
print("ECO65 in ROBOT_CLASS_MAPPING:", ROBOT_CLASS_MAPPING.get("ECO65").__name__)
print("ZhixingGripper registered   :", "ZhixingGripper" in GRIPPER_MAPPING)

r = ECO65()
print("ECO65 model    : joints =", len(r.joints), "| default_gripper =", r.default_gripper,
      "| default_mount =", r.default_mount)
print("ECO65 init_qpos:", r.init_qpos)

g = ZhixingGripper()
print("ZhixingGripper : dof =", g.dof, "| init_qpos =", g.init_qpos, "| joints =", len(g.joints))

m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
ren = mujoco.Renderer(m, 64, 64)
ren.update_scene(d)
print("EGL render OK  :", ren.render().shape)

try:
    import torch
    print("torch          :", torch.__version__, "| cuda:", torch.cuda.is_available())
except Exception as exc:  # noqa: BLE001
    print("torch          : (import failed)", exc)
