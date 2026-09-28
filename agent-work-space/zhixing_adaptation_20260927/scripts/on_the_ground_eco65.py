import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class OnTheGroundECO65(ManipulatorModel):
    """ECO65-B 6-DOF arm standing on the ground, LIBERO floor-arena compatible.

    为什么需要这个类
    ----------------
    LIBERO 的 arena 类型决定机器人挂载方式（`bddl_base_domain` 按 problem 类分派）：

    | problem 类 | 机器人前缀 | 工作区 |
    |---|---|---|
    | `Libero_Tabletop_Manipulation` | `Mounted` | `main_table` (z=0.90) |
    | `Libero_Kitchen_Tabletop_Manipulation` | `Mounted` | `kitchen_table` |
    | `Libero_Floor_Manipulation` | **`OnTheGround`** | `floor` (z=-0.035) |

    `libero_object`（10 个 YCB 日用品抓放任务）与 `libero_10` 都走
    `Libero_Floor_Manipulation`，会把机器人名加前缀 `OnTheGround`。
    仓库原本只有 `OnTheGroundPanda`，故这两套 suite 在 ECO65 下
    `KeyError: 'OnTheGroundECO65'` 无法构建。

    与 `MountedECO65` 的**唯一差别**（对齐 `OnTheGroundPanda` 的约定）：
      * `default_mount = None`（脚下就是地面，不装 RethinkMount）
      * `base_xpos_offset` 增补 `coffee_table` / `living_room_table`
        （地面类 arena 用得到；tabletop 不需要）
    夹爪、控制器、关节阻尼与 TCP 约定**完全一致**，故 N1/N2/N3 的
    夹爪结论与 TCP 修复对本类同样成立。

    `init_qpos` 初值沿用 `MountedECO65` 的 N2 对齐解；地面工作区的
    正式解由 `n4_solve_floor_init_pose.py` 按「夹持点对齐
    `OnTheGroundPanda` 夹持点」重解后写回。
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("robots/ECO65-B/robot.xml"), idn=idn)
        self.set_joint_attribute(
            attrib="damping", values=np.array((0.1, 0.1, 0.1, 0.1, 0.1, 0.1))
        )

    @property
    def default_mount(self):
        return None

    @property
    def default_gripper(self):
        return "ZhixingGripper"

    @property
    def default_controller_config(self):
        return "default_eco65"

    @property
    def init_qpos(self):
        return np.array([-0.12639, -0.222035, -0.841572, -0.563545, 1.563635, -1.697476])

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
            "coffee_table": lambda table_length: (-0.16 - table_length / 2, 0, 0.41),
            "living_room_table": lambda table_length: (
                -0.16 - table_length / 2,
                0,
                0.42,
            ),
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
