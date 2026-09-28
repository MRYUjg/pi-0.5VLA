#!/usr/bin/env python3
"""给 ZhixingGripper 加两个**默认关闭**的进程内开关（不改变既有行为）。

  ZHIXING_GRIPPER_INVERT=1          修正开合极性：+1=闭合、-1=张开（对齐 Panda/LIBERO）
  ZHIXING_GRIPPER_CONSISTENT_INIT=1 init_qpos 返回与 5 条 joint equality 自洽的 6 元素

依据 N1 体检结论（agent-work-space/zhixing_adaptation_20260927/results/）：
  - 声明式 init_qpos=[-0.91] 被广播到 6 关节 → 4 个关节符号错，最大 equality 残差 1.82 rad，
    左侧四连杆 connect 残差 34.7 mm。
  - 极性相反：action=-1 → ctrl=-0.91 → 开口 2.28 cm（近闭）；
    action=+1 → ctrl=0 → 开口 11.26 cm（全开）。Panda 基准为 +1=闭合。

幂等：重复执行不会重复插入。默认（不带环境变量）行为与打补丁前**逐位一致**。
"""
import hashlib
import os
import shutil
import sys

REPO = "/mnt/data/vla-policy/openpi"
TARGET = os.path.join(REPO, "robosuite/models/grippers/zhixing_gripper.py")
BAK = TARGET + ".pre_polarity_switch.bak"

HELPER = '''

def _flag(name):
    """[server-local] 进程内开关：默认关闭，保持既有行为不变。"""
    return os.environ.get(name, "0").strip().lower() in ("1", "true", "yes", "on")
'''

OLD_INIT = '''    @property
    def init_qpos(self):
        # Left_1_Joint ctrlrange is [-0.91, 0]; -0.91 = fully open
        return np.array([-0.91])'''

NEW_INIT = '''    @property
    def init_qpos(self):
        # Left_1_Joint ctrlrange is [-0.91, 0]；N1 实测：ctrl=-0.91 → 开口 2.3~3.0 cm（近闭），
        # ctrl=0 → 开口 11.3 cm（全开）。故 -0.91 实际是「闭合」而非原注释所说的 fully open。
        if _flag("ZHIXING_GRIPPER_CONSISTENT_INIT"):
            # 与 XML 中 5 条 joint equality 自洽的 6 元素初始化（N1 实测残差 0.0）
            return np.array([-0.91, 0.91, -0.91, 0.91, 0.91, 0.91])
        return np.array([-0.91])'''

OLD_FMT = '''    def format_action(self, action):
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + self.speed * np.sign(action), -1.0, 1.0
        )
        return self.current_action'''

NEW_FMT = '''    def format_action(self, action):
        assert len(action) == self.dof
        sign = np.sign(action)
        if _flag("ZHIXING_GRIPPER_INVERT"):
            # [server-local] N1 实测：本夹爪 action=+1 实际张开、-1 实际闭合，与 Panda/LIBERO
            # 约定（+1=闭合）相反。开启后翻转符号，使 +1=闭合、-1=张开。
            sign = -sign
        self.current_action = np.clip(
            self.current_action + self.speed * sign, -1.0, 1.0
        )
        return self.current_action'''


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if not os.path.isfile(TARGET):
        print("FATAL: 找不到 " + TARGET)
        return 1
    src = open(TARGET, encoding="utf-8-sig").read()
    before = sha(TARGET)

    if "_flag(" in src and "ZHIXING_GRIPPER_INVERT" in src:
        print("已打过补丁，跳过。sha256 =", before)
        return 0

    if not os.path.isfile(BAK):
        shutil.copy2(TARGET, BAK)
        print("备份 ->", BAK)

    if OLD_INIT not in src:
        print("FATAL: 未找到预期的 init_qpos 片段，拒绝修改")
        return 1
    if OLD_FMT not in src:
        print("FATAL: 未找到预期的 format_action 片段，拒绝修改")
        return 1

    # _flag 助手插到 class 定义之前
    anchor = "class ZhixingGripperBase(GripperModel):"
    if anchor not in src:
        print("FATAL: 未找到 class 定义锚点")
        return 1
    src = src.replace(anchor, HELPER.strip("\n") + "\n\n\n" + anchor, 1)
    src = src.replace(OLD_INIT, NEW_INIT, 1)
    src = src.replace(OLD_FMT, NEW_FMT, 1)

    open(TARGET, "w", encoding="utf-8").write(src)
    after = sha(TARGET)
    print("PATCHED")
    print("  before sha256 =", before)
    print("  after  sha256 =", after)
    print("  backup sha256 =", sha(BAK))
    return 0


if __name__ == "__main__":
    sys.exit(main())
