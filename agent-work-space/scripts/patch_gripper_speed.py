#!/usr/bin/env python3
"""让 ZhixingGripper.speed 可通过环境变量覆盖，便于做夹爪开合速度消融实验。
默认值保持 0.2（原实现），不影响既有行为。"""
import pathlib
import re

p = pathlib.Path("/mnt/data/vla-policy/openpi/robosuite/models/grippers/zhixing_gripper.py")
t = p.read_text(encoding="utf-8")

if "ZHIXING_GRIPPER_SPEED" in t:
    print("[skip] 已支持环境变量覆盖")
else:
    if "import os" not in t:
        t = t.replace("import numpy as np", "import os\n\nimport numpy as np", 1)
    old = """    @property
    def speed(self):
        return 0.2"""
    new = """    @property
    def speed(self):
        # [server-local] 可通过环境变量覆盖，用于夹爪开合速度消融实验；默认 0.2 = 原实现。
        # 参照：PandaGripper/RethinkGripper 的 speed 均为 0.01。
        return float(os.environ.get("ZHIXING_GRIPPER_SPEED", "0.2"))"""
    assert old in t, "未找到 speed 属性"
    t = t.replace(old, new, 1)
    p.write_text(t, encoding="utf-8")
    print("[patched] ZhixingGripper.speed 支持 ZHIXING_GRIPPER_SPEED")

import ast
ast.parse(p.read_text(encoding="utf-8"))
print("[ok] 语法通过")
print(p.read_text(encoding="utf-8")[-700:])
