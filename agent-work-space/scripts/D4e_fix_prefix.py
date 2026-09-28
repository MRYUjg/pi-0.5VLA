#!/usr/bin/env python3
"""D4e: 修复 ZhiXing 夹爪 <connect> 等式约束的 site 引用未加前缀问题。

背景：robosuite 的 mjcf_utils.add_prefix() 只对 MUJOCO_NAMED_ATTRIBUTES 里的属性加前缀；
该集合含 "site" 但没有 "site1"/"site2"，而 ZhiXing 夹爪 XML 用了
    <connect name="conn_left" site1="Left_Support_tip" site2="Left_2_connect"/>
merge 后 site 名被改成 gripper0_Left_Support_tip，等式约束却仍指向未加前缀的名字，
MuJoCo 编译报 "unknown element 'Left_Support_tip' in equality constraint"。

（仓库里被改坏的 mjcf_utils.py 曾试图加这两行，但同时破坏了 new_element 调用与 get_ids 断言，
  导致 robosuite 根本无法 import；此处只做正确的那一处修改。）
"""
import pathlib
import sys

path = pathlib.Path("/mnt/data/vla-policy/openpi/robosuite/utils/mjcf_utils.py")
text = path.read_text(encoding="utf-8")

if '"site1"' in text:
    print("[skip] 已包含 site1/site2")
else:
    old = '    "target",\n    "prefix",\n    "site",\n}'
    new = ('    "target",\n    "prefix",\n    "site",\n'
           '    # [server-local] <connect>/<weld> 等式约束通过 site1/site2 引用 site，同样需要加前缀\n'
           '    "site1",\n    "site2",\n}')
    assert old in text, "未找到 MUJOCO_NAMED_ATTRIBUTES 的结尾"
    text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print("[patched] MUJOCO_NAMED_ATTRIBUTES += site1, site2")

import ast
ast.parse(path.read_text(encoding="utf-8"))
print("[ok] 语法通过")
print("\n=== 当前 MUJOCO_NAMED_ATTRIBUTES ===")
lines = path.read_text(encoding="utf-8").splitlines()
start = next(i for i, ln in enumerate(lines) if ln.startswith("MUJOCO_NAMED_ATTRIBUTES"))
for ln in lines[start:start + 30]:
    print(ln)
    if ln.strip() == "}":
        break
