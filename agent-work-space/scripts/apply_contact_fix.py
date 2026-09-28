#!/usr/bin/env python3
"""把已验证的「基座—第一连杆」碰撞排除写入正式 ECO65 机器人 XML。

依据: agent-work-space/controller_validation_20260927/ECO65基座自碰撞_问题解决记录.md §5.1
 - 原始资产使用未加前缀的名字 base / link_1
 - robosuite 的 add_prefix 会加 robot0_ 前缀（MUJOCO_NAMED_ATTRIBUTES 含 body1/body2）
 - MujocoXML.merge() 会把 <contact> 合并进场景 XML
"""
import hashlib
import pathlib
import shutil
import sys

XML = pathlib.Path("/mnt/data/vla-policy/openpi/robosuite/models/assets/robots/ECO65-B/robot.xml")
EXPECT_ORIG_SHA = "e5f8dd517eda0985e4af211cc87870da342529144b6a811531c3cc6520ef9fef"
BACKUP = XML.with_suffix(".xml.pre_contact_fix.bak")

BLOCK = """  <contact>
    <!-- [contact-fix 2026-09-27] 基座固定在世界上(weldid=0)，MuJoCo 默认的父子免碰撞规则不适用；
         基座与第一连杆碰撞体初始穿透 ~12.17mm，求解器持续施加接触约束，
         导致 OSC 末端跟踪出现严重方向性误差(y +20mm 指令只走 ~3.4mm)。
         见 agent-work-space/controller_validation_20260927/ECO65基座自碰撞_问题解决记录.md -->
    <exclude body1="base" body2="link_1"/>
  </contact>
"""

text = XML.read_text(encoding="utf-8")
sha_before = hashlib.sha256(XML.read_bytes()).hexdigest()

if "<exclude" in text:
    print(f"[skip] 已存在 exclude；当前 sha256={sha_before}")
    sys.exit(0)

if sha_before != EXPECT_ORIG_SHA:
    print(f"[warn] 当前哈希与验证基线不一致：{sha_before} != {EXPECT_ORIG_SHA}")
    print("       仍继续，但请确认基线已变化的原因。")

if not BACKUP.exists():
    shutil.copy2(XML, BACKUP)
    print(f"[backup] {BACKUP}")

assert text.rstrip().endswith("</mujoco>"), "未找到 </mujoco> 结尾"
head, sep, tail = text.rpartition("</mujoco>")
new = head + BLOCK + sep + tail
XML.write_text(new, encoding="utf-8")

sha_after = hashlib.sha256(XML.read_bytes()).hexdigest()
print(f"[patched] {XML}")
print(f"  sha256: {sha_before}")
print(f"       -> {sha_after}")
print("\n--- 文件尾部 ---")
print("\n".join(XML.read_text(encoding="utf-8").splitlines()[-8:]))
