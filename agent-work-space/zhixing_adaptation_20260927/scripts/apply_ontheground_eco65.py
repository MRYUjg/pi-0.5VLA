#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
落地 `OnTheGroundECO65`（地面挂载的 ECO65）——**幂等；可 dry-run**。

为什么需要
----------
`libero_object`（10 个 YCB 抓放任务）与 `libero_10` 使用
`Libero_Floor_Manipulation`，该类会把机器人名加前缀 `OnTheGround`：
    kwargs["robots"] = [f"OnTheGround{r}" for r in kwargs["robots"]]
仓库原本只有 `OnTheGroundPanda`，所以这两套 suite 在 ECO65 下报
`KeyError: 'OnTheGroundECO65'` 而无法构建。实测（n4_suite_probe）：

| suite | 构建 | 抓取目标可抓 |
|---|---|---|
| libero_spatial | ✓ | **0/10**（全是 107 mm 黑碗） |
| libero_goal    | ✓ | 仅 2/10（酒杯；其余是抽屉/灶台等非抓取任务，或 107 mm 碗） |
| libero_object  | ✗ 缺类 | 待测（YCB 罐头/纸盒，预期可抓） |
| libero_10      | ✗ 缺类 | 待测 |

两项改动
--------
1. 新增 `third_party/libero/libero/libero/envs/robots/on_the_ground_eco65.py`
   （源文件取自 agent-work-space 的同名脚本，保持单一真源）。
2. 在 `.../robots/__init__.py` 里 import 并注册到 `ROBOT_CLASS_MAPPING`。

不改动 `MountedECO65`，不影响 libero_spatial 的 9/10 基线。
"""
import argparse
import hashlib
import pathlib
import shutil
import sys

ROBOTS_DIR = pathlib.Path(
    "/app/third_party/libero/libero/libero/envs/robots")
SRC = pathlib.Path(
    "/app/agent-work-space/zhixing_adaptation_20260927/scripts/on_the_ground_eco65.py")
DST = ROBOTS_DIR / "on_the_ground_eco65.py"
INIT = ROBOTS_DIR / "__init__.py"

IMPORT_ANCHOR = b"from .mounted_eco65 import MountedECO65"
IMPORT_ADD = b"\nfrom .on_the_ground_eco65 import OnTheGroundECO65"

REG_ANCHOR = b'        "MountedECO65": SingleArm,'
REG_ADD = b'\n        "OnTheGroundECO65": SingleArm,'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    print("[src  ] %s" % SRC)
    src_bytes = SRC.read_bytes()
    print("[class] sha256=%s  %d bytes"
          % (hashlib.sha256(src_bytes).hexdigest()[:16], len(src_bytes)))

    raw = INIT.read_bytes()
    print("\n[init ] %s\n[before] sha256=%s  %d bytes"
          % (INIT, hashlib.sha256(raw).hexdigest()[:16], len(raw)))

    out = raw
    for anchor, add, label in ((IMPORT_ANCHOR, IMPORT_ADD, "import"),
                               (REG_ANCHOR, REG_ADD, "registry")):
        if add.strip() in out:
            print("[%s] 已存在，跳过" % label)
        elif anchor in out:
            assert out.count(anchor) == 1, "%s 锚点不唯一" % label
            out = out.replace(anchor, anchor + add, 1)
            print("[%s] 已追加" % label)
        else:
            raise RuntimeError("%s 锚点缺失" % label)

    class_exists = DST.exists()
    same = class_exists and DST.read_bytes() == src_bytes
    print("[class] 目标文件%s" % ("已存在且一致" if same else
                                  ("已存在但不同" if class_exists else "不存在")))

    if same and out == raw:
        print("\n[state ] 无改动（已是最新）")
        return 0
    if not a.apply:
        print("\n[dry-run] 未写入。加 --apply 才落盘。")
        return 0

    if not same:
        bak = DST.with_suffix(".py.pre_otg.bak")
        if class_exists and not bak.exists():
            shutil.copy2(DST, bak)
            print("[backup] %s" % bak)
        DST.write_bytes(src_bytes)
        print("[class] 已写入 %s" % DST)

    if out != raw:
        bak = INIT.with_suffix(".py.pre_otg.bak")
        if not bak.exists():
            shutil.copy2(INIT, bak)
            print("[backup] %s" % bak)
        INIT.write_bytes(out)
        print("[init ] sha256=%s  %d bytes"
              % (hashlib.sha256(INIT.read_bytes()).hexdigest()[:16], len(out)))

    # 落盘校验：import 必须成功、类必须注册且属性正确
    # （容器内 libero 已按 `libero.libero` 安装，不得再往 sys.path 插内层目录）
    from libero.libero.envs.robots import OnTheGroundECO65      # noqa: E402
    from robosuite.robots import ROBOT_CLASS_MAPPING            # noqa: E402

    assert "OnTheGroundECO65" in ROBOT_CLASS_MAPPING, "未注册到 ROBOT_CLASS_MAPPING"
    r = OnTheGroundECO65()
    assert r.default_mount is None, "default_mount 应为 None（地面）"
    assert r.default_gripper == "ZhixingGripper", r.default_gripper
    assert len(r.init_qpos) == 6, "init_qpos 应为 6 维"
    print("\n[check ] import OK；已注册；default_mount=None；"
          "gripper=%s；init_qpos=%s"
          % (r.default_gripper, list(r.init_qpos)))
    print("[done  ] 回滚：cp %s.pre_otg.bak %s && cp %s.pre_otg.bak %s"
          % (INIT, INIT, DST, DST))
    return 0


if __name__ == "__main__":
    sys.exit(main())
