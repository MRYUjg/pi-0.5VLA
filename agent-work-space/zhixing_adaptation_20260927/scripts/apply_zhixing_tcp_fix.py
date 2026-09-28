#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
落地 N2 的两项高优先级修复（**字节级外科补丁**；幂等；可 dry-run）。

背景
----
N2 实测发现：ZhiXing 的 `grip_site` 落在**手指连杆根部**（`eef` pos `0 0 0.12`），
而物理夹持点（两指垫**内侧面中点**）在沿接近轴 **+82.54 mm**（闭合位）处。
PandaGripper 的 `grip_site` 则基本就在夹持点上（−3.60 mm）。

后果：π0.5 的动作作用在 `grip_site` 上，于是**整条轨迹的夹持点被系统性偏移
65~83 mm**——手指比预期深得多，会顶进桌面/压住物体。N2 判据 (c) 已确认
「PandaGripper 下的对齐结论对 ZhiXing 不成立」（平移差 23.000 mm）。

两项改动（必须**同批次**落地，否则起点会再低 82 mm）
----------------------------------------------------
1. **TCP 前移**：`zhixing_gripper.xml` 的 `<body name="eef" pos="0 0 0.12">`
   → `pos="0 0 0.20254"`（= 0.120 + 0.08254，闭合位夹持点）。
   语义：让 `grip_site` 与咬合点重合，与 PandaGripper 的约定一致。
   依赖：0.08254 m 来自 N2 权威口径（**内侧面中点** + 四连杆约束收敛后测量）。
   *注意*：N2 初版用「最近点对」量得 +66.03 mm 并建议 `0.18603`——该口径在
   平行平面间退化、低估 16.5 mm，**已作废**。

2. **配套 `init_qpos`**：`eco65_robot.py` 与 `mounted_eco65.py` 两处
   `[-0.128227, -0.080175, -1.482052, -0.064915, 1.563529, -1.699308]`
   → `[-0.12639, -0.222035, -0.841572, -0.563545, 1.563635, -1.697476]`。
   N2 §2.4：按**夹持点对齐**求解，残差 0.0007 mm / 0.00001°，最小关节余量
   1.5424 rad（优于当前 1.310）。因按夹持点求解，它同时适用于「修 TCP」与
   「不修 TCP」两条路线——与 `grip_site` 定义在哪无关。

连带影响（已核对，均不受损）
--------------------------
`GRIPPER_STATE_SELECTOR`（`grip_site` 是 site，不参与 state 构造）、
`_important_geoms`、动作维度（仍 7）、关节限位、PandaGripper 资产与
ECO65+PandaGripper 的 9/10 基线（本补丁只碰 ZhiXing 路径与机械臂 init_qpos）。

⚠️ 本补丁会**改变脚本专家的标定常数**：`n3_grasp_test.GRASP_OFFSET_CLOSED`
原为 `+0.08254`，落地后应变为 `0`（`grip_site` 即闭合位夹持点），
`GRASP_OFFSET_OPEN` 变为 `-0.01743`。见 `verify_tcp_fix.py`。
"""
import argparse
import hashlib
import pathlib
import shutil
import sys

XML = pathlib.Path("/app/robosuite/models/assets/grippers/zhixing_gripper.xml")
ROBOT_PY = pathlib.Path("/app/robosuite/models/robots/manipulators/eco65_robot.py")
MOUNTED_PY = pathlib.Path(
    "/app/third_party/libero/libero/libero/envs/robots/mounted_eco65.py")

# --- 1. TCP 前移 ---
EEF_OLD = b'<body name="eef" pos="0 0 0.12" quat="1 0 0 0">'
EEF_NEW = b'<body name="eef" pos="0 0 0.20254" quat="1 0 0 0">'

# --- 2. 配套 init_qpos ---
INIT_OLD = (b'return np.array([-0.128227, -0.080175, -1.482052, '
            b'-0.064915, 1.563529, -1.699308])')
INIT_NEW = (b'return np.array([-0.12639, -0.222035, -0.841572, '
            b'-0.563545, 1.563635, -1.697476])')


def patch_bytes(path, pairs, apply_, bak_suffix):
    """对单个文件按 (old, new) 做字节级替换；幂等；落盘前校验。"""
    raw = path.read_bytes()
    print("\n[file  ] %s" % path)
    print("[before] sha256=%s  %d bytes"
          % (hashlib.sha256(raw).hexdigest()[:16], len(raw)))
    out = raw
    for old, new, label in pairs:
        if new in out:
            print("[%s] 已落地，跳过" % label)
        elif old in out:
            assert out.count(old) == 1, "%s：锚点出现 %d 次（应为 1）" % (label, out.count(old))
            out = out.replace(old, new, 1)
            print("[%s] 已替换" % label)
        else:
            raise RuntimeError("%s：找不到锚点，文件状态与预期不符" % label)

    if out == raw:
        print("[state ] 无改动（已是最新）")
        return False
    if not apply_:
        print("[dry-run] 未写入。加 --apply 才落盘。")
        return False

    bak = path.with_name(path.name + bak_suffix)
    if not bak.exists():
        shutil.copy2(path, bak)
        print("[backup] %s" % bak)
    else:
        print("[backup] %s 已存在，保留原备份" % bak)

    path.write_bytes(out)
    print("[after ] sha256=%s  %d bytes"
          % (hashlib.sha256(path.read_bytes()).hexdigest()[:16], len(out)))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    changed = []

    # ---- 1. TCP ----
    if patch_bytes(XML, [(EEF_OLD, EEF_NEW, "tcp  ")],
                   a.apply, ".pre_tcpfix.bak"):
        changed.append(str(XML))

    # ---- 2. init_qpos（两处，同锚点文本）----
    for p in (ROBOT_PY, MOUNTED_PY):
        if patch_bytes(p, [(INIT_OLD, INIT_NEW, "init ")],
                       a.apply, ".pre_tcpfix.bak"):
            changed.append(str(p))

    if a.apply and changed:
        # 落盘前校验：改后的 XML 必须可解析且 eef 位置正确
        import xml.etree.ElementTree as ET
        root = ET.fromstring(XML.read_text(encoding="utf-8-sig"))
        eef = [b for b in root.iter("body") if b.get("name") == "eef"]
        assert len(eef) == 1, "eef body 数量异常"
        assert eef[0].get("pos") == "0 0 0.20254", eef[0].get("pos")
        site = [s for s in eef[0].iter("site") if s.get("name") == "grip_site"]
        assert len(site) == 1 and site[0].get("pos") == "0 0 0", "grip_site 被破坏"
        # 执行器与指垫参数（N3 落地项）必须原样保留
        act = [e for e in root.find("actuator") if e.get("joint") == "Left_1_Joint"]
        assert act and act[0].get("forcelimited") == "true", "N3 力上限被破坏"
        assert act[0].get("forcerange") == "-2 2", act[0].get("forcerange")
        pads = [g for g in root.iter("geom")
                if (g.get("name") or "").endswith("_fingerpad_col")]
        assert len(pads) == 2, "指垫 geom 数量异常"
        for g in pads:
            assert g.get("friction") == "2 0.05 0.0001", "N3 指垫参数被破坏"

        # init_qpos 校验：两处都必须已是新值
        for p in (ROBOT_PY, MOUNTED_PY):
            txt = p.read_text(encoding="utf-8")
            assert INIT_NEW.decode() in txt, "%s 未更新" % p
            assert INIT_OLD.decode() not in txt.replace("#", ""), \
                "%s 仍残留旧值" % p

        print("\n[check ] XML 解析 OK；eef=0 0 0.20254；grip_site=0 0 0；"
              "N3 力上限/指垫参数完整保留；两处 init_qpos 均为新值")
        print("[done  ] 改动 %d 个文件" % len(changed))
        print("[rollback] for f in zhixing_gripper.xml eco65_robot.py mounted_eco65.py; "
              "do cp <path>/$f.pre_tcpfix.bak <path>/$f; done")
    elif not a.apply:
        print("\n[dry-run] 汇总：将改动 %d 个文件" % len(changed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
