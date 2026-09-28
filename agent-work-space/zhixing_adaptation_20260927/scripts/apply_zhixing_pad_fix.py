#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
落地 ZhiXing 夹爪修正（**字节级外科补丁**，保持原文件格式；幂等；可 dry-run）。

两项改动
--------
1. **执行器力上限**：`<position ... kp="100"/>` → 追加
   `ctrllimited="true" forcelimited="true" forcerange="-2 2"`。
   依据：同夹爪厂商 URDF `zhixing_gripper.urdf` 对 `Left_1_Joint` 声明
   `<limit lower="0" upper="-1" effort="2" velocity="1000"/>`（effort=2 N·m）。
   仓库内 PandaGripper / RethinkGripper 都有 `forcelimited`，只有 ZhiXing 缺失。

2. **指垫接触参数**（默认 `--pad-mode mesh`）：把两个**唯一接触指节**上已有的
   `left_fingerpad_col` / `right_fingerpad_col` 从「手指本体软参数」改成
   「橡胶垫参数」——与 PandaGripper 的 `fingerN_pad_collision` 同参。

为什么默认改参数而不是新增 box geom（重要，含实测教训）
--------------------------------------------------------
* **结构**：CRT-CTAG2F90 是单驱动**二指平行四连杆**夹爪。逐 body 核对：只有
  `Left_Support_Link` / `Right_Support_Link` 带碰撞 geom，其余全为
  `contype=0 conaffinity=0`（仅视觉）。
* **mesh 里已经有橡胶垫零件**：厂商 URDF 里指节上挂着一个独立的 `Left_Pad_Link`
  （fixed 关节，原点 `(-0.024010, 0.041244, 0)`），与实测内侧面中心
  `(-0.025577, 0.043602, 0.000236)` 相距仅 **2.9 mm**。也就是说
  `Left_Support_Link.STL` 已包含橡胶垫，**缺的是它的接触材料属性，不是几何**。
* **新增 box 会自碰撞（实测）**：按实测内侧面加 box 后，两块新垫在闭合时互相顶住、
  并撞到对侧手指的 mesh，夹爪卡在 **40.6 mm 合不上**（正常应到 0.52 mm）。
  两指 `Support_Link` 的 body 系并非简单镜像，同一套 body-local 坐标无法同时正确落到两侧；
  与其猜坐标，不如直接用已经正确的 mesh 几何。

`--pad-mode box` 保留作对照实验（**默认不用**，会自碰撞）。
"""
import argparse
import hashlib
import pathlib
import shutil
import sys

XML = pathlib.Path("/app/robosuite/models/assets/grippers/zhixing_gripper.xml")

# --- 1. 力上限 ---
ACT_OLD = (b'<position name="gripper" joint="Left_1_Joint" '
           b'ctrlrange="-0.91 0" kp="100"/>')

# --- 2. 指垫：改 mesh 垫的接触参数（默认路径）---
# 原：手指本体软参数；新：与 PandaGripper 专用橡胶垫一致。
MESH_PAD_OLD_PARAMS = b'solref="0.02 1" friction="1 0.005 0.0001" condim="4"'
MESH_PAD_NEW_PARAMS = (b'solref="0.01 0.5" solimp="0.99 0.999 0.0001 0.5 2" '
                       b'friction="2 0.05 0.0001" condim="4"')
MESH_PAD_OLD_AFF = b'conaffinity="1" contype="0"'
MESH_PAD_NEW_AFF = b'conaffinity="1" contype="1"'

# --- 2'. 指垫：新增 box（对照；默认关闭）---
FACE_INNER_X = -0.025610
FACE_CY = 0.043602
FACE_CZ = 0.000236
HALF_E1 = 0.019900
HALF_E2 = 0.011770
PAD_COVER = 0.55
PAD_THICK = 0.0015
PAD_QUAT = "0.50000000 -0.50000000 -0.50000000 0.50000000"
PAD_TMPL = ('            <geom type="box" name="{name}" '
            'size="{sx:.6f} {sy:.6f} {sz:.6f}" '
            'pos="{px:.6f} {py:.6f} {pz:.6f}" quat="' + PAD_QUAT + '" '
            'group="0" contype="1" conaffinity="1" condim="4" '
            'solref="0.01 0.5" solimp="0.99 0.999 0.0001 0.5 2" '
            'friction="2 0.05 0.0001" rgba="0 1 0 0.35"/>')
ANCHORS = {
    b'name="left_fingerpad_col" rgba="0 1 0 0.3"/>': "left_pad_collision",
    b'name="right_fingerpad_col" rgba="0 1 0 0.3"/>': "right_pad_collision",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-range", type=float, default=2.0)
    ap.add_argument("--pad-mode", choices=["mesh", "box", "none"], default="mesh")
    ap.add_argument("--cover", type=float, default=PAD_COVER)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    raw = XML.read_bytes()
    print("[before] sha256=%s  %d bytes"
          % (hashlib.sha256(raw).hexdigest()[:16], len(raw)))
    out = raw

    # ---- 1. 力上限 ----
    if ACT_OLD in out:
        new_act = ACT_OLD[:-2] + (b' ctrllimited="true" forcelimited="true" '
                                  b'forcerange="%s %s"/>'
                                  % (b"%.4g" % -a.force_range, b"%.4g" % a.force_range))
        out = out.replace(ACT_OLD, new_act, 1)
        print("[force] forcerange=%s" % new_act.decode().split("forcerange=")[1][:12])
    elif b'forcelimited="true"' in out:
        print("[force] 已存在，跳过")
    else:
        raise RuntimeError("找不到执行器行")

    # ---- 2/3. 指垫 ----
    if a.pad_mode == "mesh":
        n1 = out.count(MESH_PAD_OLD_PARAMS)
        n2 = out.count(MESH_PAD_OLD_AFF)
        if MESH_PAD_NEW_PARAMS in out:
            print("[pad  ] mesh 垫参数已是橡胶垫，跳过")
        elif n1 == 2 and n2 == 2:
            out = out.replace(MESH_PAD_OLD_PARAMS, MESH_PAD_NEW_PARAMS)
            out = out.replace(MESH_PAD_OLD_AFF, MESH_PAD_NEW_AFF)
            print("[pad  ] 两个 *_fingerpad_col 改为橡胶垫参数"
                  "（solref=0.01 0.5, friction=2 0.05 0.0001, contype=1）")
        else:
            raise RuntimeError("mesh 垫锚点数量异常：params=%d aff=%d（应各为 2）" % (n1, n2))
    elif a.pad_mode == "box":
        size = (HALF_E1 * a.cover, HALF_E2 * a.cover, PAD_THICK / 2.0)
        cx = FACE_INNER_X + PAD_THICK / 2.0
        for anchor, padname in ANCHORS.items():
            if padname.encode() in out:
                print("[pad  ] %s 已存在，跳过" % padname)
                continue
            ins = anchor + b"\n" + PAD_TMPL.format(
                name=padname, sx=size[0], sy=size[1], sz=size[2],
                px=cx, py=FACE_CY, pz=FACE_CZ).encode()
            out = out.replace(anchor, ins, 1)
            print("[pad  ] %s <- box（对照模式，已知会自碰撞）" % padname)

    if out == raw:
        print("\n无改动。")
        return 0
    if not a.apply:
        print("\n[dry-run] 未写入。加 --apply 才落盘。")
        return 0

    # 落盘前校验
    import xml.etree.ElementTree as ET
    root = ET.fromstring(out.decode("utf-8-sig"))
    act = [e for e in root.find("actuator") if e.get("joint") == "Left_1_Joint"]
    assert act and act[0].get("kp") == "100", "kp 被破坏"
    assert act[0].get("forcelimited") == "true"
    assert act[0].get("forcerange") == "%.4g %.4g" % (-a.force_range, a.force_range)
    pads = [g for g in root.iter("geom")
            if (g.get("name") or "") in ("left_fingerpad_col", "right_fingerpad_col")]
    assert len(pads) == 2, "找不到两个 fingerpad geom"
    if a.pad_mode == "mesh":
        for g in pads:
            assert g.get("friction") == "2 0.05 0.0001", g.get("friction")
            assert g.get("solref") == "0.01 0.5", g.get("solref")
            assert g.get("contype") == "1", g.get("contype")
    print("[check ] XML 解析 OK；kp=100 forcerange=%s；两个指垫 geom 参数正确"
          % act[0].get("forcerange"))

    bak = XML.with_suffix(".xml.pre_padfix.bak")
    if not bak.exists():
        shutil.copy2(XML, bak)
        print("[backup] %s" % bak)
    XML.write_bytes(out)
    print("[after ] sha256=%s  %d bytes"
          % (hashlib.sha256(XML.read_bytes()).hexdigest()[:16], len(out)))
    print("[done  ] 回滚：cp %s %s" % (bak, XML))
    return 0


if __name__ == "__main__":
    sys.exit(main())
