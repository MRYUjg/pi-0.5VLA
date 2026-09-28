#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""量化"陈旧 GRASP_OFFSET_CLOSED 是否还有影响"。

历史常数（n3_grasp_test.py）：
    GRASP_OFFSET_CLOSED = [-0.00006, 0.00002, 0.08254]   # 82.54 mm
它把"想要的夹持点 pinch"换算成 grip_site 目标。TCP 前移（eef body 0.120 → 0.20254）后
该常数是否还成立？本脚本直接在**当前资产**上测量：
  * grip_site 原点
  * 两指垫 geom 原点的中点（= 当前真实夹持点 pinch）
  * 二者之差在 grip_site 局部系下的表达（= 当前应当使用的 offset）
再把它反代回旧式换算，给出"用旧常数会把夹持点打偏多少毫米"。
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N          # noqa: E402
import n4_collect as N4            # noqa: E402
from n3_osc_full_chain import pad_geoms, closing_axis_world  # noqa: E402

OLD = np.array([-0.00006, 0.00002, 0.08254])


def main():
    env, task = N4.build("libero_object", 0)
    env.reset()
    sim, m, d, gid = N.ids(env)
    p, R = N.gripsite(sim, m, d, gid)
    pads = pad_geoms(sim)
    pl = np.array(d.geom_xpos[pads["left"]], dtype=float)
    pr = np.array(d.geom_xpos[pads["right"]], dtype=float)
    pinch = 0.5 * (pl + pr)
    off_world = pinch - p
    off_local = R.T @ off_world
    print("=" * 74)
    print("[tcp] eef body pos        = %s" % (N4.eef_pos_of(env),))
    print("[tcp] grip_site (world)   = %s" % np.round(p, 5).tolist())
    print("[tcp] 指垫中点 pinch(world)= %s" % np.round(pinch, 5).tolist())
    print("[tcp] pinch - grip_site   = %s  (|.|=%.2f mm)"
          % (np.round(off_world, 5).tolist(), np.linalg.norm(off_world) * 1000))
    print("[tcp] 同一向量在 grip_site 局部系 = %s  (|.|=%.2f mm)"
          % (np.round(off_local, 5).tolist(), np.linalg.norm(off_local) * 1000))
    print("[old] GRASP_OFFSET_CLOSED = %s  (|.|=%.2f mm)"
          % (OLD.tolist(), np.linalg.norm(OLD) * 1000))
    # 旧式换算：gs = pinch_target - R @ offset
    # 正确换算：gs = pinch_target - R @ off_local
    for label, off in (("旧常数", OLD), ("当前实测", off_local)):
        gs = -R @ off
        print("  %-8s 换算得到的 grip_site 相对 pinch 的偏移 = %s (|.|=%.2f mm)"
              % (label, np.round(gs, 5).tolist(), np.linalg.norm(gs) * 1000))
    err = np.linalg.norm((-R @ OLD) - (-R @ off_local)) * 1000
    print("[结论] 若仍用旧常数，grip_site 目标点会偏离正确位置 %.1f mm —— "
          "必须弃用（本 N3 返工脚本全程不使用该常数）" % err)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
