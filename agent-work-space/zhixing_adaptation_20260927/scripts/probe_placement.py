#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N3 前置：找「既能正常落在桌面上、又在 ECO65 可达区内」的物体摆放点。

之前的失败教训：把碗直接 set 到 (x,y,0.97) 后它会掉到 0.90（桌面外/不同表面），
而抓取高度却沿用了默认位姿的 1.0101 → 抓空。
本脚本对候选 (x,y) 逐个试放（从 1.05 落下并沉降），报告落稳后的碰撞几何 z 范围。
"""
import sys

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")
import n3_grasp_test as N  # noqa: E402

DROP_Z = 1.05
REF_TOP = 1.0226   # 碗在默认位姿落稳后的顶面


def main():
    env, _ = N.build(task_id=0)
    env.reset()
    sim, m, d, gid = N.ids(env)
    bid = N.find_object_bodies(sim, "akita_black_bowl_1")[0][0]
    jname = N.free_joint_of(sim, "akita_black_bowl_1_main")

    print(f"{'x':>7s} {'y':>7s} | {'落稳 z 范围':<24s} {'顶面':>8s} {'与参考差':>9s} | 判定")
    print("-" * 78)
    good = []
    for x in (-0.10, -0.15, -0.20, -0.25):
        for y in (0.05, 0.10, 0.15, 0.20, 0.25):
            env.reset()
            sim, m, d, gid = N.ids(env)
            N.set_free_joint(sim, jname, [x, y, DROP_Z])
            for _ in range(120):
                env.step(np.concatenate([np.zeros(6), [-1.0]]))
            sim, m, d, gid = N.ids(env)
            pts = N.body_world_pts(sim, bid, collision_only=True)
            zlo, zhi = float(pts[:, 2].min()), float(pts[:, 2].max())
            dz = zhi - REF_TOP
            ok = abs(dz) < 0.004
            if ok:
                good.append((x, y, zhi))
            print(f"{x:7.2f} {y:7.2f} | [{zlo:.4f}, {zhi:.4f}]        {zhi:.4f} "
                  f"{dz*1000:+8.1f}mm | {'OK 正常落桌' if ok else 'XX 掉落/异常'}")

    print(f"\n可用摆放点（落稳顶面与参考一致）共 {len(good)} 个：")
    for x, y, z in good:
        print(f"   ({x:+.2f}, {y:+.2f})  顶面 {z:.4f}")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
