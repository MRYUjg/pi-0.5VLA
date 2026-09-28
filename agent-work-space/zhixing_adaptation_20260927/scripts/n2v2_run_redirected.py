#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N2-v2：把三个**没有 CLI 的历史测量脚本**重定向到新的输出目录再运行，
以便在当前资产下重测而不覆盖 results/ 里的历史证据。

被包装的脚本（源码逐字节未修改，仅覆写其模块级输出常量）：
  * probe_pad_face.py          OUT           -> <out>
  * probe_tcp_geometry.py      DEFAULT_OUT   -> <out>
  * solve_zhixing_init_pose.py DEFAULT_OUT   -> <out>

其中 probe_pad_face.py 原 CONFIGS 只有 2 个配置，本包装器把它扩成 3 个
（新增 Panda+PandaGripper），使三配置在同一口径下可比。

用法：
  python n2v2_run_redirected.py --which pad_face|tcp_geometry|init_pose --out <DIR>
"""
import argparse
import importlib.util
import os
import pathlib
import sys

DEFAULT_SCRIPTS = "/app/agent-work-space/zhixing_adaptation_20260927/scripts"
DEFAULT_OUT = "/app/agent-work-space/zhixing_adaptation_20260927/results/n2v2_20260928"

CONFIGS3 = [
    ("ECO65", "ZhixingGripper", "eco65_zhixing"),
    ("ECO65", "PandaGripper", "eco65_panda"),
    ("Panda", "PandaGripper", "panda_panda"),
]


def load(scripts_dir, name):
    path = os.path.join(scripts_dir, name + ".py")
    spec = importlib.util.spec_from_file_location("n2v2_wrapped_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # 只执行定义（有 __main__ guard）
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=["pad_face", "tcp_geometry", "init_pose"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--scripts", default=DEFAULT_SCRIPTS)
    a = ap.parse_args()

    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[n2v2] which={a.which} out={out} scripts={a.scripts}")

    if a.which == "pad_face":
        m = load(a.scripts, "probe_pad_face")
        print(f"[n2v2] original OUT={m.OUT} -> {out}")
        print(f"[n2v2] original CONFIGS={m.CONFIGS} -> {CONFIGS3}")
        m.OUT = str(out)
        m.CONFIGS = CONFIGS3
        return m.main()

    if a.which == "tcp_geometry":
        m = load(a.scripts, "probe_tcp_geometry")
        print(f"[n2v2] original DEFAULT_OUT={m.DEFAULT_OUT} -> {out}")
        m.DEFAULT_OUT = str(out)
        return m.main()

    m = load(a.scripts, "solve_zhixing_init_pose")
    print(f"[n2v2] original DEFAULT_OUT={m.DEFAULT_OUT} -> {out}")
    m.DEFAULT_OUT = str(out)
    return m.main()


if __name__ == "__main__":
    sys.exit(main())
