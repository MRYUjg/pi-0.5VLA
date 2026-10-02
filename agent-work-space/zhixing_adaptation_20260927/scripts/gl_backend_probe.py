#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GL 后端对比：EGL(GPU) vs OSMesa(CPU) —— 回答"不租 GPU 能否跑当前工作"。

只测当前工作真正依赖的那一环：**离屏渲染出双视角图像**（物理仿真本来就是 CPU）。
用法：
  MUJOCO_GL=osmesa python gl_backend_probe.py
  MUJOCO_GL=egl    python gl_backend_probe.py
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, "/app/agent-work-space/zhixing_adaptation_20260927/scripts")

OUT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927/results")


def main():
    backend = os.environ.get("MUJOCO_GL", "<unset>")
    rec = {"backend": backend}
    t0 = time.time()
    try:
        import mujoco
        rec["mujoco"] = mujoco.__version__
        rec["gl_context"] = type(mujoco.gl_context).__name__ if hasattr(mujoco, "gl_context") else None
    except Exception as e:                                # noqa: BLE001
        rec["error"] = "import mujoco: %s" % e
        print(json.dumps(rec, ensure_ascii=False))
        return 1

    # 直接用 LIBERO env 构建（与 N3/N4 同一条路径）
    try:
        import n4_collect as N4
        t_build = time.time()
        env, task = N4.build("libero_object", 0, seed=1)
        rec["build_s"] = round(time.time() - t_build, 2)

        # 渲染：reset + 取双视角图像
        t_r = time.time()
        obs = env.reset()
        av = np.asarray(obs["agentview_image"])
        wr = np.asarray(obs["robot0_eye_in_hand_image"])
        rec["reset_render_s"] = round(time.time() - t_r, 2)
        rec["agentview_shape"] = list(av.shape)
        rec["wrist_shape"] = list(wr.shape)
        rec["agentview_mean"] = round(float(av.mean()), 3)
        rec["wrist_mean"] = round(float(wr.mean()), 3)
        rec["images_nonblank"] = bool(av.mean() > 1.0 and wr.mean() > 1.0)
        rec["two_cameras_differ"] = bool(
            float(np.mean(np.abs(av.astype(np.int16) - wr.astype(np.int16)))) > 5.0)

        # 物理步进 + 每步渲染的吞吐（当前工作最重的循环）
        t_s = time.time()
        n = 60
        for _ in range(n):
            env.step(np.concatenate([np.zeros(6), [1.0]]))
        dt = time.time() - t_s
        rec["steps"] = n
        rec["step_render_s"] = round(dt, 2)
        rec["sec_per_step"] = round(dt / n, 4)
        rec["steps_per_s"] = round(n / dt, 2)
        rec["control_hz_required"] = 20.0
        rec["realtime_factor"] = round((n / dt) / 20.0, 3)
        env.close()
        rec["ok"] = True
    except Exception as e:                                # noqa: BLE001
        rec["ok"] = False
        rec["error"] = "%s: %s" % (type(e).__name__, str(e)[:300])
    rec["total_s"] = round(time.time() - t0, 2)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / ("gl_backend_%s.json" % backend.replace("/", "_"))
    p.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rec, ensure_ascii=False))
    return 0 if rec.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
