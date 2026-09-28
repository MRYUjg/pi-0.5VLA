"""从评测视频里抽关键帧，用于对比 ECO65 与 Panda 的观测差异（用 cv2 读，避免 imageio 插件依赖）。"""
import pathlib

import cv2
import numpy as np

ROOT = pathlib.Path("/app/agent-work-space/eval")
OUT = ROOT / "_compare"
OUT.mkdir(parents=True, exist_ok=True)


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


for tag in ("A_eco65_zhixing", "B_eco65_panda", "C_panda_panda"):
    d = ROOT / tag / "videos"
    vids = sorted(p for p in d.glob("*.mp4") if not p.name.startswith("wrist__"))
    if not vids:
        print(tag, "no video")
        continue
    v = vids[0]
    frames = read_frames(v)
    picks = sorted({0, 20, len(frames) // 2, len(frames) - 1})
    for idx in picks:
        cv2.imwrite(str(OUT / f"{tag}__agentview__f{idx:03d}.png"), frames[idx][..., ::-1])
    w = d / ("wrist__" + v.name)
    if w.exists():
        wf = read_frames(w)
        for idx in sorted({0, len(wf) // 2, len(wf) - 1}):
            cv2.imwrite(str(OUT / f"{tag}__wrist__f{idx:03d}.png"), wf[idx][..., ::-1])
    print(f"{tag}: {v.name}  frames={len(frames)}  picks={picks}")

print("FRAMES_OK")
