"""对比 B(ECO65+PandaGripper) 与 D(Panda+PandaGripper) 在同一任务/同 seed 下的观测差异。
两者夹爪相同、初始化协议相同，唯一变量是机械臂。"""
import pathlib
import cv2
import numpy as np

ROOT = pathlib.Path("/app/agent-work-space/eval")
OUT = ROOT / "_compare"
OUT.mkdir(parents=True, exist_ok=True)


def frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return out


def pick(tag):
    d = ROOT / tag / "videos"
    v = sorted(p for p in d.glob("*.mp4") if not p.name.startswith("wrist__"))[0]
    w = d / ("wrist__" + v.name)
    return v.name, frames(v), frames(w)


name_b, av_b, wr_b = pick("B_eco65_panda")
name_d, av_d, wr_d = pick("D_panda_panda_randinit")
print("B video:", name_b, len(av_b), "frames")
print("D video:", name_d, len(av_d), "frames")

n = min(len(av_b), len(av_d))
for cam, a, b in (("agentview", av_b, av_d), ("wrist", wr_b, wr_d)):
    for idx in (0, min(20, n - 1), min(60, n - 1)):
        cv2.imwrite(str(OUT / f"cmp_{cam}_ECO65_f{idx:03d}.png"), a[idx][..., ::-1])
        cv2.imwrite(str(OUT / f"cmp_{cam}_Panda_f{idx:03d}.png"), b[idx][..., ::-1])
    # 数值差异（缩到 64x64 再比，避免细节噪声）
    diffs = []
    for i in range(0, n, 10):
        x = cv2.resize(a[i], (64, 64)).astype(np.float32)
        y = cv2.resize(b[i], (64, 64)).astype(np.float32)
        diffs.append(float(np.abs(x - y).mean()))
    print(f"{cam:10s} 平均像素差 {np.mean(diffs):6.2f} / 255  最大 {np.max(diffs):6.2f}  (n={len(diffs)})")
    # 画面下 1/3 的亮度分布（腕部视角里夹爪/手指所在区域）
    if cam == "wrist":
        bot_a = np.mean([a[i][150:, :, :].mean() for i in range(0, n, 10)])
        bot_b = np.mean([b[i][150:, :, :].mean() for i in range(0, n, 10)])
        print(f"{cam:10s} 画面下 1/3 平均亮度: ECO65 {bot_a:.1f} vs Panda {bot_b:.1f}")
print("COMPARE_OK")
