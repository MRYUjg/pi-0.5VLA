import pathlib
import cv2
ROOT = pathlib.Path("/app/agent-work-space/eval")
OUT = ROOT / "_compare"
OUT.mkdir(parents=True, exist_ok=True)

def frames(p):
    cap = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out

for tag, n in (("G_panda_zhixing_speed001", 3), ("D_panda_panda_randinit", 3)):
    d = ROOT / tag / "videos"
    v = sorted(p for p in d.glob("*.mp4") if not p.name.startswith("wrist__"))[0]
    fr = frames(v)
    idxs = [int(len(fr)*k/4) for k in range(4)][1:]
    for i in idxs:
        cv2.imwrite(str(OUT / f"spd_{tag}_f{i:03d}.png"), fr[i])
    print(tag, v.name[:50], "frames=", len(fr), "picks=", idxs)
print("OK")
