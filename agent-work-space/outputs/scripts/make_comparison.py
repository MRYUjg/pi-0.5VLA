#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 outputs/ 下的专家验证 JSON 汇总成对照表（markdown + csv）。

只读 outputs/*.json，不改任何原始产物。
"""
import csv
import glob
import json
import pathlib
import sys

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                   "/app/agent-work-space/outputs")

FIELDS = [
    ("file", lambda p, d: p.name),
    ("gripper", lambda p, d: d.get("gripper")),
    ("block_w_mm", lambda p, d: round(d.get("block", {}).get("width_mm", 0), 1)),
    ("n_total", lambda p, d: d.get("summary", {}).get("n_total")),
    ("n_pass_n3", lambda p, d: d.get("summary", {}).get("n_success_n3_gates")),
    ("n_pass_n4", lambda p, d: d.get("summary", {}).get("n_success_n4_style")),
    ("n_transport_conv", lambda p, d: d.get("summary", {}).get("n_transport_converged")),
    ("pen_pad_max_mm", lambda p, d: _mx(d, "penetration_peak_pad_mm")),
    ("lift_min_mm", lambda p, d: _mn(d, "lift_mm")),
    ("transport_res_min_m", lambda p, d: _mn(d, "transport_residual_m")),
    ("transport_res_max_m", lambda p, d: _mx(d, "transport_residual_m")),
    ("jaw_after_close_mm", lambda p, d: _mn(d, "jaw_after_close_mm")),
    ("close_steps", lambda p, d: _mn(d, "close_steps")),
    ("wall_s_per_trial", lambda p, d: _avg(d, "wall_seconds")),
]


def _trials(d):
    return d.get("trials") or []


def _mx(d, key):
    vals = [t.get(key) for t in _trials(d) if t.get(key) is not None]
    return round(max(vals), 4) if vals else None


def _mn(d, key):
    vals = [t.get("phase_end", {}).get(key) for t in _trials(d)
            if t.get("phase_end", {}).get(key) is not None]
    return round(min(vals), 4) if vals else None


def _avg(d, key):
    vals = [t.get(key) for t in _trials(d) if t.get(key) is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def main():
    rows = []
    # 只汇总顶层 JSON（子目录 _deprecated_* 是历史上被机具缺陷污染的运行，不参与对照）
    for f in sorted(glob.glob(str(OUT / "*.json"))):
        p = pathlib.Path(f)
        if p.parent != OUT or p.name.startswith(("comparison", "SUMMARY")):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if "trials" not in d and "pilot" not in d:
            continue
        rows.append({name: fn(p, d) for name, fn in FIELDS})
        if "pilot" in d:
            pil = d["pilot"]
            rows[-1]["pilot_polarity_ok"] = pil["checks"]["sign_convention_ok"]
            rows[-1]["pilot_stroke_mm"] = round(pil["checks"]["stroke_mm"], 2)
            rows[-1]["pilot_repeat_ok"] = pil["repeatability"]["pass"]
            rows[-1]["pilot_open_mm"] = round(pil["repeatability"]["open_mm"]["mean"], 2)
            rows[-1]["pilot_closed_mm"] = round(pil["repeatability"]["closed_mm"]["mean"], 2)

    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(OUT / "comparison.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    lines = ["| " + " | ".join(keys) + " |",
             "|" + "|".join(["---"] * len(keys)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(k) is None else str(r.get(k))
                                       for k in keys) + " |")
    md = "\n".join(lines) + "\n"
    (OUT / "comparison.md").write_text(md, encoding="utf-8")
    print(md)
    print("[out]", OUT / "comparison.csv", "and", OUT / "comparison.md")


if __name__ == "__main__":
    main()
