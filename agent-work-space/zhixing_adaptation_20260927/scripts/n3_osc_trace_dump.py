#!/usr/bin/env python3
import sys
import numpy as np
p = "/app/agent-work-space/zhixing_adaptation_20260927/results/n3_osc_full_chain_traces.npz"
z = np.load(p, allow_pickle=True)
keys = sys.argv[1:] or [k for k in z.files]
for k in keys:
    a = z[k].item()
    gs, oc, ph, disp, pen = a["grip_site"], a["obj_center"], a["phase"], a["step_disp"], a["pen_pad_mm"]
    print("=" * 74)
    print(k, "frames", gs.shape[0])
    cur = None
    for i, nm in enumerate(ph):
        if nm != cur:
            cur = nm
            print("  >> %-22s @%4d" % (nm, i))
    for nm in ("descend", "close"):
        sel = np.where(ph == nm)[0]
        if sel.size == 0:
            continue
        print("  [%s] n=%d" % (nm, sel.size))
        for j in range(0, sel.size, max(1, sel.size // 12)):
            i = sel[j]
            print("     t=%4d gs=%s obj=%s disp=%.4f pen=%.3f"
                  % (i, np.round(gs[i], 4).tolist(), np.round(oc[i], 4).tolist(),
                     disp[i], pen[i]))
