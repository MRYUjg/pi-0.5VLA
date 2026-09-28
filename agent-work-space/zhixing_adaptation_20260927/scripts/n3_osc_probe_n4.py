#!/usr/bin/env python3
"""读 N4 成功 episode，看搬运/投放段 grip_site 实际能到哪里。"""
import json, sys, pathlib
import numpy as np

ROOT = pathlib.Path("/app/agent-work-space/zhixing_adaptation_20260927")
for name in sys.argv[1:] or ["raw/success/n4_libero_object_t0_s2_0002.npz"]:
    p = ROOT / name
    z = np.load(p, allow_pickle=True)
    meta = json.loads(str(z["_meta_json"]))
    ph = json.loads(str(z["_phases_json"]))
    pid = z["phase_id"]
    gs = z["grip_site"]
    print("=" * 70)
    print(name, " frames=", gs.shape[0])
    print("basket lo/hi:", np.round(meta["basket_bbox_lo"], 4).tolist(),
          np.round(meta["basket_bbox_hi"], 4).tolist())
    print("obj bbox lo/hi:", np.round(meta["object_bbox_lo"], 4).tolist(),
          np.round(meta["object_bbox_hi"], 4).tolist())
    print("pin:", np.round(meta["grasp_pinch_target"], 4).tolist(),
          " z_g=", round(meta["grasp_z"], 4))
    print("phase_end:", json.dumps(meta["phase_end"], ensure_ascii=False))
    print("final obj center:", np.round(meta["final_object_center"], 4).tolist())
    print("success:", meta["success"], "pred:", meta["success_predicate_libero"],
          "geom:", meta["geometric_in_basket"])
    for i, nm in enumerate(ph):
        sel = np.where(pid == i)[0]
        if sel.size == 0:
            continue
        g = gs[sel]
        print("  %-10s n=%3d  grip_site x[%.4f,%.4f] y[%.4f,%.4f] z[%.4f,%.4f]  last=%s"
              % (nm, sel.size, g[:, 0].min(), g[:, 0].max(), g[:, 1].min(),
                 g[:, 1].max(), g[:, 2].min(), g[:, 2].max(),
                 np.round(g[-1], 4).tolist()))
