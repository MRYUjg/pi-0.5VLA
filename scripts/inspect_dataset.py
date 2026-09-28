"""Quick inspect script for the converted LeRobot dataset."""
import json, sys
from pathlib import Path

DATASET_DIR = Path(r"E:\111PROject\robot-vla\openpi\libreo_raw_data\local\libero_spatial")

info = json.loads((DATASET_DIR / "meta/info.json").read_text())
print(f"=== Dataset: {info['total_episodes']} eps x {info['total_frames']} frames ===")

try:
    import pyarrow.parquet as pq
    table = pq.read_table(str(DATASET_DIR / "data/chunk-000/episode_000000.parquet"))
    print(f"Episode 0: {table.num_rows} frames, columns: {table.column_names}\n")

    # Check what image column actually contains
    for key in ["image", "wrist_image"]:
        val = table.column(key)[0].as_py()
        if isinstance(val, dict):
            print(f"{key}: dict keys={list(val.keys())}, path='{val.get('path','')}'")
            # Search for actual image files
            for search_dir in [DATASET_DIR, DATASET_DIR.parent]:
                for pattern in ["images/**/*.png", "images/**/*.jpg", "**/*.png", "**/*.jpg"]:
                    found = list(search_dir.glob(pattern))
                    if found:
                        print(f"  -> found {len(found)} images in {search_dir}/{pattern}")
                        print(f"     sample: {found[0]}")
                        break
                else:
                    continue
                break
            else:
                print(f"  -> no image files found on disk (but data loader will resolve at runtime)")
        elif isinstance(val, bytes):
            print(f"{key}: bytes, len={len(val)} (embedded)")
        else:
            print(f"{key}: type={type(val).__name__}")

    # State & action stats
    import numpy as np
    states = np.array([table.column("state")[i].as_py() for i in range(min(500, table.num_rows))])
    actions = np.array([table.column("actions")[i].as_py() for i in range(min(500, table.num_rows))])
    
    print(f"\n=== Stats (first {len(states)} frames) ===")
    print(f"{'Dim':<12} {'state_mean':>10} {'state_std':>10}  {'action_mean':>11} {'action_std':>10}")
    for d in range(max(states.shape[1], actions.shape[1])):
        sm, ss = (states[:,d].mean(), states[:,d].std()) if d < states.shape[1] else (0,0)
        am, as_ = (actions[:,d].mean(), actions[:,d].std()) if d < actions.shape[1] else (0,0)
        print(f"dim[{d}]     {sm:10.4f} {ss:10.4f}  {am:11.4f} {as_:10.4f}")

    print(f"\nstate  range: [{states.min(axis=0).round(3).tolist()}] -> [{states.max(axis=0).round(3).tolist()}]")
    print(f"action range: [{actions.min(axis=0).round(4).tolist()}] -> [{actions.max(axis=0).round(4).tolist()}]")

    # Task distribution
    task_ids = set()
    for i in range(table.num_rows):
        task_ids.add(table.column("task_index")[i].as_py())
    tasks_data = [json.loads(l) for l in (DATASET_DIR / "meta/tasks.jsonl").read_text().strip().split("\n")]
    print(f"\n=== Tasks ({len(task_ids)} unique) ===")
    for t in tasks_data:
        if t["task_index"] in task_ids:
            print(f"  [{t['task_index']}] {t['task']}")

except ImportError as e:
    print(f"Missing: {e}\nRun: pip install pyarrow numpy")
    sys.exit(1)
