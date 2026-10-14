"""Read-only early evidence check while other episodes continue."""
import json
from pathlib import Path
import sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path('/control')
sys.path.insert(0, str(ROOT / 'versions/eco65_zhixing_20261007_r4'))
from audit import audit_episode

RUN = ROOT / 'runs/eco65_zhixing_spatial10_20261007_r1'
OUT = ROOT / 'reports/eco65_zhixing_20261007_r1'
reports = []
for task in [0, 1]:
    episode = RUN / f'task{task:02d}_seed0'
    reports.append(audit_episode(episode))
    reader = imageio.get_reader(episode / 'env/dual_camera.mp4')
    frames = [reader.get_data(i) for i in [0, 110, 220]]
    reader.close()
    Image.fromarray(np.concatenate(frames, axis=0)).save(OUT / f'task{task:02d}_early_preview.png')
with (OUT / 'early_two_audit.json').open('x') as file:
    json.dump(reports, file, ensure_ascii=False, indent=2)
print(json.dumps([dict(task=r['task'], audit_passed=r['audit_passed'],
                       native_success=r['native_success'], steps=r['steps']) for r in reports]))
