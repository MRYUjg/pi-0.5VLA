"""Decode both retained videos and build an explicit per-task video manifest."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def verify(path, expected_frames, expected_shape):
    reader = imageio.get_reader(path)
    metadata = reader.get_meta_data()
    frames = 0
    first = middle = last = None
    valid_shapes = True
    for frame in reader:
        valid_shapes &= frame.shape == expected_shape
        if first is None:
            first = frame.copy()
        if frames == expected_frames // 2:
            middle = frame.copy()
        last = frame.copy()
        frames += 1
    reader.close()
    fps = float(metadata['fps'])
    passed = bool(frames == expected_frames and valid_shapes and np.isclose(fps, 20))
    return dict(file=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                frames=frames, fps=fps, seconds=frames / fps,
                expected_frames=expected_frames, shape=list(expected_shape), codec=metadata.get('codec'),
                all_frames_decoded=passed, image_changes=bool(np.any(first != last))), (first, middle, last)


parser = argparse.ArgumentParser()
parser.add_argument('--run', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
rows = []
for episode in sorted(args.run.glob('task*_seed*')):
    if not episode.is_dir():
        continue
    result = json.loads((episode / 'result.json').read_text())
    registration = json.loads((episode / 'env/registration.json').read_text())
    dual, samples = verify(episode / 'env/dual_camera.mp4', result['steps'] + 1, (256, 512, 3))
    native, _ = verify(episode / 'artifacts/episode.mp4', result['steps'], (256, 256, 3))
    Image.fromarray(np.concatenate(samples, axis=0)).save(args.output / (episode.name + '_sample.png'))
    rows.append(dict(task=result['task'], seed=result['seed'], language=registration['language'],
                     native_success=result['native_success'], steps=result['steps'],
                     dual_camera=dual, rpent_episode=native))
report = dict(episodes=rows, tasks=len(rows), native_successes=sum(r['native_success'] for r in rows),
              MP4_count=2 * len(rows),
              all_videos_decoded=bool(rows) and all(r[k]['all_frames_decoded'] for r in rows for k in ['dual_camera', 'rpent_episode']))
with (args.output / 'video_manifest.json').open('x') as file:
    json.dump(report, file, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in report.items() if k != 'episodes'}))
