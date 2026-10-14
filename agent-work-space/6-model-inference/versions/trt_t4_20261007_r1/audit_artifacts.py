"""Read back all saved frames, initial observations and executed action records."""
import argparse
import concurrent.futures
import hashlib
import json
import pathlib

import imageio.v2 as imageio
import numpy as np


VARIANTS = ['original_torch', 'trt_float', 'awq', 'gptq']


def inspect(path):
    body = json.loads(path.read_text())
    ep = body['episode']
    initial = np.load(path.with_name(path.stem + '_initialobs.npz'), allow_pickle=False)
    settled = np.load(path.with_name(path.stem + '_settled.npy'), allow_pickle=False)
    assert np.isfinite(settled).all() and np.isfinite(initial['state']).all()
    assert hashlib.sha256(settled.tobytes()).hexdigest() == ep['initialization']['simulation_state_sha256']
    pixels_and_state = initial['image'].tobytes() + initial['wrist_image'].tobytes() + initial['state'].tobytes()
    assert hashlib.sha256(pixels_and_state).hexdigest() == ep['initialization']['initial_observation_sha256']
    assert len(body['chunks']) == (len(body['steps']) + 4) // 5
    for i, chunk in enumerate(body['chunks']):
        assert chunk['step'] == i * 5 and chunk['noise_seed'] == ep['seed'] * 1000 + i
        assert np.isfinite(chunk['raw_actions']).all()
    for i, step in enumerate(body['steps']):
        assert step['step'] == i
        assert step['raw_action'] == body['chunks'][i // 5]['raw_actions'][i % 5]
        assert np.isfinite(step['eef_pos']).all() and np.isfinite(step['gripper_qpos']).all()
        assert not step['success'] or i == len(body['steps']) - 1
    assert body['steps'][-1]['success'] == ep['success']
    if not ep['success']:
        assert len(body['steps']) == 220
    reader = imageio.get_reader(path.parent / ep['video'], input_params=['-threads', '1'])
    frames = 0
    try:
        for frame in reader:
            assert frame.shape == (256, 512, 3) and frame.dtype == np.uint8
            frames += 1
    finally:
        reader.close()
    assert frames == ep['video_frames'] == ep['steps'] + 1
    return {'path': str(path), 'variant': ep['variant'], 'task': ep['task'],
            'init_index': ep['init_index'], 'success': ep['success'],
            'fully_decoded_video_frames': frames, 'initial_arrays_and_action_sequence_passed': True}


def main(root, out):
    out.mkdir(exist_ok=False)
    paths = []
    for variant in VARIANTS:
        paths += [root / f'closedloop_{variant}_r1/task{task:02d}/init{index:02d}.json'
                  for task in range(10) for index in [0, 1, 2]]
        paths.append(root / f'latency_{variant}_r1/episode/init00.json')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(inspect, paths))
    assert len(rows) == 124
    result = {'passed': True, 'primary_episodes': 120, 'latency_probe_episodes': 4,
              'full_video_decodes': len(rows),
              'decoded_frames': sum(row['fully_decoded_video_frames'] for row in rows),
              'scope': 'readback audit only; no changes to controls or success criteria', 'episodes': rows}
    with (out / 'verified.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print('ARTIFACT READBACK PASS', len(rows), 'videos;', result['decoded_frames'], 'frames', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=pathlib.Path, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    a = p.parse_args()
    main(a.root, a.out)
