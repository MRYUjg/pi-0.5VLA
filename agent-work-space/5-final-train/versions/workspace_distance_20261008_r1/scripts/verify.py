"""Independent checks of paired development records, actions, kinematics and videos."""
import hashlib
import json
import pathlib
import subprocess
import sys

import h5py
import imageio_ffmpeg
import numpy as np

V = pathlib.Path(__file__).resolve().parents[1]
ROOT = V.parents[1]


def main():
    out = ROOT / 'runs' / 'workspace_distance_20261008_r1'
    jobs = json.loads((out / 'attempts.json').read_text())
    errors, videos, pairs = [], [], {}
    groups = {}
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for job in jobs:
        path = ROOT / job['path']
        execution = json.loads((path / 'execution.json').read_text())
        result = execution.get('result', {})
        if execution['process_exit'] != 0 or not result:
            errors.append('process failure: ' + str(path))
            continue
        steps = [json.loads(x) for x in (path / 'steps.jsonl').read_text().splitlines()]
        native = [s['step'] for s in steps if s['native_success']]
        first = native[0] if native else None
        if first != result.get('first_success_step') or len(steps) != result['steps']:
            errors.append('step/native accounting: ' + str(path))
        valid = result['initialization_pass'] and all(s['finite'] and not any(s['physics_warnings']) for s in steps)
        if result['primary_success'] != bool(valid and first is not None and first <= 220):
            errors.append('220 accounting: ' + str(path))
        if len(steps) > 1000:
            errors.append('undeclared extension: ' + str(path))
        with h5py.File(path / 'trajectory.h5', 'r') as f:
            if f['preparation_actions'].shape != (120, 7):
                errors.append('preparation count: ' + str(path))
            queries = f.get('queries', {})
            if len(queries) != result.get('queries', 0):
                errors.append('query count: ' + str(path))
            for g in queries.values():
                action = g['actions'][...]
                start = int(g.attrs['step'])
                if action.shape != (10, 7) or not np.isfinite(action).all():
                    errors.append('action shape/finite: ' + str(path))
                for i in range(min(5, len(steps)-start)):
                    row = steps[start+i]
                    if not np.array_equal(action[i], row['raw_action']) or not np.array_equal(np.clip(action[i], -1, 1), row['executed_action']):
                        errors.append('action execution: ' + str(path))
                if g['image'].shape != (256, 256, 3) or g['wrist_image'].shape != (256, 256, 3):
                    errors.append('image shape: ' + str(path))
        geometry = result['settled_geometry']
        if abs(geometry['base_world_m'][0] - (-.66 + job['offset_m'])) > 1e-8:
            errors.append('base translation: ' + str(path))
        pairs.setdefault((job['task'], job['seed']), []).append(result['reset_object_hash'])
        mp4 = path / 'rollout.mp4'
        frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(mp4))
        decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(mp4), '-f', 'null', '-'], capture_output=True)
        sha = hashlib.sha256(mp4.read_bytes()).hexdigest()
        if decoded.returncode != 0 or frames != result['video_frames'] or sha != result['video_sha256']:
            errors.append('MP4 decode/frame/hash: ' + str(path))
        row = {**job, 'video': str(mp4.relative_to(ROOT)), 'sha256': sha, 'frames': frames, 'seconds': seconds,
               'initialization_pass': result['initialization_pass'], 'first_success_step': first,
               'successes_by_budget': {str(n): bool(valid and first is not None and first <= n) for n in [220, 650, 1000]},
               'settled_geometry': geometry, 'minimum_tcp_target_distance_m': min([s['geometry']['tcp_target_distance_m'] for s in steps] or [geometry['tcp_target_distance_m']]),
               'maximum_elbow_extension_ratio': max([s['geometry']['elbow_extension_ratio'] for s in steps] or [geometry['elbow_extension_ratio']]),
               'minimum_jacobian_position_smin': min([s['geometry']['jacobian_position_smin_m_per_rad'] for s in steps] or [geometry['jacobian_position_smin_m_per_rad']]),
               'decode_pass': decoded.returncode == 0}
        videos.append(row)
        group = groups.setdefault(str(job['offset_m']), {'attempts': 0, 'initialization_rejected': 0, 'successes_by_budget': {str(n): 0 for n in [220, 650, 1000]}})
        group['attempts'] += 1
        group['initialization_rejected'] += not result['initialization_pass']
        for n, success in row['successes_by_budget'].items():
            group['successes_by_budget'][n] += success
    if len(jobs) != 18 or len(videos) != 18:
        errors.append('expected all18 registered development attempts')
    if any(len(values) != 3 or len(set(values)) != 1 for values in pairs.values()) or len(pairs) != 6:
        errors.append('paired object reset hash differs')
    report = {'verification_pass': not errors, 'errors': errors, 'complete': len(videos) == 18,
              'scope': 'paired validation development diagnosis; not adaptation acceptance',
              'groups': groups, 'videos': videos, 'paired_native_scenes': {'pairs': len(pairs), 'all_equal': all(len(set(v)) == 1 for v in pairs.values())}}
    with (out / 'verified_results.json').open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print('DISTANCE VERIFICATION', not errors, 'VIDEOS', len(videos), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
