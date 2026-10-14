"""Read actual paired process outputs; do not weaken failed quality gates."""
import argparse
import json
import pathlib

import numpy as np
import torch

from common import write_json, sha


def pair(a, b):
    x = json.loads((a / 'cases.json').read_text())
    y = json.loads((b / 'cases.json').read_text())
    assert len(x) == len(y) == 128
    for p, q in zip(x, y):
        assert tuple(p[k] for k in ['task', 'seed', 'step', 'file_sha256']) == tuple(q[k] for k in ['task', 'seed', 'step', 'file_sha256'])
    return np.asarray([r['env_actions'] for r in x]), np.asarray([r['env_actions'] for r in y])


def quality(a, b):
    d = np.abs(a[:, :, :6] - b[:, :, :6]) / 2
    confident = np.abs(a[:, :, 6]) > .2
    disagreement = (np.sign(a[:, :, 6][confident]) != np.sign(b[:, :, 6][confident])).sum()
    clips_a = (np.abs(a) > 1).sum(axis=(0,1))
    clips_b = (np.abs(b) > 1).sum(axis=(0,1))
    report = {'motion_mae': float(d.mean()), 'motion_p99_abs': float(np.quantile(d,.99)),
        'motion_max_abs': float(d.max()), 'gripper_disagreements': int(disagreement),
        'confident_gripper_count': int(confident.sum()),
        'gripper_disagreement_fraction': float(disagreement/max(1, confident.sum())),
        'clipping_rate_increase': float((clips_b.sum()-clips_a.sum())/a.size),
        'reference_clips_per_dimension': clips_a.tolist(), 'candidate_clips_per_dimension': clips_b.tolist()}
    report['original_gates_passed'] = report['motion_mae'] <= .01 and report['motion_p99_abs'] <= .05 and \
        report['gripper_disagreement_fraction'] <= .005 and report['clipping_rate_increase'] <= .01
    return report


def main(out, reference, candidate, reload, old_float_cases):
    assert not out.exists()
    a, b = pair(reference, candidate)
    _, c = pair(candidate, reload)
    first = torch.load(candidate / 'outputs.pt', weights_only=True)
    second = torch.load(reload / 'outputs.pt', weights_only=True)
    changes = {k: float((first[k] - second[k]).abs().max()) for k in first}
    reload_summary = {'passed': changes['actions'] <= 1e-6, 'max_abs_internal_actions': changes['actions'],
        'max_abs_velocity_fields': changes['velocity_fields'], 'max_abs_env_actions': float(np.abs(b-c).max()),
        'same_engine_bundle_sha256': sha(candidate / 'runtime.json') == sha(reload / 'runtime.json'),
        'different_process_pids': json.loads((candidate/'summary.json').read_text())['pid'] !=
                                  json.loads((reload/'summary.json').read_text())['pid']}
    assert reload_summary['same_engine_bundle_sha256'] and reload_summary['different_process_pids']
    old = json.loads(old_float_cases.read_text())
    old_reference = np.asarray([r['reference_env_actions'] for r in old])
    old_trt = np.asarray([r['tensorrt_env_actions'] for r in old])
    for p, q in zip(old, json.loads((reference/'cases.json').read_text())):
        assert tuple(p[k] for k in ['task','seed','step','file_sha256']) == tuple(q[k] for k in ['task','seed','step','file_sha256'])
    baseline_replay = float(np.abs(old_trt-a).max())
    report = {'algorithm': json.loads((candidate/'summary.json').read_text())['algorithm'],
        'observations': 128, 'same_backend_TensorRT_quantization_error': quality(a,b),
        'cumulative_original_Torch_to_TensorRT_quantized_error': quality(old_reference,b),
        'independent_process_reload': reload_summary,
        'original_T1_float_case_replay_max_abs_env_actions': baseline_replay,
        'original_T1_float_case_replay_passed': baseline_replay <= 1e-6,
        'T1_boundary_drift_temporarily_allowed_by_user': True,
        'new_quantization_errors_automatically_waived': False,
        '4070_8GB_accepted': False, 'closed_loop_accepted': False}
    report['T3_reload_passed'] = reload_summary['passed'] and baseline_replay <= 1e-6
    write_json(out, report)
    assert report['T3_reload_passed'], report
    print('T3 RELOAD PASS; QUALITY', report['algorithm'], report['same_backend_TensorRT_quantization_error'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--reference', type=pathlib.Path, required=True)
    p.add_argument('--candidate', type=pathlib.Path, required=True)
    p.add_argument('--reload', type=pathlib.Path, required=True)
    p.add_argument('--old-float-cases', type=pathlib.Path, required=True)
    a = p.parse_args()
    main(a.out, a.reference, a.candidate, a.reload, a.old_float_cases)
