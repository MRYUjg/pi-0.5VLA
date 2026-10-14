"""Create a new immutable T4 manifest; verify all inherited frozen artifacts."""
import argparse
import datetime
import hashlib
import json
import pathlib

VERSION = 'trt_t4_20261007_r1'
PREVIOUS = 'releases/trt_t3_20261007_r1/manifest.json'
PREVIOUS_SHA = '016814fb9a9eaba7d398bb9fa7e2b3952de2995d8767b215c76ee7ed10bcbe17'


def digest(path, cache):
    s = path.stat()
    key = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        cache[key] = h.hexdigest()
    return cache[key]


def build(root, manifest):
    assert not manifest.exists()
    cache = {}
    assert digest(root / PREVIOUS, cache) == PREVIOUS_SHA
    previous = json.loads((root / PREVIOUS).read_text())
    dependencies = {k: v['sha256'] for k, v in previous['files_relative_to_6_model_inference'].items()}
    dependencies.update(previous['unchanged_frozen_dependencies'])
    dependencies[PREVIOUS] = PREVIOUS_SHA
    run = root / 'runs' / VERSION
    report = json.loads((run / 'report_r1/summary.json').read_text())
    assert report['all_120_episodes_complete']
    assert json.loads((run / 'dependencies_r1/verified.json').read_text())['passed']
    assert json.loads((run / 'artifact_audit_r1/verified.json').read_text())['passed']
    files = [root / '当前实施索引_20261007_r3.md']
    for folder in [root / 'versions' / VERSION, run]:
        files += [p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    entries = {}
    for path in sorted(set(files)):
        if path.suffix == '.py':
            compile(path.read_text(), str(path), 'exec')
        if path.suffix == '.json':
            json.loads(path.read_text())
        entries[str(path.relative_to(root))] = {'sha256': digest(path, cache), 'bytes': path.stat().st_size}
    document = {'version': VERSION, 'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'T4_tests_complete_individual_gates_recorded_without_waiver',
        'files_relative_to_6_model_inference': entries, 'unchanged_frozen_dependencies': dependencies,
        'previous_release_manifest_sha256': PREVIOUS_SHA,
        'manifest_excludes_itself_and_verification_receipts': True,
        'acceptance': {'T1': 'failed_original_gate_temporary_user_exception',
            'T3_AWQ_offline': 'passed_retained', 'T3_GPTQ_offline': 'failed_retained',
            'T4_test_completion': '120_matched_development_episodes_and_4_single_client_latency_probes',
            'T4_initial_conditions_aligned': report['initial_condition_alignment_passed'],
            'T4_AWQ_quality_candidate_qualified': report['variants']['awq']['T4_quality_candidate_qualified'],
            'T4_GPTQ_quality_candidate_qualified': report['variants']['gptq']['T4_quality_candidate_qualified'],
            'all_124_videos_fully_decoded_and_initial_arrays_read_back': True,
            'camera_to_action_200ms_gate_per_variant': {
                v: s['provisional_camera_to_dispatch_p99_200ms_passed']
                for v, s in report['variants'].items()},
            'current_4090_1000_requests_per_variant': 'completed_no_OOM_or_memory_growth',
            'T5_actual_4070_8GB': 'not_executed', 'two_hour_continuous_run': 'not_executed',
            'T6_formal_200_trials': 'not_executed', 'T7_robot': 'not_executed',
            'production_release': 'not_accepted'}}
    manifest.parent.mkdir(parents=True, exist_ok=False)
    with manifest.open('x') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print('T4 MANIFEST CREATED', len(entries), 'files', digest(manifest, cache), flush=True)


def verify(root, manifest, receipt, partial):
    assert not receipt.exists()
    doc = json.loads(manifest.read_text())
    entries = dict(doc['files_relative_to_6_model_inference'])
    entries.update({k: {'sha256': v} for k, v in doc['unchanged_frozen_dependencies'].items()})
    cache, absent, failures = {}, [], []
    for i, (name, expected) in enumerate(entries.items()):
        path = root / name
        if not path.exists():
            absent.append(name)
            continue
        if digest(path, cache) != expected['sha256'] or ('bytes' in expected and path.stat().st_size != expected['bytes']):
            failures.append(name)
        if (i + 1) % 500 == 0:
            print('VERIFIED', i + 1, flush=True)
    passed = not failures and (partial or not absent)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with receipt.open('x') as f:
        json.dump({'passed': passed, 'verification_mode': 'partial_local_metadata_mirror' if partial else 'complete_server_artifacts',
            'manifest_sha256': digest(manifest, cache), 'listed_files': len(entries),
            'present_files_verified': len(entries) - len(absent), 'mismatches': failures, 'absent_files': absent,
            'deployment_accepted': False}, f, indent=2)
        f.write('\n')
    assert passed, (failures, len(absent))
    print('T4 FILE VERIFICATION PASS', len(entries) - len(absent), 'present;', len(absent), 'absent', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=pathlib.Path, required=True)
    p.add_argument('--manifest', type=pathlib.Path, required=True)
    p.add_argument('--verify-receipt', type=pathlib.Path)
    p.add_argument('--partial-local', action='store_true')
    a = p.parse_args()
    if a.verify_receipt:
        verify(a.root, a.manifest, a.verify_receipt, a.partial_local)
    else:
        assert not a.partial_local
        build(a.root, a.manifest)
