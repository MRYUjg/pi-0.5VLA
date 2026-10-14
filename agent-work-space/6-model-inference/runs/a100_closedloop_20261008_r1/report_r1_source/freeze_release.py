"""Hash this run and its used inherited files; freeze only the new version."""
import datetime
import hashlib
import json
import pathlib
import subprocess
import time

from resources import IMAGE, ROOT, RUN, VERSION

PARENTS = {
    'trt_t0_t2_20261007_r1': '4f359c7b939e10c7eebba4213f4156aef7dc02874101c9919f9ad0a3222c23ee',
    'trt_t3_20261007_r1': '016814fb9a9eaba7d398bb9fa7e2b3952de2995d8767b215c76ee7ed10bcbe17',
    'trt_t4_20261007_r1': 'b9875e01fb7820d4c1e49b289421e3daa130ebe3b309001d374198232375c1c0',
    'trt_full4_20261007_r1': '63b7561e7be672d83069c3c883a0142efee583f2a26088d0e4e94768ad325cfc',
}
cache = {}


def signature(path):
    st = path.stat()
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def digest(path):
    path = pathlib.Path(path)
    before = signature(path)
    if path in cache:
        assert cache[path][0] == before, str(path)
        return cache[path][1]
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    assert signature(path) == before, str(path)
    value = h.hexdigest()
    cache[path] = (before, value)
    return value


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def main():
    started = time.monotonic()
    report = json.loads((RUN / 'report_r1/summary.json').read_text())
    assert report['all_150_episodes_complete'] and report['initial_condition_alignment_passed']
    assert json.loads((RUN / 'artifact_audit_r1/verified.json').read_text())['passed']
    assert json.loads((RUN / 'closedloop_sequence_r1_exit.json').read_text())['exit_code'] == 0
    assert not json.loads((RUN / 'build_container_shutdown.json').read_text())['state']['Running']
    release = ROOT / 'releases' / VERSION
    release.mkdir(exist_ok=False)
    old_entries = {}
    dependencies = {}
    for name, expected in PARENTS.items():
        path = ROOT / 'releases' / name / 'manifest.json'
        assert digest(path) == expected, name
        dependencies[str(path.relative_to(ROOT))] = expected
        body = json.loads(path.read_text())
        entries = dict(body['unchanged_frozen_dependencies'])
        entries.update({k: v['sha256'] for k, v in body['files_relative_to_6_model_inference'].items()})
        for key, value in entries.items():
            assert key not in old_entries or old_entries[key] == value, key
            old_entries[key] = value

    def inherited(path, expected=None):
        path = pathlib.Path(path)
        key = str(path.relative_to(ROOT))
        pinned = old_entries[key]
        if expected is not None:
            assert pinned == expected, key
        assert digest(path) == pinned, key
        dependencies[key] = pinned

    for path in [
        'versions/trt_t3_20261007_r1/common.py',
        'versions/trt_t3_20261007_r1/runtime.py',
        'versions/trt_t3_20261007_r1/compare.py',
        'versions/trt_t4_20261007_r1/closedloop.py',
        'versions/trt_t4_20261007_r1/preflight.py',
        'versions/trt_t4_20261007_r1/verify_dependencies.py',
        'versions/trt_t4_20261007_r1/audit_artifacts.py',
        'runs/trt_t3_20261007_r1/environment_r1/runtime.json',
        'runs/trt_t3_20261007_r1/float_infer_r1/cases.json',
        'runs/trt_t4_20261007_r1/preflight_r1/scenes.json',
        'runs/trt_t4_20261007_r1/preflight_r1/controller.json',
        'runs/trt_t4_20261007_r1/preflight_r1/environment.json',
        'runs/trt_t0_t2_20261007_r1/calibration_r1/pilot.json',
        'weights/pi05_libero_bf16_values_r1/assets/physical-intelligence/libero/norm_stats.json',
    ]:
        inherited(ROOT / path)
    for row in json.loads((ROOT / 'runs/trt_t0_t2_20261007_r1/calibration_r1/pilot.json').read_text()):
        inherited(ROOT / pathlib.Path(row['path']).relative_to('/work'), row['file_sha256'])
    for task in range(10):
        inherited(ROOT / f'runs/trt_t0_t2_20261007_r1/panda_calibration_r1/task{task:02d}/registration.json')
    build_sequence = json.loads((RUN / 'build_sequence_summary.json').read_text())
    assert build_sequence['all_nine_completed']
    for row in build_sequence['results']:
        cmd = row['command']
        source = ROOT / pathlib.Path(cmd[cmd.index('--source') + 1]).relative_to('/work')
        registration = json.loads((RUN / row['name'] / 'registration.json').read_text())
        inherited(source, registration['source_ONNX_sha256'])
        for name, expected in registration['external_data_sha256'].items():
            inherited(source.parent / name, expected)
    print('USED INHERITED FILES VERIFIED', len(dependencies), flush=True)

    paths = sorted(p for folder in [ROOT / 'versions' / VERSION, RUN]
                   for p in folder.rglob('*') if p.is_file())
    paths += [ROOT / '当前实施索引_20261008_r1.md']
    assert not any(p.is_symlink() for p in paths)
    files = {str(p.relative_to(ROOT)): {'sha256': digest(p), 'bytes': p.stat().st_size}
             for p in paths}
    manifest = {
        'version': VERSION,
        'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'experiment_complete_evidence_frozen_not_production_accepted',
        'files_relative_to_6_model_inference': files,
        'unchanged_frozen_dependencies': dependencies,
        'parent_manifest_sha256': PARENTS,
        'inherited_verification_scope': 'Only actually used inherited files are rehashed; historical files not used by this run remain behind pinned parent manifests.',
        'runtime_code_package_verification': 'dependencies_r1/verified.json and identical preflight environment.json; images are pinned by digest.',
        'manifest_excludes_itself_and_verification_receipts': True,
        'acceptance': {k: report[k] for k in ['production_release_accepted', 'formal_200_tested',
                         'two_hour_tested', 'actual_8GB_device_tested', 'real_robot_tested',
                         'prior_4090_quality_failures_not_waived']},
    }
    write(release / 'manifest.json', manifest)
    # Container-written files are root-owned; chmod runs in a CPU-only container
    # with only this new run writable. Never chmod an old release or dependency.
    chmod_command = ['docker', 'run', '--rm', '--network', 'none', '-v', str(RUN) + ':/freeze:rw',
                     '--entrypoint', '/.venv/bin/python', IMAGE, '-c',
                     "import pathlib; [p.chmod(0o444) for p in pathlib.Path('/freeze').rglob('*') if p.is_file()]"]
    chmod_exit = subprocess.run(chmod_command).returncode
    assert chmod_exit == 0
    for path in paths:
        if RUN not in path.parents:
            path.chmod(0o444)
    (release / 'manifest.json').chmod(0o444)
    # This verifies the files stayed identical after hashing and chmod. Cached
    # hashes are reused only when inode, bytes and nanosecond mtime stay equal.
    for path in paths:
        key = str(path.relative_to(ROOT))
        assert digest(path) == files[key]['sha256']
        assert path.stat().st_mode & 0o222 == 0, str(path)
    for key, expected in dependencies.items():
        assert digest(ROOT / key) == expected
    write(release / 'server_verify_r1.json', {
        'passed': True, 'manifest_sha256': digest(release / 'manifest.json'),
        'current_files_verified': len(files), 'used_inherited_files_verified': len(dependencies),
        'all_current_files_readonly': True, 'chmod_command': chmod_command,
        'chmod_exit_code': chmod_exit, 'elapsed_seconds': time.monotonic() - started,
        'method': 'Each file receives one complete SHA256 content read; stable inode/size/mtime checks permit reuse within this verification process. No claim of a second independent content read.',
        'historical_unused_dependencies_rehashed': False,
        'evidence_freeze_is_not_deployment_acceptance': True,
    })
    (release / 'server_verify_r1.json').chmod(0o444)
    print('A100 NEW RELEASE FROZEN', len(files), 'current files;', len(dependencies), 'used inherited files', flush=True)


if __name__ == '__main__':
    main()
