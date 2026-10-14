"""Freeze a new release; independent verification never rewrites its manifest."""
import argparse
import datetime
import hashlib
import json
import pathlib

VERSION = 'trt_t0_t2_20261007_r1'


def digest(path, cache):
    stat = path.stat()
    key = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in cache:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(block)
        cache[key] = h.hexdigest()
    return cache[key]


def retained(root, cache):
    plan = root / 'versions/w4_plan_20261007_r4/manifest.json'
    assert digest(plan, cache) == 'b2d47a077017110825fcaafc6e454876f32ffcbc784eec62100a85732bbba745'
    old = json.loads(plan.read_text())
    dependencies = {}
    for field in ['files_relative_to_6_model_inference', 'retained_asset_metadata_files_relative_to_6_model_inference']:
        for name, expected in old[field].items():
            assert digest(root / name, cache) == expected, name
            dependencies[name] = expected
    dependencies[str(plan.relative_to(root))] = digest(plan, cache)
    for folder in sorted((root / 'weights').iterdir()):
        if not folder.is_dir():
            continue
        manifest = folder / 'bundle_manifest.json'
        if not manifest.exists():
            continue
        for name, expected in json.loads(manifest.read_text())['files'].items():
            p = folder / name
            assert digest(p, cache) == expected, p
            dependencies[str(p.relative_to(root))] = expected
    return dependencies


def build(root, manifest):
    assert not manifest.exists(), manifest
    manifest.parent.mkdir(parents=True, exist_ok=False)
    cache = {}
    dependencies = retained(root, cache)
    files = [root / '当前实施索引_20261007_r1.md']
    for folder in [root / 'versions' / VERSION, root / 'runs' / VERSION]:
        files.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    entries = {}
    for i, path in enumerate(sorted(set(files))):
        if path.suffix == '.py':
            compile(path.read_text(), str(path), 'exec')
        if path.suffix == '.json':
            json.loads(path.read_text())
        entries[str(path.relative_to(root))] = {'bytes': path.stat().st_size, 'sha256': digest(path, cache)}
        if (i + 1) % 500 == 0:
            print('FREEZE HASHED', i + 1, flush=True)
    document = {
        'version': VERSION, 'frozen_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'files_frozen_with_failed_quality_gates',
        'manifest_excludes_itself_and_verification_receipts': True,
        'files_relative_to_6_model_inference': entries,
        'unchanged_frozen_dependencies': dependencies,
        'acceptance': {'T0': 'passed', 'T1': 'failed_extended_clipping_gate',
                       'T2_INT4_exports': 'passed', 'AWQ_offline_policy': 'passed',
                       'GPTQ_offline_policy': 'failed', 'RTN_offline_policy': 'failed',
                       'full_quantized_engine': 'not_executed', '4070_8GB': 'not_executed',
                       'closed_loop': 'not_executed', 'production_release': 'held'},
    }
    with manifest.open('x') as f:
        json.dump(document, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print('MANIFEST CREATED', manifest, 'files', len(entries), 'sha256', digest(manifest, cache), flush=True)


def verify(root, manifest, receipt, partial):
    assert not receipt.exists(), receipt
    document = json.loads(manifest.read_text())
    cache = {}
    failures, absent = [], []
    entries = dict(document['files_relative_to_6_model_inference'])
    for name, expected in document['unchanged_frozen_dependencies'].items():
        entries[name] = {'sha256': expected}
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
                   'present_files_verified': len(entries) - len(absent), 'mismatches': failures,
                   'absent_files': absent, 'deployment_accepted': False}, f, indent=2)
        f.write('\n')
    assert passed, {'mismatches': failures, 'absent_count': len(absent)}
    print('FILE VERIFICATION PASS', len(entries) - len(absent), 'present;', len(absent), 'absent', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=pathlib.Path, required=True)
    parser.add_argument('--manifest', type=pathlib.Path, required=True)
    parser.add_argument('--verify-receipt', type=pathlib.Path)
    parser.add_argument('--partial-local', action='store_true')
    args = parser.parse_args()
    if args.verify_receipt:
        verify(args.root, args.manifest, args.verify_receipt, args.partial_local)
    else:
        assert not args.partial_local
        build(args.root, args.manifest)
