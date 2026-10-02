#!/usr/bin/env python3
"""Versioned N0 source/asset snapshots. Default action verifies; never blesses drift."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tarfile
import tempfile

NODE_REL = pathlib.Path('agent-work-space/zhixing_adaptation_20260927')
REPO = pathlib.Path(__file__).resolve().parents[3]
ALLOWED = {'.py', '.json', '.xml', '.stl', '.obj', '.mtl', '.msh', '.dae',
           '.png', '.jpg', '.jpeg', '.txt', '.yaml', '.yml', '.toml', '.lock',
           '.bddl', '.init', '.pruned_init', '.sh', '.in'}
EXCLUDED = {'__pycache__', '.git', '.venv', 'datasets', 'logs'}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def catalog(root, old, retired=()):
    paths = {v['path'] for v in old['sources'].values()} - set(retired)
    for tree in ('robosuite', 'third_party/libero/libero/libero'):
        for p in (root / tree).rglob('*'):
            rel = p.relative_to(root)
            if (p.is_file() and not p.is_symlink() and not EXCLUDED.intersection(rel.parts)
                    and p.suffix.lower() in ALLOWED and not p.name.endswith('.bak')):
                paths.add(rel.as_posix())
    node = root / NODE_REL
    for p in (node / 'scripts').iterdir():
        if p.suffix in {'.py', '.sh'} and not p.name.startswith('ssh_'):
            paths.add(p.relative_to(root).as_posix())
    for p in (node / 'configs').iterdir():
        if p.suffix in {'.json', '.md'}:
            paths.add(p.relative_to(root).as_posix())
    for rel in ['pyproject.toml', 'uv.lock', 'examples/libero/Dockerfile',
                'examples/libero/compose.yml', 'examples/libero/requirements.in',
                'examples/libero/requirements.txt', 'examples/libero/build-constraints.txt',
                'examples/libero/convert_libero_data_to_lerobot.py',
                str(NODE_REL / '当前版本与文件索引.md'), str(NODE_REL / '阶段进展.md')]:
        if not (root / rel).is_file():
            raise RuntimeError('Required file missing: ' + rel)
        paths.add(rel)
    # Node-level records (plan, node design, index, progress and audit records) are part
    # of the frozen baseline, so that an audit conclusion cannot drift away from the code
    # revision it was written against.
    for p in sorted((root / NODE_REL).glob('*.md')):
        if p.is_file() and not p.is_symlink():
            paths.add(p.relative_to(root).as_posix())
    keys = {v['path']: k for k, v in old['sources'].items()}
    result = {}
    for rel in sorted(paths):
        p = root / rel
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root.resolve()):
            raise RuntimeError('Invalid source: ' + rel)
        result[keys.get(rel, 'file::' + rel)] = {
            'path': rel, 'sha256': digest(p), 'bytes': p.stat().st_size,
            'mode': p.stat().st_mode & 0o777,
        }
    return result


def capture_environment(root):
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', 'libero'], text=True))[0]
    # Do not capture container environment variables or command lines, which may contain credentials.
    probe = ('import importlib.metadata as m,json,platform; '
             'print(json.dumps({"python":platform.python_version(),'
             '"packages":{d.metadata["Name"]:d.version for d in m.distributions() if d.metadata["Name"]}}))')
    proc = subprocess.run(['docker', 'run', '--rm', '--network', 'none', '--entrypoint',
                           '/.venv/bin/python', image['Id'], '-c', probe],
                          check=True, capture_output=True, text=True, timeout=60)
    deps = json.loads(proc.stdout)
    gpu_command = shutil.which('nvidia-smi')
    gpu_result = subprocess.run(
        [gpu_command, '--query-gpu=name,memory.total', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=15) if gpu_command else None
    gpu_available = bool(gpu_result is not None and gpu_result.returncode == 0
                         and gpu_result.stdout.strip())
    gpu = gpu_result.stdout.strip() if gpu_available else 'GPU unavailable; CPU verification supported'
    meminfo = pathlib.Path('/proc/meminfo')
    memory_kib = None
    if meminfo.exists():
        memory_kib = next((int(line.split()[1]) for line in meminfo.read_text().splitlines()
                           if line.startswith('MemTotal:')), None)
    return {'captured_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'image': {k: image.get(k) for k in ['Id', 'RepoDigests', 'Created', 'Architecture', 'Os']},
            'container_dependencies': deps, 'gpu': gpu,
            'hardware': {'logical_cpu_count': os.cpu_count(), 'memory_total_kib': memory_kib,
                         'gpu_available': gpu_available,
                         'gpu_probe_returncode': None if gpu_result is None else gpu_result.returncode,
                         'gpu_probe_note': ('nvidia-smi not installed' if gpu_result is None else
                                            (gpu_result.stderr or gpu_result.stdout).strip()[:600])},
            'scope': 'Source/assets snapshot plus image identity. Image layers, policy weights and datasets are not exported.'}


def verify(root, version_dir, check_archive=False, check_aliases=False):
    issues = []
    seal = json.loads((version_dir / 'integrity.json').read_text())
    for name, expected in seal['files'].items():
        p = version_dir / name
        if not p.is_file() or digest(p) != expected:
            issues.append('sealed_file_mismatch:' + name)
    if issues:
        return {'verdict': 'FAIL', 'issues': issues}
    manifest = json.loads((version_dir / 'manifest.json').read_text())
    paths = {}
    for item in manifest['sources'].values():
        rel = item['path']; p = root / rel; paths[rel] = item
        if not p.is_file() or p.is_symlink() or digest(p) != item['sha256']:
            issues.append('source_mismatch:' + rel)
    # The catalog also detects newly added managed files, not only changed existing files.
    actual_catalog = catalog(root, manifest)
    extra = {v['path'] for v in actual_catalog.values()} - set(paths)
    issues.extend('unfrozen_source:' + p for p in sorted(extra))
    restored = 0
    if check_archive:
        with tempfile.TemporaryDirectory(prefix='n0-restore-') as td:
            temp = pathlib.Path(td)
            with tarfile.open(version_dir / 'files.tar.gz', 'r:gz') as archive:
                members = archive.getmembers()
                names = [m.name for m in members]
                if len(names) != len(set(names)) or set(names) != set(paths):
                    issues.append('archive_members_mismatch')
                for member in members:
                    dest = (temp / member.name).resolve()
                    if (not member.isfile() or not dest.is_relative_to(temp.resolve())
                            or member.name not in paths):
                        raise RuntimeError('Invalid snapshot member: ' + member.name)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, open(dest, 'wb') as out:
                        for block in iter(lambda: src.read(1024 * 1024), b''):
                            out.write(block)
                    if digest(dest) != paths[member.name]['sha256']:
                        issues.append('restored_hash_mismatch:' + member.name)
                    restored += 1
    if check_aliases:
        node = root / NODE_REL
        ptr = json.loads((node / 'manifests/current.json').read_text())
        if ptr['manifest_sha256'] != digest(version_dir / 'manifest.json'):
            issues.append('current_pointer_manifest_mismatch')
        if ptr['integrity_sha256'] != digest(version_dir / 'integrity.json'):
            issues.append('current_pointer_integrity_mismatch')
        if digest(node / 'manifests/source_fingerprint_current_candidate.json') != digest(version_dir / 'manifest.json'):
            issues.append('current_alias_mismatch')
    return {'version': manifest['version'], 'verdict': 'PASS' if not issues else 'FAIL',
            'scope': 'N0 source/asset integrity only; not N1-N4 or training approval',
            'source_count': len(paths), 'source_bytes': sum(v['bytes'] for v in paths.values()),
            'archive_verified': check_archive, 'restored_file_count': restored, 'issues': issues}


def validate_retirements(root, old, retirement_file):
    """Only an explicit, predecessor-bound record may remove a managed source."""
    if retirement_file is None:
        return []
    record = json.loads(retirement_file.read_text())
    expected = hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    if record.get('predecessor_content_sha256') != expected:
        raise ValueError('Retirement record does not match predecessor manifest')
    old_by_path = {v['path']: v for v in old['sources'].values()}
    retired = record['retired_sources']
    seen = set()
    for item in retired:
        rel = item['path']
        path = pathlib.PurePosixPath(rel)
        if path.is_absolute() or '..' in path.parts or rel in seen:
            raise ValueError('Invalid/duplicate retirement path: ' + rel)
        if rel not in old_by_path or item['sha256'] != old_by_path[rel]['sha256']:
            raise ValueError('Retirement hash not in predecessor: ' + rel)
        if not item.get('reason', '').strip():
            raise ValueError('Retirement requires a reason: ' + rel)
        if (root / rel).exists() or (root / rel).is_symlink():
            raise ValueError('Retired source still exists: ' + rel)
        seen.add(rel)
    return retired


def create(root, version, retirement_file=None):
    if not re.fullmatch(r'n0_\d{8}_r\d+', version):
        raise ValueError('Version must look like n0_20260928_r1')
    node = root / NODE_REL
    final = node / 'manifests/freezes' / version
    if final.exists():
        raise FileExistsError('Immutable version already exists: ' + str(final))
    oldpath = node / 'manifests/source_fingerprint_current_candidate.json'
    old = json.loads(oldpath.read_text())
    retired = validate_retirements(root, old, retirement_file)
    config = json.loads((node / 'configs/frozen_protocol.json').read_text())
    if config.get('freeze_version') != version:
        raise ValueError('frozen_protocol.freeze_version must match the new version')
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = pathlib.Path(tempfile.mkdtemp(prefix='.' + version + '-', dir=final.parent))
    sources = catalog(root, old, [item['path'] for item in retired])
    old_paths = {v['path'] for v in old['sources'].values()}
    new_by_path = {v['path']: v for v in sources.values()}
    change = {'previous_manifest_sha256': digest(oldpath), 'previous_source_count': len(old['sources']),
              'changed_previous_sources': [], 'added_source_count': len(set(new_by_path) - old_paths),
              'retired_source_count': len(retired), 'retired_sources': retired}
    for name, item in old['sources'].items():
        now = new_by_path.get(item['path'])
        if now is None:
            continue  # validate_retirements bound this removal to the predecessor hash.
        if item['sha256'] != now['sha256']:
            change['changed_previous_sources'].append({'key': name, 'path': item['path'],
                                                       'before': item['sha256'], 'after': now['sha256']})
    write_json(staging / 'environment.json', capture_environment(root))
    write_json(staging / 'change_summary.json', change)
    # Snapshot bytes are verified against the manifest after packaging, detecting concurrent edits.
    with tarfile.open(staging / 'files.tar.gz', 'w:gz', compresslevel=1) as archive:
        for v in sources.values():
            archive.add(root / v['path'], arcname=v['path'], recursive=False)
    manifest = {'schema_version': 2, 'version': version,
                'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'scope': 'Managed N0 source, configuration, simulation assets and initialization files',
                'sources': sources, 'N0_status': 'SOURCE_BASELINE_SEALED',
                'N1_N4_acceptance': 'NOT_GRANTED', 'production_runtime_unified': False,
                'unresolved': config['unresolved'],
                'excluded': ['raw episodes', 'credentials and ssh helpers', 'cache and backups',
                             'policy weights', 'container image layers'],
                'recovery': 'files.tar.gz restores every listed source; use a separate directory first'}
    write_json(staging / 'manifest.json', manifest)
    write_json(staging / 'integrity.json', {'version': version, 'files': {
        f: digest(staging / f) for f in ['manifest.json', 'files.tar.gz', 'environment.json', 'change_summary.json']}})
    result = verify(root, staging, check_archive=True)
    if result['verdict'] != 'PASS':
        raise RuntimeError('Snapshot failed verification; staging retained: ' + json.dumps(result))
    os.rename(staging, final)
    # Keep the exact predecessor manifest before advancing the compatibility alias.
    predecessor = final / 'predecessor_manifest.json'
    predecessor.write_bytes(oldpath.read_bytes())
    seal = json.loads((final / 'integrity.json').read_text())
    seal['files']['predecessor_manifest.json'] = digest(predecessor)
    write_json(final / 'integrity.json', seal)
    # Alias is a copy, not a mutable substitute for the independently sealed version.
    temp_alias = oldpath.with_suffix('.json.tmp')
    temp_alias.write_bytes((final / 'manifest.json').read_bytes())
    os.replace(temp_alias, oldpath)
    pointer = {'version': version, 'manifest': str((final / 'manifest.json').relative_to(root)),
               'manifest_sha256': digest(final / 'manifest.json'),
               'integrity_sha256': digest(final / 'integrity.json'),
               'N0_status': 'PASS', 'N1_N4_acceptance': 'NOT_GRANTED'}
    write_json(node / 'manifests/current.json', pointer)
    result.update({'snapshot': str(final.relative_to(root)), 'archive_sha256': digest(final / 'files.tar.gz')})
    write_json(node / 'results' / (version + '_verification.json'), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['verify', 'create'], default='verify', nargs='?')
    p.add_argument('--version')
    p.add_argument('--check-archive', action='store_true')
    p.add_argument('--output', type=pathlib.Path)
    p.add_argument('--retire-list', type=pathlib.Path,
                   help='create only: explicit predecessor-bound source retirement record')
    a = p.parse_args()
    if a.command == 'create':
        if not a.version:
            p.error('create requires an explicit new --version')
        result = create(REPO, a.version, a.retire_list)
    else:
        if a.retire_list:
            p.error('--retire-list is only valid for create')
        node = REPO / NODE_REL
        current = json.loads((node / 'manifests/current.json').read_text())
        version = a.version or current['version']
        if not re.fullmatch(r'n0_\d{8}_r\d+', version):
            p.error('invalid version')
        result = verify(REPO, node / 'manifests/freezes' / version, a.check_archive,
                        check_aliases=version == current['version'])
    if a.output:
        write_json(a.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['verdict'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
