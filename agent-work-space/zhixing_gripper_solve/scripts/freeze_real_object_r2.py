"""Seal the native-object investigation without overwriting prior versions."""
import argparse, ast, datetime, hashlib, json, pathlib, tarfile

ap = argparse.ArgumentParser()
ap.add_argument('--teacher', required=True)
ap.add_argument('--lock', required=True)
a = ap.parse_args()
P = pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/zhixing_gripper_solve')
out = P / 'maintenance/real_objects_20261002_r1'
root = P.parent.parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, value):
    with p.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
lock = json.loads(pathlib.Path(a.lock).read_text())
for path, digest in lock['files'].items():
    assert sha(root / path) == digest, path
source = {P / 'README.md'}
for pattern in ('真实碗罐*.md', '控制配置*.md', 'scripts/*real*.py'):
    source.update(P.glob(pattern))
source.update(P / 'scripts' / n for n in (
    'safe_teacher_r8.py', 'loaded_diagnostics_r5.py', 'eco65_expert_validation.py'))
for p in source:
    if p.suffix == '.py':
        ast.parse(p.read_text())
manifest = {
    'version': 'real_objects_20261002_r1',
    'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'latest_teacher': a.teacher,
    'files': {str(p.relative_to(P)): {'sha256': sha(p), 'bytes': p.stat().st_size}
              for p in sorted(source)},
    'note': 'Historical development sources included; report identifies tested candidates and failures.'}
dump(out / 'source_manifest.json', manifest)
with tarfile.open(out / 'source_snapshot.tar.gz', 'x:gz') as tf:
    for p in sorted(source):
        tf.add(p, arcname=str(p.relative_to(P)), recursive=False)
with tarfile.open(out / 'source_snapshot.tar.gz') as tf:
    for n, info in manifest['files'].items():
        assert hashlib.sha256(tf.extractfile(n).read()).hexdigest() == info['sha256']
evidence = []
for p in (P / 'evidence').glob('real*'):
    evidence.extend([p] if p.is_file() else [f for f in p.rglob('*') if f.is_file()])
em = {str(p.relative_to(P)): {'sha256': sha(p), 'bytes': p.stat().st_size}
      for p in sorted(set(evidence))}
dump(out / 'evidence_manifest.json', {
    'files': em,
    'note': 'All native-object development, failed trials, A/B and validation evidence retained separately.'})
funcs = []
for filename in ('real_object_teacher_r8.py', a.teacher + '.py'):
    tree = ast.parse((P / 'scripts' / filename).read_text())
    funcs.append(next(ast.dump(n, include_attributes=False) for n in tree.body
                      if isinstance(n, ast.FunctionDef) and n.name == 'occupied_intervals'))
assert funcs[0] == funcs[1]
dump(out / 'verification.json', {
    'source_manifest': 'PASS', 'archive_restore': 'PASS',
    'source_count': len(source), 'evidence_files': len(em),
    'evidence_bytes': sum(v['bytes'] for v in em.values()),
    'candidate_lock': 'PASS', 'ray_intersection_function_matches_tested_r8': True,
    'upstream': json.loads((out / 'upstream_verification.json').read_text())['verdict'],
    'scope': 'Integrity verification only; not a claim of task acceptance.'})
print('FREEZE PASS', len(source), 'sources;', len(em), 'evidence files')
