"""Freeze new evidence, preserving both earlier Panda releases and failed development runs."""
from pathlib import Path
import datetime
import hashlib
import json

ROOT = Path('/mnt/data/vla-policy/openpi/gpt-rpent-control')
RUN_ID = 'eco65_zhixing_deploy_spatial10_20261007_r1'
REPORT = ROOT / 'reports/eco65_zhixing_deploy_20261007_r1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks = []
for name in ['panda_libero_20261007_r1', 'spatial10_video_20261007_r2', 'eco65_zhixing_spatial10_20261007_r1']:
    old = json.loads((ROOT / 'releases' / name / 'manifest.json').read_text())
    files = old['files_relative_to_gpt_rpent_control']
    mismatches = [rel for rel, entry in files.items()
                  if not (ROOT / rel).exists() or sha(ROOT / rel) != entry['sha256']]
    checks.append(dict(release=name, files=len(files), mismatches=mismatches))
assert all(not row['mismatches'] for row in checks)

selected = [ROOT / 'runs' / RUN_ID, ROOT / 'videos' / RUN_ID, REPORT]
selected += [ROOT / 'versions/eco65_zhixing_20261007_r5']
selected += [ROOT / 'runs/eco65_zhixing_deploy_chain_20261007_r1']
selected += [ROOT / 'reports/eco65_zhixing_deploy_dependencies_before_20261007_r1.json',
             ROOT / 'reports/eco65_zhixing_deploy_dependencies_after_20261007_r1.json',
             ROOT / 'ECO65_ZhiXing部署配置复验与视频_20261007_r2.md',
             ROOT / '当前版本与文件索引_20261007_r4.md']

files = set()
for path in selected:
    assert path.exists(), path
    files.update([p for p in path.rglob('*') if p.is_file()] if path.is_dir() else [path])
files = {p for p in files if '__pycache__' not in p.parts and not p.name.startswith('._')}
key = (ROOT.parent / 'gpt-6-control/deepseek_api_key').read_bytes().strip()
assert key and all(key not in path.read_bytes() for path in files), 'credential value present in new evidence'
receipt = dict(prior_releases=checks, credential_value_found=False,
               credential_files_scanned=len(files), frozen_files_are_not_adaptation_acceptance=True)
with (REPORT / 'prior_and_credential_check.json').open('x') as file:
    json.dump(receipt, file, indent=2)
files.add(REPORT / 'prior_and_credential_check.json')

summary = json.loads((REPORT / 'summary.json').read_text())
manifest = dict(run_id=RUN_ID, code_version=summary['code_version'],
                frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                files_relative_to_gpt_rpent_control={str(p.relative_to(ROOT)):dict(
                    sha256=sha(p), bytes=p.stat().st_size) for p in sorted(files)},
                acceptance=summary,
                excluded=['runtime', 'credentials', 'model_weights', 'other_users_runs'])
output = ROOT / 'releases' / RUN_ID
output.mkdir(parents=True, exist_ok=False)
with (output / 'manifest.json').open('x') as file:
    json.dump(manifest, file, ensure_ascii=False, indent=2)
assert all(sha(ROOT / rel) == entry['sha256']
           for rel, entry in manifest['files_relative_to_gpt_rpent_control'].items())
print(json.dumps(dict(new_files_frozen=len(files), prior_releases=checks,
                     native_successes=summary['native_successes'], retained_MP4=20)))
