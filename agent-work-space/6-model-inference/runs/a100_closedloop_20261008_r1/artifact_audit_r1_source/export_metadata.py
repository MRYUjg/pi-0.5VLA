"""Stream a partial evidence archive; never copy engines or old weight trees."""
import json
import pathlib
import sys
import tarfile

ROOT = pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference')
VERSION = 'a100_closedloop_20261008_r1'
release = ROOT / 'releases' / VERSION
manifest = json.loads((release / 'manifest.json').read_text())
report = json.loads((ROOT / 'runs' / VERSION / 'report_r1/summary.json').read_text())
suffixes = {'.json', '.jsonl', '.log', '.csv', '.py', '.md', '.npy', '.npz'}
paths = {key for key in manifest['files_relative_to_6_model_inference']
         if pathlib.Path(key).suffix in suffixes}
paths.update({f'releases/{VERSION}/manifest.json', f'releases/{VERSION}/server_verify_r1.json'})
for full, old in [('full_awq', 'awq'), ('full_gptq', 'gptq')]:
    lost = report['variants'][full]['paired_vs_same_algorithm_prefix_only']['lost']
    if lost:
        task, index = lost[0]['task'], lost[0]['init_index']
        for variant in [old, full]:
            paths.add(f'runs/{VERSION}/closedloop_{variant}_r1/task{task:02d}/init{index:02d}.mp4')
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for key in sorted(paths):
        archive.add(ROOT / key, arcname=key, recursive=False)
print('EXPORTED PARTIAL EVIDENCE', len(paths), 'files', file=sys.stderr)
