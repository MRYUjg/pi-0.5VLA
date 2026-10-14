"""Host launcher: exclusive outputs, source snapshot, logs and real exit codes."""
import argparse
import json
import pathlib
import shutil
import subprocess
import time

ROOT = pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference')
VERSION = 'trt_t3_20261007_r1'
RUN = ROOT / 'runs' / VERSION
CONTAINER = 'pi05-trt-t3-20261007-r1'
parser = argparse.ArgumentParser()
parser.add_argument('--name', required=True)
parser.add_argument('script')
parser.add_argument('args', nargs=argparse.REMAINDER)
a = parser.parse_args()
assert pathlib.Path(a.name).name == a.name and a.name not in ('.', '..')
assert pathlib.Path(a.script).name == a.script and a.script.endswith('.py')
snapshot = RUN / (a.name + '_source')
snapshot.mkdir(exist_ok=False)
for p in (ROOT / 'versions' / VERSION).glob('*.py'):
    shutil.copy2(p, snapshot / p.name)
cmd = ['docker', 'exec', CONTAINER, '/.venv/bin/python', '/work/runs/' + VERSION + '/' + snapshot.name + '/' + a.script, *a.args]
start = time.time()
with (RUN / (a.name + '.log')).open('x') as f:
    code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
with (RUN / (a.name + '_exit.json')).open('x') as f:
    json.dump({'command': cmd, 'exit_code': code, 'seconds': time.time() - start}, f, indent=2)
print('EXIT', a.name, code, flush=True)
raise SystemExit(code)
