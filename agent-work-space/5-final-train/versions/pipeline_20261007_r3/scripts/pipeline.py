"""Start the frozen LoRA phases after explicit user authorization and data readback."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

VERSION = Path(__file__).resolve().parents[1]
ROOT = VERSION.parents[1]
TRAINING_VERSION = 'lora_20261007_r1'
RUN = 'spatial_lora_20261007_r1'
POLICY = ROOT / 'reports/execution_policy_20261007_r2.json'


def read(path):
    return json.loads(path.read_text())


def verify():
    policy = read(POLICY)
    if not policy['training_authorized'] or policy['training_run'] != RUN:
        raise RuntimeError('This training run is not authorized')
    for version in (TRAINING_VERSION, VERSION.name):
        for relative, expected in read(ROOT / 'releases' / version / 'manifest.json')['files'].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen file changed: ' + relative)


def main():
    output = ROOT / 'runs' / VERSION.name
    output.mkdir(parents=True, exist_ok=False)
    state = {
        'pid': os.getpid(), 'started': time.time(), 'state': 'running',
        'stage': 'preflight', 'stages': [], 'dataset_export': 'spatial_formal_r3',
        'training_run': RUN, 'training_version': TRAINING_VERSION,
        'automatic_training': True, 'user_instruction': '启动对pai0.5的微调训练',
        'execution_policy': str(POLICY), 'expected_training_steps': 10000,
        'scope': 'Authorized LoRA training and checkpoint reload; policy evaluation pending.',
    }

    def status():
        state['updated'] = time.time()
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output / 'status.json')

    def phase(name):
        verify()
        state['stage'] = name
        status()
        command = [sys.executable, str(ROOT / 'versions' / TRAINING_VERSION / 'scripts/launch.py'),
                   '--run', RUN, '--phase', name]
        started = time.time()
        print('START', name, flush=True)
        with (output / (name + '.log')).open('x') as log:
            code = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        state['stages'].append({'stage': name, 'exit_code': code,
                                'seconds': time.time() - started, 'command': command})
        status()
        print('END', name, code, flush=True)
        if code:
            raise RuntimeError(name + ' failed with exit ' + str(code))

    try:
        verify()
        status()
        release = read(ROOT / 'runs/formal_merged_20261006_r1/summary.json')
        validation = read(ROOT / 'lerobot_exports/spatial_formal_r3/validation.json')
        conversion = read(ROOT / 'runs/spatial_formal_r3_conversion/summary.json')
        if not release['pass'] or release['attempts'] != 310:
            raise RuntimeError('Merged data admission has not passed')
        if not validation['pass'] or validation['frames_verified'] != 170724 or not validation['source_unchanged']:
            raise RuntimeError('Full dataset readback has not passed')
        if not conversion['pass'] or any(row['exit_code'] != 0 for row in conversion['records']):
            raise RuntimeError('Conversion/readback process did not exit normally')
        state.update(data_release_pass=True, dataset_validation_pass=True, counts=release['counts'])
        if (ROOT / 'runs' / RUN).exists() or (ROOT / 'runs' / (RUN + '_execution')).exists():
            raise RuntimeError('Training outputs already exist; refusing to overwrite')
        free = shutil.disk_usage(ROOT).free
        if free < 60 * 1024**3:
            raise RuntimeError('Less than 60 GiB free for reader caches and checkpoints')
        state['disk_free_gib_at_start'] = free / 1024**3
        status()
        for name in ('prepare', 'smoke', 'train', 'reload'):
            phase(name)
            if name == 'prepare':
                state['normalization_ready'] = True
            elif name == 'smoke':
                if not read(ROOT / 'runs' / RUN / 'resource_smoke_result.json')['pass']:
                    raise RuntimeError('Resource smoke test did not pass')
                state['resource_smoke_pass'] = True
            elif name == 'train':
                if not read(ROOT / 'runs' / RUN / 'training_result.json')['pass']:
                    raise RuntimeError('Formal training did not pass')
                state['training_pass'] = True
            status()
        if not read(ROOT / 'runs' / RUN / 'reload_result.json')['pass']:
            raise RuntimeError('Independent checkpoint reload did not pass')
        state.update(state='completed', stage='training_and_reload_complete', reload_pass=True)
        status()
        return 0
    except BaseException as error:
        state.update(state='stopped', error=repr(error), traceback=traceback.format_exc())
        status()
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
