"""Observe the existing dataset readback and finish without launching training."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

VERSION = Path(__file__).resolve().parents[1]
ROOT = VERSION.parents[1]
RUNNER_PID = 60566
RUNNER_SCRIPT = str(ROOT / 'versions/conversion_20261007_r1/scripts/run.py')
EXPORT = ROOT / 'lerobot_exports/spatial_formal_r3'
CONVERSION = ROOT / 'runs/spatial_formal_r3_conversion'


def read(path):
    return json.loads(path.read_text())


def verify():
    for version in ('conversion_20261007_r1', VERSION.name):
        for relative, expected in read(ROOT / 'releases' / version / 'manifest.json')['files'].items():
            if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
                raise RuntimeError('Frozen file changed: ' + relative)


def main():
    output = ROOT / 'runs' / VERSION.name
    output.mkdir(parents=True, exist_ok=False)
    state = {
        'pid': os.getpid(), 'started': time.time(), 'state': 'running',
        'stage': 'dataset_readback', 'dataset_export': EXPORT.name,
        'automatic_training': False, 'training_status': 'not_started_user_hold',
        'user_instruction': '读回验证后，不要自动启动微调训练。',
        'observed_conversion_runner_pid': RUNNER_PID,
        'scope': 'Dataset readback only; training requires a new user instruction.',
    }

    def status():
        state['updated'] = time.time()
        temporary = output / 'status.tmp'
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output / 'status.json')

    try:
        verify()
        release = read(ROOT / 'runs/formal_merged_20261006_r1/summary.json')
        if not release['pass'] or release['attempts'] != 310:
            raise RuntimeError('Merged data release has not passed')
        state['data_release_pass'] = True
        state['counts'] = release['counts']
        status()
        print('Observing existing readback; automatic training is disabled.', flush=True)
        while True:
            try:
                result = read(CONVERSION / 'summary.json')
            except (FileNotFoundError, json.JSONDecodeError):
                try:
                    command = Path(f'/proc/{RUNNER_PID}/cmdline').read_bytes().decode(errors='replace')
                except FileNotFoundError:
                    command = ''
                if RUNNER_SCRIPT not in command:
                    # The runner may have exited between reading its summary and cmdline.
                    time.sleep(1)
                    result = read(CONVERSION / 'summary.json')
                else:
                    time.sleep(15)
                    continue
            state['conversion_result'] = result
            records = result.get('records', [])
            if not result['pass'] or [r['script'] for r in records] != ['convert.py', 'validate.py'] or any(r['exit_code'] != 0 for r in records):
                raise RuntimeError('Conversion/readback failed; training remains disabled')
            validation = read(EXPORT / 'validation.json')
            if not validation['pass'] or validation['frames_verified'] != 170724 or not validation['source_unchanged']:
                raise RuntimeError('Full readback validation has not passed')
            verify()
            state.update(
                state='completed', stage='data_validated_awaiting_user_training_instruction',
                dataset_validation_pass=True, frames_verified=validation['frames_verified'],
                rgb_images_verified=validation['rgb_images_verified'],
            )
            status()
            print('Dataset readback passed. Finished; training was not launched.', flush=True)
            return 0
    except BaseException as error:
        state.update(state='stopped', error=repr(error), traceback=traceback.format_exc())
        status()
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
