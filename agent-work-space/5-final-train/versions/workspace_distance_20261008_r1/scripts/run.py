"""Run the registered distance comparison without changing production source."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess

V = pathlib.Path(__file__).resolve().parents[1]
ROOT = V.parents[1]
NAME = 'workspace_distance_20261008_r1'
OUT = ROOT / 'runs' / NAME
spec = importlib.util.spec_from_file_location('frozen_runner', ROOT / 'versions/rollout_20261008_r1/scripts/run.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temp.replace(path)


def execute(job, preparation=False):
    offset, task, seed = job
    label = 'base_%02dcm' % round(offset*100)
    dest = OUT / ('preflight' if preparation else 'rollouts') / label / ('task%02d_s%d' % (task, seed))
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = old.episode_command('eco65_zhixing_lora', task, seed, dest, video=True, smoke=preparation)
    index = command.index(old.rel(old.V / 'scripts/episode.py'))
    command[index] = old.rel(V / 'scripts/episode.py')
    command += ['--offset', str(offset)]
    if preparation:
        command += ['--preparation-only']
    with dest.with_suffix('.log').open('x') as log:
        try:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            code = 124
    if dest.exists():
        subprocess.run(['docker', 'run', '--rm', '--network', 'none', '-v', str(dest)+':/output',
                        '--entrypoint', 'chown', 'libero:latest', '-h', '-R', f'{os.getuid()}:{os.getgid()}', '/output'], check=True)
    row = {'offset_m': offset, 'task': task, 'seed': seed, 'process_exit': code, 'command': command,
           'preparation_only': preparation, 'path': str(dest.relative_to(ROOT))}
    if (dest / 'result.json').exists():
        row['result'] = json.loads((dest / 'result.json').read_text())
    dest.mkdir(exist_ok=True)
    save(dest / 'execution.json', row)
    print('PRECHECK' if preparation else 'ROLLOUT', label, task, seed, 'exit', code,
          'init', row.get('result', {}).get('initialization_pass'),
          'first_success', row.get('result', {}).get('first_success_step'), flush=True)
    return row


def main():
    old.verify()
    for name, want in json.loads((ROOT / 'releases' / V.name / 'manifest.json').read_text())['files'].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == want, name
    protocol = json.loads((V / 'configs/protocol.json').read_text())
    OUT.mkdir(exist_ok=False)
    status = {'phase': 'preflight', 'pid': os.getpid(), 'version': V.name, 'planned': 18}
    save(OUT / 'status.json', status)
    jobs = [(offset, int(task), seed) for offset in protocol['offsets_m']
            for task, seeds in protocol['development_seeds'].items() for seed in seeds]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        prechecks = list(pool.map(lambda job: execute(job, True), jobs))
    assert all(r['process_exit'] == 0 and r.get('result') for r in prechecks), 'preflight execution failure'
    admitted, attempts = [], []
    for job, row in zip(jobs, prechecks):
        if row['result']['initialization_pass']:
            admitted.append(job)
        else:
            attempts.append({k: row[k] for k in ['offset_m', 'task', 'seed', 'path', 'preparation_only']})
    status.update(phase='policy_running', preparation_pass=len(admitted), preparation_rejected=18-len(admitted), completed=len(attempts))
    save(OUT / 'status.json', status)
    command = old.server_command('finetuned', OUT)
    with (OUT / 'server.log').open('x') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            meta = old.wait_ready(server)
            assert meta['kind'] == 'finetuned'
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                futures = [pool.submit(execute, job) for job in admitted]
                for future in concurrent.futures.as_completed(futures):
                    row = future.result()
                    attempts.append({k: row[k] for k in ['offset_m', 'task', 'seed', 'path', 'preparation_only']})
                    save(OUT / 'attempts.json', attempts)
                    status['completed'] = len(attempts)
                    save(OUT / 'status.json', status)
        finally:
            old.stop_server(server)
            save(OUT / 'server_execution.json', {'process_exit': server.returncode, 'command': command})
    assert server.returncode == 0
    save(OUT / 'attempts.json', attempts)
    status['phase'] = 'verifying'
    save(OUT / 'status.json', status)
    command = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '2', '--memory', '3g',
               '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
               '-v', str(old.REPO)+':/app:ro', '-v', str(ROOT)+':'+old.rel(ROOT),
               '--entrypoint', '/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python',
               'libero:latest', old.rel(V / 'scripts/verify.py')]
    with (OUT / 'verification.log').open('x') as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
    save(OUT / 'verification_execution.json', {'process_exit': code, 'command': command})
    status.update(phase='complete' if code == 0 else 'verification_failed', verification_exit=code)
    save(OUT / 'status.json', status)
    assert code == 0
    print('DISTANCE EXPERIMENT COMPLETE', flush=True)


if __name__ == '__main__':
    main()
