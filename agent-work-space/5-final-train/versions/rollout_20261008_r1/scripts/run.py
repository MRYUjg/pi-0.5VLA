"""Resumable, four-condition 10x20 evaluation; every reserved attempt stays in denominator."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import urllib.request

V = pathlib.Path(__file__).resolve().parents[1]
ROOT = V.parents[1]
REPO = ROOT.parents[1]
PORT = 8872


def rel(p):
    return '/app/' + str(p.relative_to(REPO))


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def verify():
    manifest = ROOT/'releases'/V.name/'manifest.json'
    for name,want in json.loads(manifest.read_text())['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == want, name
    for name in ('lora_20261007_r1','collection_20261005_r4'):
        for file,want in json.loads((ROOT/'releases'/name/'manifest.json').read_text())['files'].items():
            assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == want, file
    environment = json.loads((V/'environment_manifest.json').read_text())
    for file,want in environment['files'].items():
        assert hashlib.sha256((REPO/file).read_bytes()).hexdigest() == want, file
    for image,want in environment['docker_images'].items():
        actual = subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()
        assert actual == want, image


def server_command(kind,out):
    return ['docker','run','--rm','--name','openpi-closed-loop-policy','--gpus','all','--network','host',
            '--memory','22g','--shm-size','1g','--cpus','8',
            '-e','PYTHONPATH=/app/src:/app/packages/openpi-client/src:/app',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENPI_DATA_HOME=/openpi_assets',
            '-e','HF_HUB_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
            '-e','HF_HOME=/tmp/hf','-e','XDG_CACHE_HOME=/tmp/cache',
            '-e','XLA_PYTHON_CLIENT_MEM_FRACTION=0.85','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false',
            '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1',
            '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':'+rel(ROOT),
            '-v','/mnt/data/vla-policy/caches/openpi:/openpi_assets:ro',
            '-w','/app','--entrypoint','/.venv/bin/python','openpi_server:latest',
            rel(V/'scripts/serve.py'),'--kind',kind,'--out',rel(out),'--port',str(PORT)]


def episode_command(condition,task,seed,out,video=False,smoke=False):
    cmd = ['docker','run','--rm','--network','host','--cpus','2','--memory','3g',
           '-e','MUJOCO_GL=osmesa','-e','PYTHONDONTWRITEBYTECODE=1',
           '-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','LP_NUM_THREADS=1',
           '-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero',
           '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':'+rel(ROOT),
           '-w','/app','--entrypoint','/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python',
           'libero:latest',rel(V/'scripts/episode.py'),'--condition',condition,'--task',str(task),
           '--seed',str(seed),'--out',rel(out),'--port',str(PORT)]
    if video: cmd.append('--video')
    if smoke: cmd.append('--smoke')
    return cmd


def wait_ready(server):
    for _ in range(450):
        if server.poll() is not None:
            raise RuntimeError('server exited: '+str(server.returncode))
        try:
            data = json.loads(urllib.request.urlopen(f'http://127.0.0.1:{PORT}',timeout=1).read())
            if data['ready']: return data
        except (OSError,ValueError): pass
        time.sleep(2)
    raise TimeoutError('policy server startup')


def stop_server(server):
    try:
        urllib.request.urlopen(urllib.request.Request(f'http://127.0.0.1:{PORT}/shutdown',data=b''),timeout=5).read()
    except OSError: pass
    try: server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker','stop','-t','15','openpi-closed-loop-policy'],check=False)
        server.wait(timeout=30)


def summary(out,protocol,seeds):
    result = {'complete':False,'conditions':{},'target':0.90,'source_version':V.name,
              'trained_tasks':[0,2,3,5,6],'untrained_tasks':[1,4,7,8,9]}
    for condition in protocol['conditions']:
        tasks = {}
        for task,task_seeds in seeds.items():
            rows = []
            for seed in task_seeds:
                path = out/condition/f'task{int(task):02d}_s{seed}'/'execution.json'
                if path.exists(): rows.append(json.loads(path.read_text()))
            tasks[task] = {'completed':len(rows),'planned':20,
                          'successes':sum(r.get('process_exit')==0 and r.get('result',{}).get('primary_success',False) for r in rows),
                          'invalid_initializations':sum(r.get('result',{}).get('failure')=='invalid_native_initialization' for r in rows),
                          'process_failures':sum(r.get('process_exit')!=0 for r in rows)}
        total = sum(r['completed'] for r in tasks.values())
        success = sum(r['successes'] for r in tasks.values())
        result['conditions'][condition] = {'tasks':tasks,'completed':total,'planned':200,'successes':success,
            'observed_success_rate':success/total if total else None,
            'fixed_denominator_success_rate':success/200,
            'acceptance_pass':total==200 and success>=180}
    result['complete'] = all(c['completed']==200 for c in result['conditions'].values())
    write(out/'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    a = p.parse_args()
    assert pathlib.Path(a.name).name == a.name and a.name not in ('.','..')
    if not a.smoke: verify()
    protocol = json.loads((V/'configs/protocol.json').read_text())
    seeds = json.loads((V/'configs/test_seeds.json').read_text())['tasks']
    out = ROOT/'runs'/a.name
    out.mkdir(parents=True,exist_ok=a.resume)
    status = {'phase':'starting','name':a.name,'pid':os.getpid(),'smoke':a.smoke,'version':V.name,'started':time.time()}
    write(out/'status.json',status)
    sequence = [('finetuned',['eco65_zhixing_lora']), ('base',['panda_base','eco65_zhixing_base','eco65_panda_base'])]
    for kind,conditions in sequence:
        if a.resume and all((out/c/f'task{int(t):02d}_s{s}'/'execution.json').exists() for c in conditions for t,ss in seeds.items() for s in ss):
            continue
        cmd = server_command(kind,out)
        server_index = len(list(out.glob('server_'+kind+'*.log')))
        with (out/f'server_{kind}_{server_index}.log').open('x') as log:
            server = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            try:
                meta = wait_ready(server)
                assert meta['kind'] == kind
                status.update(phase='evaluating',model=kind)
                write(out/'status.json',status)
                def execute(job):
                    condition,task,seed,video = job
                    episode = out/condition/f'task{task:02d}_s{seed}'
                    done = episode/'execution.json'
                    if done.exists(): return json.loads(done.read_text())
                    if episode.exists():
                        # Interrupted attempts are retained, counted as failures; no retry/replacement.
                        row = {'condition':condition,'task':task,'seed':seed,'process_exit':-1,'interrupted':True}
                        write(done,row)
                        return row
                    episode.parent.mkdir(parents=True,exist_ok=True)
                    command = episode_command(condition,task,seed,episode,video,a.smoke)
                    t = time.time()
                    with (episode.parent/(episode.name+'.log')).open('x') as f:
                        try: code = subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=1200).returncode
                        except subprocess.TimeoutExpired: code = 124
                    # Container-created artifacts are root-owned; use a small isolated chown helper.
                    subprocess.run(['docker','run','--rm','--network','none','-v',str(episode)+':/output',
                                    '--entrypoint','chown','libero:latest','-R',f'{os.getuid()}:{os.getgid()}','/output'],check=True)
                    row = {'condition':condition,'task':task,'seed':seed,'process_exit':code,
                           'seconds':time.time()-t,'command':command}
                    if (episode/'result.json').exists(): row['result'] = json.loads((episode/'result.json').read_text())
                    write(done,row)
                    print('EPISODE',condition,task,seed,'exit',code,'success',row.get('result',{}).get('primary_success'),flush=True)
                    return row
                jobs = [(c,t,s,i==0) for c in conditions for i in range(20) for t in range(10) for s in [seeds[str(t)][i]]]
                if a.smoke: jobs = [(c,2,630201,True) for c in conditions]
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                    futures = [pool.submit(execute,job) for job in jobs]
                    for future in concurrent.futures.as_completed(futures):
                        row = future.result()
                        if a.smoke:
                            assert row['process_exit']==0 and row['result']['initialization_pass'] and row['result']['queries']==2, row
                        else:
                            current = summary(out,protocol,seeds)
                            status['completed'] = {k:v['completed'] for k,v in current['conditions'].items()}
                            write(out/'status.json',status)
            finally:
                stop_server(server)
                write(out/f'server_{kind}_{server_index}_execution.json',{'process_exit':server.returncode,'command':cmd})
            assert server.returncode == 0, server.returncode
    status.update(phase='smoke_complete' if a.smoke else 'evaluation_complete',completed_at=time.time())
    write(out/'status.json',status)
    if not a.smoke:
        summary(out,protocol,seeds)
        status['phase'] = 'verifying_results_and_videos'
        write(out/'status.json',status)
        cmd = ['docker','run','--rm','--network','none','--cpus','2','--memory','3g',
               '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
               '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':'+rel(ROOT),
               '--entrypoint','/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python',
               'libero:latest',rel(V/'scripts/verify_results.py'),'--run',a.name]
        with (out/'verification.log').open('x') as log:
            code = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        status.update(phase='evaluation_and_verification_complete' if code==0 else 'verification_failed',
                      verification_exit=code, completed_at=time.time())
        write(out/'status.json',status)
        assert code==0, 'Independent verification failed; inspect verification.log'


if __name__ == '__main__':
    main()
