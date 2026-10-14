"""Isolated converter/reader using the previously verified pinned LeRobot runtime."""
import argparse
import json
import os
import pathlib
import subprocess
import time
V=pathlib.Path(__file__).resolve().parents[1]
ROOT=V.parents[1]
REPO=ROOT.parents[1]
def command(script,out):
    rel=lambda p:'/app/'+str(p.relative_to(REPO))
    runtime=REPO/'agent-work-space/2-task-collection/runtime/lerobot_py310_r1/venv/bin/python'
    return ['docker','run','--rm','--network','none','--cpus','4','--memory','8g',
        '--user',f'{os.getuid()}:{os.getgid()}',
        '-e','CUDA_VISIBLE_DEVICES=','-e','HF_HUB_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
        '-e','HF_HOME=/tmp/hf','-e','XDG_CACHE_HOME=/tmp/cache',
        '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1',
        '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':'+rel(ROOT),
        '--entrypoint',rel(runtime),'libero:latest',rel(V/'scripts'/script),'--out',rel(out)]
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='spatial_formal_r1');a=p.parse_args()
    assert pathlib.Path(a.name).name==a.name and a.name not in ('.','..')
    out=ROOT/'lerobot_exports'/a.name;logs=ROOT/'runs'/(a.name+'_conversion')
    logs.mkdir(parents=True,exist_ok=False)
    records=[]
    for script in ('convert.py','validate.py'):
        cmd=command(script,out);start=time.time()
        with (logs/(script+'.log')).open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        row={'script':script,'exit_code':code,'seconds':time.time()-start,'command':cmd};records.append(row)
        with (logs/(script+'.status.json')).open('x') as f:json.dump(row,f,indent=2)
        print(script,code,flush=True)
        if code:break
    with (logs/'summary.json').open('x') as f:json.dump({'pass':all(r['exit_code']==0 for r in records) and len(records)==2,'records':records},f,indent=2)
    raise SystemExit(int(len(records)!=2 or any(r['exit_code'] for r in records)))
