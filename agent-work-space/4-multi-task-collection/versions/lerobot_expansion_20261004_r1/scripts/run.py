"""Host launcher: isolated CPU Docker runtime, read-only raw inputs, real exit records."""
import argparse,json,os,pathlib,subprocess,time
V=pathlib.Path(__file__).resolve().parents[1];C=V.parents[1];REPO=C.parents[1]
R=REPO/'agent-work-space/2-task-collection/runtime/lerobot_py310_r1'
def command(script,args):
    return ['docker','run','--rm','--network','none','--cpus','2','--memory','8g','--user',f'{os.getuid()}:{os.getgid()}',
      '-e','HOME=/tmp','-e','CUDA_VISIBLE_DEVICES=','-e','HF_HUB_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1',
      '-e','HF_HOME=/tmp/hf','-v',str(REPO)+':/app:ro','-v',str(C/'lerobot_exports')+':/app/agent-work-space/4-multi-task-collection/lerobot_exports',
      '--entrypoint','/app/'+str((R/'venv/bin/python').relative_to(REPO)),'libero:latest','/app/'+str((V/'scripts'/script).relative_to(REPO))]+args

def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--limit-each',type=int);a=p.parse_args();assert pathlib.Path(a.name).name==a.name and a.name not in ('.','..')
    parent=C/'lerobot_exports';parent.mkdir(exist_ok=True);out=parent/a.name;assert not out.exists(),'output already exists'
    logdir=parent/(a.name+'_execution');logdir.mkdir(exist_ok=False)
    rel='/app/'+str(out.relative_to(REPO));statuses=[]
    for script in ('convert.py','validate.py'):
        args=['--out',rel]
        if script=='convert.py' and a.limit_each:args+=['--limit-each',str(a.limit_each)]
        cmd=command(script,args);t=time.time()
        with (logdir/(script+'.log')).open('x') as f:
            try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=3600).returncode
            except subprocess.TimeoutExpired:code=-999
        st={'script':script,'exit_code':code,'seconds':time.time()-t,'command':cmd};statuses.append(st)
        with (logdir/(script+'.status.json')).open('x') as f:json.dump(st,f,indent=2)
        print(script,'exit',code,'seconds',round(st['seconds'],1),flush=True)
        if code:break
    with (logdir/'process_status.json').open('x') as f:json.dump(statuses,f,indent=2)
    return int(any(r['exit_code'] for r in statuses))
if __name__=='__main__':raise SystemExit(main())
