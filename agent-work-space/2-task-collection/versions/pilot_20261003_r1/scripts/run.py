"""Host launcher; immutable inputs, unique outputs, real process status."""
import argparse,hashlib,json,pathlib,subprocess,time
VERSION=pathlib.Path(__file__).resolve().parents[1]
COLLECTION=VERSION.parents[1]
REPO=COLLECTION.parents[1]
SOLVE=REPO/'agent-work-space/1.3-zhixing_gripper_solve'

def container_command(script,args):
    rel=lambda p:'/app/'+str(p.relative_to(REPO))
    return ['docker','run','--rm','--network','none','--cpus','2','--memory','5g',
      '-e','EXTRA_JOINT_PD_PROFILE=eco65_no_extra_pd','-e','EXTRA_JOINT_PD_KP=',
      '-e','MUJOCO_GL=osmesa','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1',
      '-e','OMP_NUM_THREADS=1','-e','LP_NUM_THREADS=1',
      '-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero',
      '-v',str(REPO)+':/app:ro','-v',str(COLLECTION)+':'+rel(COLLECTION),
      '-w','/app','--entrypoint',rel(SOLVE/'runtime/py310_mj326/bin/python'),'libero:latest',rel(VERSION/'scripts'/script)]+args

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--seeds',default='53100,53101,53102,53103,53104');ap.add_argument('--probe',action='store_true');a=ap.parse_args()
    assert pathlib.Path(a.batch).name==a.batch and a.batch not in ('.','..')
    seeds=[int(s) for s in a.seeds.split(',')];assert len(set(seeds))==len(seeds)
    dst=COLLECTION/'runs'/a.batch;dst.mkdir(parents=True,exist_ok=False)
    hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in VERSION.rglob('*') if p.is_file() and '__pycache__' not in str(p)}
    (dst/'protocol.json').write_text(json.dumps({'version':VERSION.name,'seeds':seeds,'suite':'libero_object','task_id':0,'probe':a.probe,'sources':hashes,'profile':'eco65_no_extra_pd','init_profile':'legacy','scope':'single-task pipeline pilot, not Spatial acceptance'},indent=2))
    statuses=[]
    for seed in seeds:
        out=dst/('can_s'+str(seed));cmd=container_command('collect.py',['--seed',str(seed),'--out','/app/'+str(out.relative_to(REPO))]+(['--probe'] if a.probe else []))
        start=time.time()
        with out.with_suffix('.log').open('x') as f:
            try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
            except subprocess.TimeoutExpired:code=-999
        st={'seed':seed,'exit_code':code,'normal_exit':code==0,'wall_seconds':time.time()-start,'command':cmd}
        out.with_suffix('.status.json').write_text(json.dumps(st,indent=2));statuses.append(st)
        print(seed,'exit',code,'seconds',round(st['wall_seconds'],1),flush=True)
    (dst/'process_status.json').write_text(json.dumps(statuses,indent=2))
    return int(not all(s['normal_exit'] for s in statuses))
if __name__=='__main__':raise SystemExit(main())
