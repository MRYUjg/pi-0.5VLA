#!/usr/bin/env python3
"""Run independent CPU episodes of safe_teacher_r8; keep failures and real exit codes."""
import argparse,concurrent.futures,hashlib,json,pathlib,subprocess,time
ROOT=pathlib.Path('/mnt/data/vla-policy/openpi')
SOLVE=ROOT/'agent-work-space/zhixing_gripper_solve'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--seeds',default='47100,47101,47103');ap.add_argument('--workers',type=int,choices=[1,2],default=2);a=ap.parse_args()
    if pathlib.Path(a.batch).name!=a.batch or a.batch in ('.','..'):ap.error('batch must be a new single directory name')
    seeds=[int(s) for s in a.seeds.split(',')]
    if len(seeds)!=len(set(seeds)):ap.error('duplicate seeds')
    batch=SOLVE/'evidence'/a.batch;batch.mkdir(parents=True,exist_ok=False)
    jobs=[{'gripper':'ZhixingGripper','mode':'chain','seed':seed,'force':0} for seed in seeds]
    src=SOLVE/'scripts/safe_teacher_r8.py'
    protocol={'version':'run_safe_teacher_r9','teacher':'safe_teacher_r8','teacher_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'jobs':jobs,'workers':a.workers,'task':'libero_object task0 with 60x30x70 mm diagnostic block','purpose':'scripted safe-path validation; no training dataset, no formal Spatial acceptance','runtime':'Python3.10 / MuJoCo3.2.6 CPU headless','upstream_assets_controller_and_gates_changed':False,'acceptance':'safe_teacher_success plus independent verify_safe_teacher_r2.py'}
    (batch/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    def run(job):
        name='ZhixingGripper_chain_s%d_f0'%job['seed'];dst=batch/name
        cmd=['docker','run','--rm','--network','none','--cpus','2','--memory','5g','-e','MUJOCO_GL=osmesa','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero','-v',str(ROOT)+':/app:ro','-v',str(SOLVE)+':/app/agent-work-space/zhixing_gripper_solve','-w','/app','--entrypoint','/app/agent-work-space/zhixing_gripper_solve/runtime/py310_mj326/bin/python','libero:latest','/app/agent-work-space/zhixing_gripper_solve/scripts/safe_teacher_r8.py','--gripper','ZhixingGripper','--mode','chain','--seed',str(job['seed']),'--out','/app/'+str(dst.relative_to(ROOT))]
        start=time.time();error=None
        with (batch/(name+'.log')).open('x') as log:
            try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
            except subprocess.TimeoutExpired:code=None;error='process_timeout'
        result={'job':job,'exit_code':code,'normal_exit':code==0,'error':error,'wall_seconds':time.time()-start,'command':cmd,'result_exists':(dst/'result.json').exists()}
        (batch/(name+'.status.json')).write_text(json.dumps(result,indent=2)+'\n');print(name,'exit',code,'seconds',round(time.time()-start,1),flush=True)
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(run,jobs))
    (batch/'process_status.json').write_text(json.dumps(results,indent=2)+'\n')
    return 0 if all(r['normal_exit'] and r['result_exists'] for r in results) else 1
if __name__=='__main__':raise SystemExit(main())
