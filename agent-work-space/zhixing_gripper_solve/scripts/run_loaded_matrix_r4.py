#!/usr/bin/env python3
"""Host launcher. Records actual subprocess exit codes; never masks shutdown faults."""
import argparse, concurrent.futures, datetime, hashlib, json, pathlib, subprocess, time
ROOT=pathlib.Path('/mnt/data/vla-policy/openpi')
SOLVE=ROOT/'agent-work-space/zhixing_gripper_solve'
ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--seeds',default='47100,47101,47103');ap.add_argument('--modes',default='chain,pull');ap.add_argument('--forces',default='0,2,5,10');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
batch=SOLVE/'evidence'/a.batch;batch.mkdir(exist_ok=False,parents=True)
seeds=[int(x) for x in a.seeds.split(',')];jobs=[]
for grip in ['ZhixingGripper','PandaGripper']:
 for mode in a.modes.split(','):
  for seed in (seeds if mode=='chain' else seeds[:1]):
   for force in ([0] if mode=='chain' else [float(x) for x in a.forces.split(',')]):
    jobs.append({'gripper':grip,'mode':mode,'seed':seed,'force':force})
(batch/'protocol.json').write_text(json.dumps({'jobs':jobs,'note':'Diagnostic only. No official acceptance gates altered; each load is an independent grasp.','force_duration_s':2,'relative_slip_limit_mm':5,'bilateral_fraction_min':.95,'baseline_relative_slip_limit_mm':2,'baseline_lift_min_mm':30,'workers':a.workers},indent=2))
def run(j):
 name='%s_%s_s%d_f%g'%(j['gripper'],j['mode'],j['seed'],j['force']);dst=batch/name
 cmd=['docker','run','--rm','--network','none','--cpus','2','--memory','5g','-e','MUJOCO_GL=osmesa','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero','-v',str(ROOT)+':/app:ro','-v',str(SOLVE)+':/app/agent-work-space/zhixing_gripper_solve','-w','/app','--entrypoint','/app/agent-work-space/zhixing_gripper_solve/runtime/py310_mj326/bin/python','libero:latest','/app/agent-work-space/zhixing_gripper_solve/scripts/loaded_diagnostics_r5.py','--gripper',j['gripper'],'--mode',j['mode'],'--seed',str(j['seed']),'--force',str(j['force']),'--out','/app/'+str(dst.relative_to(ROOT))]
 t=time.time()
 with (batch/(name+'.log')).open('x') as log:
  proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 result={'job':j,'exit_code':proc.returncode,'wall_seconds':time.time()-t,'command':cmd,'result_exists':(dst/'result.json').exists(),'normal_exit':proc.returncode==0}
 (batch/(name+'.status.json')).write_text(json.dumps(result,indent=2))
 print(name,'exit',proc.returncode,'seconds',round(time.time()-t,1),flush=True)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
 results=list(pool.map(run,jobs))
(batch/'process_status.json').write_text(json.dumps(results,indent=2))
raise SystemExit(0 if all(x['normal_exit'] and x['result_exists'] for x in results) else 1)
