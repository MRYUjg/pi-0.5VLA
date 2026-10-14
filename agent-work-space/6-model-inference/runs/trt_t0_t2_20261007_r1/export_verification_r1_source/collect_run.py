"""Host orchestrator, append-only attempts with actual container exit codes."""
import concurrent.futures
import json
import pathlib
import subprocess
import time

REPO=pathlib.Path('/mnt/data/vla-policy/openpi')
ROOT=REPO/'agent-work-space/6-model-inference'
RUN=ROOT/'runs/trt_t0_t2_20261007_r1'
OUT=RUN/'panda_calibration_r1'
OUT.mkdir(exist_ok=False)


def collect(task):
    cmd=['docker','run','--rm','--network','container:pi05-trt-t0t2-20261007-r1','--cpus','2','--memory','3g',
         '-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero',
         '-e','PYTHONDONTWRITEBYTECODE=1','-e','MUJOCO_GL=osmesa','-e','LP_NUM_THREADS=1',
         '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','EXTRA_JOINT_PD_PROFILE=legacy',
         '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':/work','-w','/app',
         '--entrypoint','/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python',
         'libero:latest','/work/versions/trt_t0_t2_20261007_r1/collect.py',
         '--task',str(task),'--out',f'/work/runs/trt_t0_t2_20261007_r1/panda_calibration_r1/task{task:02d}']
    start=time.time()
    with (OUT/f'task{task:02d}.log').open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
    result={'task':task,'exit_code':code,'seconds':time.time()-start,'command':cmd}
    (OUT/f'task{task:02d}_exit.json').write_text(json.dumps(result,indent=2))
    print('COLLECTION EXIT',task,code,flush=True)
    return result


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results=list(pool.map(collect,range(10)))
(OUT/'exit_summary.json').write_text(json.dumps({'results':results,'passed':all(r['exit_code']==0 for r in results)},indent=2))
assert all(r['exit_code']==0 for r in results),results
