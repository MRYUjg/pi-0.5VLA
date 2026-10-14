"""Sequential GPU models, two CPU simulation workers, real exits and NVML."""
import argparse
import concurrent.futures
import json
import pathlib
import subprocess
import time

from memory import Monitor
from resources import ROOT, REPO, RUN, PORT, launch, snapshot, stop, write

SIM_IMAGE='sha256:1ac69906b0b877c9fe1c2964794aa5b90112850da01967a445658e41a958f284'


def collect(variant,task,out,source,container):
    command=['docker','run','--rm','--name',f'pi05-a100-sim-{variant}-{task:02d}',
        '--network','container:'+container,'--cpus','2','--memory','3g',
        '-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','MUJOCO_GL=osmesa','-e','LP_NUM_THREADS=1',
        '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','EXTRA_JOINT_PD_PROFILE=legacy',
        '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':/work:ro','-v',str(out)+':/out','-w','/app',
        '--entrypoint','/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python',
        SIM_IMAGE,source+'/closedloop.py','--task',str(task),'--variant',variant,
        '--out',f'/out/task{task:02d}','--port',str(PORT),
        '--scenes','/work/runs/a100_closedloop_20261008_r1/preflight_r1/scenes.json']
    start=time.monotonic()
    with (out/f'task{task:02d}.log').open('x') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
    result={'task':task,'variant':variant,'exit_code':code,'seconds':time.monotonic()-start,'command':command}
    write(out/f'task{task:02d}_exit.json',result)
    print('CLOSED LOOP TASK',variant,task,'EXIT',code,flush=True)
    return result


def run(variant,attempt):
    label='closedloop_'+variant+'_'+attempt;source=snapshot(label);out=RUN/label
    container='pi05-a100-'+label.replace('_','-')
    monitor=Monitor(RUN/(label+'_memory.jsonl'));pid=None
    try:
        pid,ready=launch(variant,container,out,source)
        write(out/'ready_host.json',ready)
        command=['docker','exec',container,'/.venv/bin/python',source+'/pilot.py','--variant',variant,'--out','/out/pilot','--port',str(PORT)]
        with (out/'pilot.log').open('x') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
        write(out/'pilot_exit.json',{'command':command,'exit_code':code});assert code==0,code
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results=list(pool.map(lambda task:collect(variant,task,out,source,container),range(10)))
    finally:
        if out.exists():state=stop(container,out)
        samples=monitor.close()
    write(out/'exit_summary.json',{'results':results,'all_tasks_exit_zero':all(r['exit_code']==0 for r in results)})
    own=[s['processes'].get(str(pid),0) for s in samples]
    foreign=sorted(set(k for s in samples for k in s['processes'] if k!=str(pid)))
    write(out/'memory_summary.json',{'host_policy_pid':pid,'process_peak_bytes':max(own),
        'device_peak_bytes':max(s['device_used_bytes'] for s in samples),'other_compute_pids':foreign,
        'single_model_resident':not foreign,'sample_interval_seconds':.02,'NVML_samples':len(samples),
        'includes_renderer_context_if_any':True,'actual_4070_accepted':False,'device':'A100-SXM4-40GB'})
    assert all(r['exit_code']==0 for r in results) and state['ExitCode']==0 and not foreign,(results,state,foreign)
    print('CLOSED LOOP VARIANT COMPLETE',variant,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',nargs='+',required=True);p.add_argument('--attempt',default='r1')
    a=p.parse_args()
    for variant in a.variant:run(variant,a.attempt)
