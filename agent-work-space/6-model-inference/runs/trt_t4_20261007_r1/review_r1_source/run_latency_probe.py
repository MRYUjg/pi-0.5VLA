"""Separate one-client measurements; preserve the main paired experiment."""
import argparse
import json
import subprocess

from memory import Monitor
from resources import ROOT, REPO, RUN, PORT, launch, snapshot, stop, write
from run_closedloop import SIM_IMAGE


def probe(variant,attempt):
    label='latency_'+variant+'_'+attempt;source=snapshot(label);out=RUN/label
    name='pi05-trt-t4-'+label.replace('_','-');monitor=Monitor(RUN/(label+'_memory.jsonl'))
    try:
        pid,ready=launch(variant,name,out,source);write(out/'ready_host.json',ready)
        command=['docker','run','--rm','--name','pi05-t4-latency-sim-'+variant,'--network','container:'+name,
            '--cpus','2','--memory','3g','-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero',
            '-e','PYTHONDONTWRITEBYTECODE=1','-e','MUJOCO_GL=osmesa','-e','LP_NUM_THREADS=1',
            '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','EXTRA_JOINT_PD_PROFILE=legacy',
            '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':/work:ro','-v',str(out)+':/out','-w','/app',
            '--entrypoint','/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python',SIM_IMAGE,
            source+'/latency_probe.py','--variant',variant,'--out','/out/episode','--port',str(PORT),
            '--scenes','/work/runs/trt_t4_20261007_r1/preflight_r1/scenes.json']
        with (out/'simulation.log').open('x') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
        write(out/'simulation_exit.json',{'command':command,'exit_code':code})
    finally:
        if out.exists():state=stop(name,out)
        samples=monitor.close()
    write(out/'memory_summary.json',{'host_policy_pid':pid,'process_peak_bytes':max(s['processes'].get(str(pid),0) for s in samples),
        'device_peak_bytes':max(s['device_used_bytes'] for s in samples),'single_simulation_client':True,
        'sample_interval_seconds':.02,'actual_4070_accepted':False})
    assert code==0 and state['ExitCode']==0,(code,state)
    print('SINGLE CLIENT LATENCY PROBE COMPLETE',variant,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',nargs='+',required=True);p.add_argument('--attempt',default='r1')
    a=p.parse_args()
    for variant in a.variant:probe(variant,a.attempt)
