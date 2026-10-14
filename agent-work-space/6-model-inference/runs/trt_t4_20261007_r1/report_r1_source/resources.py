"""Host orchestration: sequential isolated policy containers and cold NVML audit."""
import argparse
import json
import pathlib
import shutil
import subprocess
import time
import urllib.request

from memory import Monitor

REPO=pathlib.Path('/mnt/data/vla-policy/openpi')
ROOT=REPO/'agent-work-space/6-model-inference'
VERSION='trt_t4_20261007_r1'
RUN=ROOT/'runs'/VERSION
IMAGE='sha256:cf96556ab8b55eb3f80770d2a684d6961c0128c8100660a8cc4c7e61ecc60369'
PORT=18879


def write(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')


def snapshot(name):
    folder=RUN/(name+'_source');folder.mkdir(exist_ok=False)
    for p in (ROOT/'versions'/VERSION).iterdir():
        if p.is_file() and p.suffix in ['.py','.json']:shutil.copy2(p,folder/p.name)
    return '/work/'+str(folder.relative_to(ROOT))


def launch(variant,name,output,source):
    output.mkdir(exist_ok=False)
    command=['docker','run','-d','--name',name,'--gpus','all','--ipc','host',
        '-p',f'127.0.0.1:{PORT}:{PORT}','-e','PYTHONPATH=/app/src:/app/packages/openpi-client/src',
        '-e','OPENPI_DATA_HOME=/openpi_assets','-e','HF_HUB_OFFLINE=1','-e','JAX_PLATFORMS=cpu',
        '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':/work:ro','-v',str(output)+':/out',
        '-w','/app','--entrypoint','/.venv/bin/python',IMAGE,source+'/server.py',
        '--variant',variant,'--out','/out/server','--port',str(PORT)]
    subprocess.run(command,check=True,stdout=subprocess.PIPE,text=True)
    info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
    write(output/'launch.json',{'command':command,'host_pid':info['State']['Pid'],'image':info['Image'],
        'container_id':info['Id'],'mounts':info['Mounts']})
    start=time.monotonic()
    while True:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=2) as r:
                ready=json.loads(r.read())
            assert ready['variant']==variant
            return info['State']['Pid'],ready
        except Exception:
            state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State']
            assert state['Running'],state
            if time.monotonic()-start>600:raise RuntimeError('ready timeout')
            time.sleep(1)


def stop(name,out):
    try:
        req=urllib.request.Request(f'http://127.0.0.1:{PORT}/shutdown',data=b'{}')
        urllib.request.urlopen(req,timeout=5).read()
    except Exception:pass
    for _ in range(10):
        state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State']
        if not state['Running']:break
        time.sleep(.5)
    if state['Running']:subprocess.run(['docker','stop','-t','2',name],check=True,stdout=subprocess.PIPE)
    state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State']
    logs=subprocess.check_output(['docker','logs',name],stderr=subprocess.STDOUT)
    with (out/'server.log').open('xb') as f:f.write(logs)
    write(out/'container_exit.json',state)
    return state


def resource(variant,attempt):
    label='resource_'+variant+'_'+attempt
    source=snapshot(label);out=RUN/label
    name='pi05-trt-t4-'+label.replace('_','-')
    start=time.monotonic();monitor=Monitor(RUN/(label+'_memory.jsonl'))
    pid=None;code=None;state=None
    try:
        pid,ready=launch(variant,name,out,source)
        ready_at=time.monotonic();write(out/'ready_host.json',{'cold_launch_until_ready_seconds':ready_at-start,'registration':ready})
        bench_start=time.monotonic()
        command=['docker','exec',name,'/.venv/bin/python',source+'/benchmark_client.py',
            '--variant',variant,'--out','/out/benchmark','--port',str(PORT)]
        with (out/'benchmark.log').open('x') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
        bench_end=time.monotonic()
        write(out/'benchmark_exit.json',{'exit_code':code,'command':command})
    finally:
        if out.exists():state=stop(name,out)
        samples=monitor.close()
    assert pid is not None and code==0 and state['ExitCode']==0,(pid,code,state)
    pid_key=str(pid)
    vals=[(s['monotonic_seconds'],s['processes'].get(pid_key,0)) for s in samples if s['processes'].get(pid_key,0)>0]
    assert vals,'NVML did not witness the policy process'
    active=[v for t,v in vals if bench_start<=t<=bench_end]
    assert len(active)>100
    segment=max(1,len(active)//10)
    before=max(active[:segment]);after=max(active[-segment:]);growth=after-before
    foreign=sorted(set(k for s in samples for k in s['processes'] if k!=pid_key))
    report={'variant':variant,'development_device':'RTX4090','host_policy_pid':pid,
        'process_peak_bytes':max(v for _,v in vals),'device_peak_bytes':max(s['device_used_bytes'] for s in samples),
        'process_warm_first_decile_max_bytes':before,'process_warm_last_decile_max_bytes':after,
        'process_warm_growth_bytes':growth,'growth_gate_bytes':67108864,'no_sustained_growth_screen_passed':growth<=67108864,
        'NVML_sample_interval_seconds':.02,'samples':len(samples),'includes_launch_load_warmup_and_shutdown':True,
        'other_GPU_compute_pids':foreign,'isolated_compute_process':not foreign,'OOM':False,
        'cold_launch_until_ready_seconds':ready_at-start,'continuous_1000_request_seconds':bench_end-bench_start,
        'process_6_4GB_budget_screen':max(v for _,v in vals)<=6400000000,
        'device_7_2GB_budget_screen':max(s['device_used_bytes'] for s in samples)<=7200000000,
        'two_hour_continuous_test':False,'actual_4070_accepted':False}
    write(out/'memory_summary.json',report)
    assert report['isolated_compute_process'] and report['no_sustained_growth_screen_passed'],report
    print('RESOURCE COMPLETE',variant,'PROCESS',report['process_peak_bytes'],'DEVICE',report['device_peak_bytes'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--attempt',default='r1');p.add_argument('--variant',nargs='+',required=True)
    a=p.parse_args()
    for variant in a.variant:resource(variant,a.attempt)
