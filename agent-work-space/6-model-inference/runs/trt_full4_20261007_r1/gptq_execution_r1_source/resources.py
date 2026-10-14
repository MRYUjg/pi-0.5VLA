"""Same single-policy resource path, isolated launches and cold NVML capture."""
import argparse
import hashlib
import importlib.util
import json
import pathlib
import shutil
import subprocess
import time
import urllib.request

REPO=pathlib.Path('/mnt/data/vla-policy/openpi');ROOT=REPO/'agent-work-space/6-model-inference'
VERSION='trt_full4_20261007_r1';RUN=ROOT/'runs'/VERSION;PORT=18883
manifest=ROOT/'releases/trt_t4_20261007_r1/manifest.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='b9875e01fb7820d4c1e49b289421e3daa130ebe3b309001d374198232375c1c0'
entry=json.loads(manifest.read_text())['files_relative_to_6_model_inference']['versions/trt_t4_20261007_r1/memory.py']
assert hashlib.sha256((ROOT/'versions/trt_t4_20261007_r1/memory.py').read_bytes()).hexdigest()==entry['sha256']
spec=importlib.util.spec_from_file_location('frozen_full4_nvml',ROOT/'versions/trt_t4_20261007_r1/memory.py')
memory=importlib.util.module_from_spec(spec);spec.loader.exec_module(memory)
IMAGE='sha256:cf96556ab8b55eb3f80770d2a684d6961c0128c8100660a8cc4c7e61ecc60369'


def write(path,data):
    with path.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')


def run(variant,attempt,count):
    label='resource_'+variant+'_'+attempt;out=RUN/label;out.mkdir(exist_ok=False)
    snapshot=RUN/(label+'_source');snapshot.mkdir(exist_ok=False)
    for p in (ROOT/'versions'/VERSION).iterdir():
        if p.is_file():shutil.copy2(p,snapshot/p.name);(snapshot/p.name).chmod(0o444)
    source='/work/'+str(snapshot.relative_to(ROOT));name='pi05-full4-'+label.replace('_','-')
    monitor=memory.Monitor(RUN/(label+'_memory.jsonl'));start=time.monotonic();ready=None;state=None;pid=None
    try:
        cmd=['docker','run','-d','--name',name,'--gpus','all','--ipc','host',
            '-p',f'127.0.0.1:{PORT}:{PORT}','-e','PYTHONPATH=/app/src:/app/packages/openpi-client/src',
            '-e','OPENPI_DATA_HOME=/openpi_assets','-e','HF_HUB_OFFLINE=1','-e','JAX_PLATFORMS=cpu',
            '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':/work:ro','-v',str(out)+':/out',
            '-w','/app','--entrypoint','/.venv/bin/python',IMAGE,source+'/server.py',
            '--variant',variant,'--out','/out/server','--port',str(PORT)]
        subprocess.run(cmd,check=True,capture_output=True,text=True)
        info=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0];pid=info['State']['Pid']
        write(out/'launch.json',{'command':cmd,'host_pid':pid,'image':info['Image'],'container_id':info['Id']})
        while ready is None:
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health',timeout=2) as response:ready=json.loads(response.read())
                assert ready['variant']==variant
            except Exception:
                status=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State']
                assert status['Running'],status
                assert time.monotonic()-start<600,'ready timeout'
                time.sleep(1)
        ready_at=time.monotonic();write(out/'ready_host.json',{'ready':ready,'seconds':ready_at-start})
        bench_start=time.monotonic()
        command=['docker','exec',name,'/.venv/bin/python',source+'/client.py','--out','/out/benchmark',
                 '--variant',variant,'--port',str(PORT),'--count',str(count)]
        with (out/'client.log').open('x') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
        bench_end=time.monotonic();write(out/'client_exit.json',{'exit_code':code,'command':command})
        assert code==0,code
    finally:
        try:
            request=urllib.request.Request(f'http://127.0.0.1:{PORT}/shutdown',data=b'{}')
            urllib.request.urlopen(request,timeout=5).read()
        except Exception:pass
        if pid is not None:
            for _ in range(10):
                state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State']
                if not state['Running']:break
                time.sleep(.5)
            if state['Running']:subprocess.run(['docker','stop','-t','2',name],check=True,capture_output=True)
            state=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]['State']
            (out/'server.log').write_bytes(subprocess.check_output(['docker','logs',name],stderr=subprocess.STDOUT))
            write(out/'container_exit.json',state)
        samples=monitor.close()
    assert state['ExitCode']==0,state
    vals=[(s['monotonic_seconds'],s['processes'].get(str(pid),0)) for s in samples if s['processes'].get(str(pid),0)>0]
    assert vals
    active=[v for t,v in vals if bench_start<=t<=bench_end];assert active
    n=max(1,len(active)//10);growth=max(active[-n:])-max(active[:n])
    foreign=sorted({k for s in samples for k in s['processes'] if k!=str(pid)})
    peak=max(v for _,v in vals);device=max(s['device_used_bytes'] for s in samples)
    write(out/'memory_summary.json',{'process_peak_bytes':peak,'device_peak_bytes':device,'samples':len(samples),
        'sample_interval_ms':20,'includes_launch_load_warmup_shutdown':True,'policy_host_pid':pid,
        'warm_growth_bytes':growth,'warm_growth_screen_passed':growth<=67108864,
        'foreign_GPU_compute_pids':foreign,'isolated_compute':not foreign,
        'process_6_4GB_screen_passed':peak<=6400000000,'device_7_2GB_screen_passed':device<=7200000000,
        'two_hour_run':False,'actual_4070_or_8GB_hardware_test':False,'variant':variant,'count':count})
    assert not foreign and growth<=67108864,(foreign,growth)
    print('RESOURCE COMPLETE',variant,'process',peak,'device',device,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',nargs='+',required=True)
    p.add_argument('--attempt',default='r1');p.add_argument('--count',type=int,default=1000);a=p.parse_args()
    for variant in a.variant:run(variant,a.attempt,a.count)
