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
VERSION='a100_closedloop_20261008_r1'
RUN=ROOT/'runs'/VERSION
IMAGE='sha256:cf96556ab8b55eb3f80770d2a684d6961c0128c8100660a8cc4c7e61ecc60369'
PORT=18884


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
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENPI_DATA_HOME=/openpi_assets','-e','HF_HUB_OFFLINE=1','-e','JAX_PLATFORMS=cpu',
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

