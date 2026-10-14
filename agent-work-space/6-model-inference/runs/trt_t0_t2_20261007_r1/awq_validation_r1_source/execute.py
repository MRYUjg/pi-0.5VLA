"""Host entry retaining true exits, command, log and independent output names."""
import argparse
import json
import pathlib
import subprocess
import time
import shutil

REPO=pathlib.Path('/mnt/data/vla-policy/openpi')
ROOT=REPO/'agent-work-space/6-model-inference'
RUN=ROOT/'runs/trt_t0_t2_20261007_r1'
CONTAINER='pi05-trt-t0t2-20261007-r1'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('script');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
assert pathlib.Path(a.name).name==a.name and a.name not in ('.','..')
assert pathlib.Path(a.script).name==a.script and a.script.endswith('.py')
snapshot=RUN/(a.name+'_source');snapshot.mkdir(exist_ok=False)
for source in (ROOT/'versions/trt_t0_t2_20261007_r1').glob('*.py'):shutil.copy2(source,snapshot/source.name)
cmd=['docker','exec','-e','PYTHONPATH=/app/src:/app/packages/openpi-client/src',
     '-e','OPENPI_DATA_HOME=/openpi_assets','-e','HF_HUB_OFFLINE=1',CONTAINER,'/.venv/bin/python',
     '/work/runs/trt_t0_t2_20261007_r1/'+snapshot.name+'/'+a.script,*a.args]
start=time.time()
with (RUN/(a.name+'.log')).open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
with (RUN/(a.name+'_exit.json')).open('x') as f:json.dump({'command':cmd,'exit_code':code,'seconds':time.time()-start},f,indent=2)
print('EXIT',a.name,code,flush=True)
raise SystemExit(code)
