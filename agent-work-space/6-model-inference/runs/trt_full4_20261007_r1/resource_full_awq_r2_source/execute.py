"""Exclusive named attempts with immutable source snapshots and real exit status."""
import argparse
import json
import pathlib
import shutil
import subprocess
import time

ROOT=pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference')
VERSION='trt_full4_20261007_r1';RUN=ROOT/'runs'/VERSION
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('script');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
assert pathlib.Path(a.name).name==a.name and pathlib.Path(a.script).name==a.script
snapshot=RUN/(a.name+'_source');snapshot.mkdir(exist_ok=False)
for file in (ROOT/'versions'/VERSION).iterdir():
    if file.is_file():
        shutil.copy2(file,snapshot/file.name);(snapshot/file.name).chmod(0o444)
command=['docker','exec','pi05-trt-full4-20261007-r1','/.venv/bin/python',
         '/work/runs/'+VERSION+'/'+snapshot.name+'/'+a.script,*a.args]
start=time.time()
with (RUN/(a.name+'.log')).open('x') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
with (RUN/(a.name+'_exit.json')).open('x') as f:json.dump({'command':command,'exit_code':code,'seconds':time.time()-start},f,indent=2)
print('EXIT',a.name,code,flush=True)
raise SystemExit(code)
