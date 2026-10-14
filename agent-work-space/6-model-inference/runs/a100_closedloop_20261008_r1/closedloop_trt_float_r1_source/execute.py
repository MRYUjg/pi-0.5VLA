"""Immutable attempt sources and actual exit codes in an exclusive new run."""
import argparse,json,pathlib,shutil,subprocess,time
ROOT=pathlib.Path('/mnt/data/vla-policy/openpi/agent-work-space/6-model-inference')
VERSION='a100_closedloop_20261008_r1';RUN=ROOT/'runs'/VERSION
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('script');p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
folder=RUN/(a.name+'_source');folder.mkdir(exist_ok=False)
for file in (ROOT/'versions'/VERSION).iterdir():
    if file.is_file():shutil.copy2(file,folder/file.name);(folder/file.name).chmod(0o444)
command=['docker','exec','pi05-a100-build-20261008-r1','/.venv/bin/python','/work/runs/'+VERSION+'/'+folder.name+'/'+a.script,*a.args]
start=time.time()
with (RUN/(a.name+'.log')).open('x') as f:code=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT).returncode
with (RUN/(a.name+'_exit.json')).open('x') as f:json.dump({'command':command,'exit_code':code,'seconds':time.time()-start},f,indent=2)
print('EXIT',a.name,code,flush=True);raise SystemExit(code)
