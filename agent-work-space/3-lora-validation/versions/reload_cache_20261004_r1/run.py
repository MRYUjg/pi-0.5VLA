"""Retest frozen reload with the training process's persistent XLA compilation cache."""
import argparse,json,pathlib,subprocess,sys,time
V=pathlib.Path(__file__).resolve().parent;ROOT=V.parents[1];REPO=ROOT.parents[1]
sys.path.insert(0,str(ROOT/'versions/lora_20261004_r1/scripts'))
from launch import command
p=argparse.ArgumentParser();p.add_argument('--run',default='smoke_train_r2');a=p.parse_args();assert pathlib.Path(a.run).name==a.run
out=ROOT/'runs'/a.run;logs=ROOT/'runs'/(a.run+'_execution');cmd=command('reload.py',out)
cmd[3:3]=['-e','JAX_COMPILATION_CACHE_DIR=/app/'+str((out/'jax_cache').relative_to(REPO)),'-e','JAX_LOG_COMPILES=1']
start=time.time()
with (logs/'reload_cached.log').open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
with (logs/'reload_cached.status.json').open('x') as f:json.dump({'exit_code':code,'seconds':time.time()-start,'command':cmd,'purpose':'same frozen verification and threshold; reuse training compilation cache'},f,indent=2)
print('reload_cached',code,time.time()-start,flush=True);raise SystemExit(code)
