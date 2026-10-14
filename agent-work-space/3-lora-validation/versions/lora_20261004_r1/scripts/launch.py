import argparse,json,os,pathlib,subprocess,time,threading
V=pathlib.Path(__file__).resolve().parents[1];ROOT=V.parents[1];REPO=ROOT.parents[1]
def command(script,out):
 return ['docker','run','--rm','--gpus','all','--network','none','--memory','28g','--shm-size','1g','--cpus','6',
 '-e','PYTHONPATH=/app/src:/app/packages/openpi-client/src:/app','-e','HOME=/tmp','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENPI_DATA_HOME=/openpi_assets','-e','HF_HUB_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1','-e','HF_HOME=/tmp/hf','-e','HF_LEROBOT_HOME=/app/agent-work-space/2-task-collection/lerobot_exports/coverage_lerobot_r1/datasets','-e','XLA_PYTHON_CLIENT_MEM_FRACTION=0.85','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false','-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','WANDB_MODE=disabled',
 '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':/app/agent-work-space/3-lora-validation','-v','/mnt/data/vla-policy/caches/openpi:/openpi_assets:ro','--workdir','/app','--entrypoint','/.venv/bin/python','openpi_server:latest','/app/'+str((V/'scripts'/script).relative_to(REPO)),'--out','/app/'+str(out.relative_to(REPO))]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--phase',choices=['prepare','train','reload'],required=True);a=p.parse_args();assert pathlib.Path(a.run).name==a.run and a.run not in ('.','..')
 out=ROOT/'runs'/a.run;logs=ROOT/'runs'/(a.run+'_execution');logs.mkdir(parents=True,exist_ok=True);cmd=command(a.phase+'.py',out)
 stop=threading.Event();samples=[]
 def monitor():
  while not stop.is_set():
   samples.append({'time':time.time(),'gpu':subprocess.getoutput('nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')});stop.wait(2)
 th=threading.Thread(target=monitor);th.start();start=time.time()
 try:
  with (logs/(a.phase+'.log')).open('x') as f:
   code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
 finally:stop.set();th.join()
 with (logs/(a.phase+'.status.json')).open('x') as f:json.dump({'exit_code':code,'seconds':time.time()-start,'command':cmd,'gpu_samples':samples},f,indent=2)
 print(a.phase,'exit',code,'seconds',time.time()-start,flush=True);return code
if __name__=='__main__':raise SystemExit(main())
