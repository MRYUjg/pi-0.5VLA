"""Host launcher: isolated GPU phases, process status, resource samples, no overwrite."""
import argparse,json,os,pathlib,subprocess,time,threading,hashlib
V=pathlib.Path(__file__).resolve().parents[1];ROOT=V.parents[1];REPO=ROOT.parents[1]
rel=lambda p:'/app/'+str(p.relative_to(REPO))
def verify():
 for name,want in json.loads((ROOT/'releases'/V.name/'manifest.json').read_text())['files'].items():
  assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==want,name

def command(script,out):
 script=pathlib.Path(script);script=script if script.is_absolute() else V/'scripts'/script
 return ['docker','run','--rm','--gpus','all','--network','none','--memory','27g','--shm-size','1g','--cpus','8',
 '-e','PYTHONPATH=/app/src:/app/packages/openpi-client/src:/app','-e','PYTHONDONTWRITEBYTECODE=1',
 '-e','OPENPI_DATA_HOME=/openpi_assets','-e','HF_HUB_OFFLINE=1','-e','HF_DATASETS_OFFLINE=1',
 '-e','HF_HOME=/tmp/hf','-e','XDG_CACHE_HOME=/tmp/cache',
 '-e','HF_LEROBOT_HOME='+rel(ROOT/'lerobot_exports/spatial_formal_r3/datasets'),
 '-e','XLA_PYTHON_CLIENT_MEM_FRACTION=0.85','-e','XLA_PYTHON_CLIENT_PREALLOCATE=false',
 '-e','OMP_NUM_THREADS=1','-e','OPENBLAS_NUM_THREADS=1','-e','WANDB_MODE=disabled',
 '-v',str(REPO)+':/app:ro','-v',str(ROOT)+':'+rel(ROOT),
 '-v','/mnt/data/vla-policy/caches/openpi:/openpi_assets:ro','--workdir','/app',
 '--entrypoint','/.venv/bin/python','openpi_server:latest',rel(script),'--out',rel(out)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--run',default='spatial_lora_20261007_r1');p.add_argument('--phase',choices=['prepare','smoke','train','reload'],required=True);a=p.parse_args()
 assert pathlib.Path(a.run).name==a.run and a.run not in ('.','..');verify()
 out=ROOT/'runs'/a.run;logs=ROOT/'runs'/(a.run+'_execution');logs.mkdir(parents=True,exist_ok=True)
 cmd=command(('train' if a.phase=='smoke' else a.phase)+'.py',out)
 if a.phase=='smoke':cmd.append('--smoke')
 stop=threading.Event();samples=[]
 def monitor():
  while not stop.is_set():
   samples.append({'time':time.time(),'gpu':subprocess.getoutput('nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits')});stop.wait(5)
 th=threading.Thread(target=monitor);th.start();start=time.time();code=-1
 try:
  with (logs/(a.phase+'.log')).open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
 finally:
  stop.set();th.join()
  if out.exists():subprocess.run(['docker','run','--rm','--network','none','-v',str(out)+':/output','--entrypoint','chown','libero:latest','-R',f'{os.getuid()}:{os.getgid()}','/output'],check=True)
  with (logs/(a.phase+'.status.json')).open('x') as f:json.dump({'exit_code':code,'seconds':time.time()-start,'command':cmd,'gpu_samples':samples},f,indent=2)
 print(a.phase,'exit',code,'seconds',time.time()-start,flush=True);return code
if __name__=='__main__':raise SystemExit(main())
