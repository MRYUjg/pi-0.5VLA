"""Resume after a completed data release; preserve earlier failures and outputs."""
import argparse,hashlib,json,os,pathlib,shutil,subprocess,sys,time,traceback
V=pathlib.Path(__file__).resolve().parents[1];ROOT=V.parents[1];REPO=ROOT.parents[1]
VERSIONS=('conversion_20261007_r1','lora_20261007_r1',V.name)
RUN='spatial_lora_20261007_r1'
read=lambda p:json.loads(p.read_text())
def verify():
 for name in VERSIONS:
  for rel,want in read(ROOT/'releases'/name/'manifest.json')['files'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==want,rel

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',default='pipeline_20261007_r1');a=p.parse_args()
 assert pathlib.Path(a.name).name==a.name and a.name not in ('.','..')
 out=ROOT/'runs'/a.name;out.mkdir(parents=True,exist_ok=False)
 state={'pid':os.getpid(),'started':time.time(),'state':'running','stage':'conversion','stages':[],'versions':VERSIONS,'training_run':RUN,'dataset_export':'spatial_formal_r3'}
 def status():
  state['updated']=time.time();temp=out/'status.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(out/'status.json')
 def run(name,cmd):
  verify();state['stage']=name;status();start=time.time()
  with (out/(name+'.log')).open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
  state['stages'].append({'stage':name,'exit_code':code,'seconds':time.time()-start,'command':cmd});status()
  if code:raise RuntimeError(name+' failed with exit '+str(code))
 try:
  verify();status();release=read(ROOT/'runs/formal_merged_20261006_r1/summary.json');assert release['pass'] and release['attempts']==310
  state['data_release_pass']=True;state['counts']=release['counts'];status()
  assert len(read(ROOT/'videos/formal_20261006_r1/index.json'))==5
  run('conversion',[sys.executable,str(ROOT/'versions/conversion_20261007_r1/scripts/run.py')])
  validation=read(ROOT/'lerobot_exports/spatial_formal_r3/validation.json');assert validation['pass'] and validation['frames_verified']==170724
  state['dataset_validation_pass']=True;status()
  assert shutil.disk_usage(ROOT).free>=60*1024**3,'less than 60GiB for dataset reader cache and full checkpoints; preserve all sources and stop'
  for phase in ('prepare','smoke','train','reload'):
   run(phase,[sys.executable,str(ROOT/'versions/lora_20261007_r1/scripts/launch.py'),'--run',RUN,'--phase',phase])
  assert read(ROOT/'runs'/RUN/'reload_result.json')['pass']
  state.update(state='completed',stage='training_and_reload_complete',scope='formal collection/data validation/LoRA/reload complete; closed-loop policy evaluation pending');status();return 0
 except BaseException as e:
  state.update(state='stopped',error=repr(e),traceback=traceback.format_exc());status();traceback.print_exc();return 1
if __name__=='__main__':raise SystemExit(main())
