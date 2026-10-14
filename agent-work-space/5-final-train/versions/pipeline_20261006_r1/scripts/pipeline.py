"""Sequential release gates; detached host execution with an explicit durable status."""
import argparse,hashlib,json,os,pathlib,subprocess,sys,time,traceback
V=pathlib.Path(__file__).resolve().parents[1];ROOT=V.parents[1];REPO=ROOT.parents[1]
VERSIONS=('collection_20261005_r4','collection_20261006_r1','validation_20261006_r1','conversion_20261006_r1','lora_20261006_r1',V.name)
RUN='spatial_lora_20261006_r1'
read=lambda p:json.loads(p.read_text())
def verify():
 for version in VERSIONS:
  for rel,want in read(ROOT/'releases'/version/'manifest.json')['files'].items():
   assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==want,rel

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',default='pipeline_20261006_r1');a=ap.parse_args()
 assert pathlib.Path(a.name).name==a.name and a.name not in ('.','..')
 out=ROOT/'runs'/a.name;out.mkdir(parents=True,exist_ok=False)
 state={'pid':os.getpid(),'started':time.time(),'state':'running','stage':'waiting_for_resumed_collection','stages':[],'versions':VERSIONS}
 def status():
  state['updated']=time.time();p=out/'status.tmp';p.write_text(json.dumps(state,indent=2));p.replace(out/'status.json')
 def run(name,cmd):
  verify();state['stage']=name;status();start=time.time()
  with (out/(name+'.log')).open('x') as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
  state['stages'].append({'stage':name,'exit_code':code,'seconds':time.time()-start,'command':cmd});status()
  if code:raise RuntimeError(name+' failed with exit '+str(code))
 try:
  verify();status();reg=ROOT/'runs/regression_20261005_r1/summary.json';start=time.time()
  while not reg.exists():
   if time.time()-start>3*3600:raise TimeoutError('regression summary missing after three hours')
   time.sleep(15)
  regression=read(reg);assert regression['pass'],'teacher regression not admitted'
  state['eligible_tasks']=regression['eligible_tasks'];state['excluded_tasks']=regression['excluded_tasks'];status()
  original=ROOT/'runs/formal_20261005_r1/summary.json';start=time.time()
  while not original.exists():
   if time.time()-start>8*3600:raise TimeoutError('resumed original collection summary missing after eight hours')
   time.sleep(15)
  orig=read(original);assert orig['attempts']==300 and not orig['duplicate_initializations']
  state['original_collection_pass']=orig['pass'];state['original_counts']=orig['counts'];status()
  run('supplemental_validation',[sys.executable,str(ROOT/'versions/validation_20261006_r1/scripts/batch.py'),'--stage','formal','--workers','6','--supplement'])
  run('merged_release',[sys.executable,str(V/'scripts/release.py')])
  summary=read(ROOT/'runs/formal_merged_20261006_r1/summary.json');assert summary['pass']
  # One full recorded demonstration MP4 for every admitted task.
  sys.path.insert(0,str(ROOT/'versions/collection_20261005_r4/scripts'));from run import command
  selected={}
  for rel in summary['records']:
   ep=ROOT/rel;attempt=read(ep.with_suffix('.summary.json'))
   if attempt['accepted'] and attempt['scene']['split']=='train':selected.setdefault(attempt['scene']['task_id'],ep)
  video_dir=ROOT/'videos/formal_20261006_r1';video_dir.mkdir(parents=True,exist_ok=False);index=[]
  for tid,ep in sorted(selected.items()):
   dest=video_dir/f'task{tid:02d}.mp4'
   cmd=command(V/'scripts/video.py',['--episode','/app/'+str(ep.relative_to(REPO)),
    '--out','/app/'+str(dest.relative_to(REPO)),'--label',f'LIBERO Spatial task {tid}'])
   run(f'video_task{tid:02d}',cmd);index.append({'task_id':tid,'episode':str(ep.relative_to(ROOT)),'video':str(dest.relative_to(ROOT))})
  (video_dir/'index.json').write_text(json.dumps(index,indent=2))
  run('conversion',[sys.executable,str(ROOT/'versions/conversion_20261006_r1/scripts/run.py')])
  assert read(ROOT/'lerobot_exports/spatial_formal_r2/validation.json')['pass']
  for phase in ('prepare','smoke','train','reload'):
   run(phase,[sys.executable,str(ROOT/'versions/lora_20261006_r1/scripts/launch.py'),'--run',RUN,'--phase',phase])
  assert read(ROOT/'runs'/RUN/'reload_result.json')['pass']
  state.update(state='completed',stage='training_and_reload_complete',scope='formal collection + dataset validation + LoRA + reload; closed-loop evaluation still pending')
  status();print(json.dumps(state),flush=True);return 0
 except BaseException as e:
  state.update(state='stopped',error=repr(e),traceback=traceback.format_exc());status();traceback.print_exc();return 1
if __name__=='__main__':raise SystemExit(main())
