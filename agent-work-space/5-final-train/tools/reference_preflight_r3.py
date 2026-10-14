"""Reference preparation checks; fixed old successful seeds, no teacher/data release."""
import concurrent.futures,hashlib,json,os,pathlib,subprocess,sys,time
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parents[1];REPO=ROOT.parents[1];V=ROOT/'versions/collection_20261005_r3'
sys.path.insert(0,str(V/'scripts'))
from run import command
BATCH=ROOT/'runs/reference_preflight_20261005_r3'
def write(p,v):
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def job(s):
 ep=BATCH/f'task{s["task_id"]:02d}_s{s["seed"]}';cmd=command('collect.py',['--seed',str(s['seed']),'--out','/app/'+str(ep.relative_to(REPO)),'--preflight']);start=time.time()
 with ep.with_suffix('.log').open('x') as f:
  try:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900).returncode
  except subprocess.TimeoutExpired:rc=-999
 r={'task_id':s['task_id'],'seed':s['seed'],'exit_code':rc,'seconds':time.time()-start,'command':cmd}
 for name in ('initialization','preflight'):
  f=ep/(name+'.json')
  if f.exists():r[name]=json.loads(f.read_text())
 r['pass']=rc==0 and r.get('preflight',{}).get('same_host_observer_unchanged',False)
 write(ep.with_suffix('.status.json'),r)
 print(json.dumps({'task':r['task_id'],'pass':r['pass'],'seconds':r['seconds'],'initialization_errors':r.get('initialization',{}).get('errors'),'reference':r.get('preflight',{}).get('reference_errors')},ensure_ascii=False),flush=True)
 return r
if __name__=='__main__':
 m=json.loads((ROOT/'releases/collection_20261005_r3/manifest.json').read_text())
 for p,h in m['files'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 BATCH.mkdir(parents=True,exist_ok=False)
 jobs=[s for s in json.loads((V/'scenes.json').read_text())['jobs'] if s['split']=='reference']
 write(BATCH/'protocol.json',{'jobs':jobs,'source_manifest_sha256':hashlib.sha256((ROOT/'releases/collection_20261005_r3/manifest.json').read_bytes()).hexdigest(),'launcher_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()})
 with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:rows=list(pool.map(job,jobs))
 write(BATCH/'summary.json',{'pass':all(r['pass'] for r in rows),'records':rows})
 subprocess.run(['docker','run','--rm','--network','none','-v',str(ROOT)+':/output','--entrypoint','chown','libero:latest','-R',f'{os.getuid()}:{os.getgid()}','/output'],check=True)
 raise SystemExit(int(not all(r['pass'] for r in rows)))
