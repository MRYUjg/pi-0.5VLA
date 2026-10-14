"""Versioned host launcher: collect registered training/validation scenes only."""
import argparse,hashlib,json,pathlib,subprocess,time,concurrent.futures,os
from run import VERSION,COLLECTION,REPO,container_command

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--scene-ids',default='');ap.add_argument('--workers',type=int,choices=[1,2],default=2);a=ap.parse_args()
 assert pathlib.Path(a.batch).name==a.batch and a.batch not in ('.','..')
 specs=json.loads((VERSION/'scenes.json').read_text())['scenes'];ids=a.scene_ids.split(',') if a.scene_ids else [s['id'] for s in specs if s['split']!='test']
 assert len(set(ids))==len(ids);jobs=[next(s for s in specs if s['id']==i) for i in ids];assert all(s['split']!='test' for s in jobs),'held-out test collection is forbidden in this entry'
 dst=COLLECTION/'runs'/a.batch;dst.mkdir(parents=True,exist_ok=False)
 hashes={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in VERSION.rglob('*') if p.is_file() and '__pycache__' not in str(p) and not p.name.startswith('._')}
 (dst/'protocol.json').write_text(json.dumps({'version':VERSION.name,'seeds':[s['seed'] for s in jobs],'scenes':jobs,'scene_manifest_sha256':hashlib.sha256((VERSION/'scenes.json').read_bytes()).hexdigest(),'sources':hashes,'scope':'controlled scene coverage, not official LIBERO evaluation'},indent=2))
 def job(scene):
  seed=scene['seed'];out=dst/('can_s'+str(seed));cmd=container_command('collect.py',['--seed',str(seed),'--out','/app/'+str(out.relative_to(REPO))]);start=time.time()
  with out.with_suffix('.log').open('x') as f:
   try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800).returncode
   except subprocess.TimeoutExpired:code=-999
  if out.exists():subprocess.run(['docker','run','--rm','--network','none','-v',str(dst)+':/data','--entrypoint','chown','libero:latest','-R',str(os.getuid())+':'+str(os.getgid()),'/data/'+out.name],check=True)
  st={'seed':seed,'scene':scene,'exit_code':code,'normal_exit':code==0,'wall_seconds':time.time()-start,'command':cmd}
  out.with_suffix('.status.json').write_text(json.dumps(st,indent=2));print(scene['id'],'exit',code,'seconds',round(st['wall_seconds'],1),flush=True);return st
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:status=list(pool.map(job,jobs))
 (dst/'process_status.json').write_text(json.dumps(status,indent=2));return int(not all(s['normal_exit'] for s in status))
if __name__=='__main__':raise SystemExit(main())
