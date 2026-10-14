import pathlib,sys,json,subprocess,concurrent.futures,time
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parents[1];REPO=R.parents[1]
sys.path.insert(0,str(R/'versions/collection_20261005_r2/scripts'))
from run import command
B=R/'runs/reference_baseline_20261005_r1';B.mkdir(parents=True,exist_ok=False)
def run(t):
 s=57100+t;p=B/f'task{t:02d}_s{s}';cmd=command(R/'versions/diagnostics_20261005_r1/scripts/baseline_preflight.py',['--seed',str(s),'--out','/app/'+str(p.relative_to(REPO)),'--preflight']);start=time.time()
 with p.with_suffix('.log').open('x') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 p.with_suffix('.status.json').write_text(json.dumps({'exit_code':r.returncode,'seconds':time.time()-start,'command':cmd},indent=2));print(t,r.returncode,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:list(pool.map(run,[0,2,3,5,6,8]))
