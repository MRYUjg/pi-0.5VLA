"""Host orchestration for three validation rollouts; all attempts retained."""
import argparse,json,pathlib,subprocess,sys,time,urllib.request
V=pathlib.Path(__file__).resolve().parent;ROOT=V.parents[1];REPO=ROOT.parents[1]
rel=lambda p:'/app/'+str(p.relative_to(REPO))
def main():
 p=argparse.ArgumentParser();p.add_argument('--training-run',default='smoke_train_r2');p.add_argument('--name',required=True);a=p.parse_args()
 assert all(pathlib.Path(x).name==x and x not in ('.','..') for x in (a.name,a.training_run))
 train=ROOT/'runs'/a.training_run;out=ROOT/'rollouts'/a.name;out.mkdir(parents=True,exist_ok=False)
 assert json.loads((train/'reload_result.json').read_text())['pass']
 sys.path.insert(0,str(ROOT/'versions/lora_20261004_r1/scripts'));from launch import command
 cmd=command('serve.py',train);cmd[cmd.index('none')]='host';cmd[-3]=rel(V/'serve.py')
 commands={'server':cmd,'episodes':[]};results=[];started=time.time()
 with (out/'server.log').open('x') as log:
  server=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  try:
   for _ in range(300):
    if server.poll() is not None:raise RuntimeError('policy server exited '+str(server.returncode))
    try:
     if json.loads(urllib.request.urlopen('http://127.0.0.1:8871',timeout=1).read())['ready']:break
    except (OSError,ValueError):time.sleep(2)
   else:raise TimeoutError('policy startup')
   for seed in (55200,55201,55202):
    episode=out/f'seed_{seed}'
    c=['docker','run','--rm','--network','host','--cpus','2','--memory','5g','-e','MUJOCO_GL=osmesa','-e','PYTHONDONTWRITEBYTECODE=1','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','LP_NUM_THREADS=1','-e','PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero','-v',str(REPO)+':/app:ro','-v',str(ROOT)+':'+rel(ROOT),'-w','/app','--entrypoint','/app/agent-work-space/1.3-zhixing_gripper_solve/runtime/py310_mj326/bin/python','libero:latest',rel(V/'rollout.py'),'--seed',str(seed),'--out',rel(episode)]
    commands['episodes'].append(c);t=time.time()
    with (out/f'seed_{seed}.log').open('x') as f:code=subprocess.run(c,stdout=f,stderr=subprocess.STDOUT).returncode
    r={'seed':seed,'exit_code':code,'seconds':time.time()-t}
    if (episode/'result.json').exists():r['result']=json.loads((episode/'result.json').read_text())
    results.append(r);print('EPISODE',seed,'exit',code,flush=True)
    if code:break
  finally:
   try:urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8871/shutdown',data=b''),timeout=5).read()
   except OSError:pass
   try:server.wait(timeout=30)
   except subprocess.TimeoutExpired:server.terminate();server.wait(timeout=15)
   (out/'execution.json').write_text(json.dumps({'server_exit':server.returncode,'seconds':time.time()-started,'episodes':results,'commands':commands},indent=2))
 return int(len(results)!=3 or any(r['exit_code'] for r in results) or server.returncode!=0)
if __name__=='__main__':raise SystemExit(main())
