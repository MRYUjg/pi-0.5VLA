"""Collect every registered attempt once; preserve failures; resume only complete jobs."""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
sys.dont_write_bytecode=True
VERSION=pathlib.Path(__file__).resolve().parents[1]
ROOT=VERSION.parents[1]
REPO=ROOT.parents[1]
COLLECTION=ROOT/'versions/collection_20261005_r4'
sys.path.insert(0,str(COLLECTION/'scripts'))
from run import command

def read(p):return json.loads(p.read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)

def verify_sources():
    for name in (COLLECTION.name,VERSION.name):
        for rel,want in read(ROOT/'releases'/name/'manifest.json')['files'].items():
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==want,rel

def execute(script,args,log,timeout):
    script=pathlib.Path(script);script=script if script.is_absolute() else COLLECTION/'scripts'/script
    cmd=command(script,args);start=time.time()
    container='eco65-final-'+hashlib.sha256((str(log)+str(start)).encode()).hexdigest()[:16]
    cmd[2:2]=['--name',container]
    with log.open('x') as f:
        try:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','rm','-f',container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            code=-999
    for flag in ('--episode','--out'):
        if flag in args:
            target=REPO/pathlib.Path(args[args.index(flag)+1]).relative_to('/app')
            target=target if target.is_dir() else target.parent
            assert ROOT in target.parents or target==ROOT
            if target.exists():
                subprocess.run(['docker','run','--rm','--network','none','-v',str(target)+':/output','--entrypoint','chown','libero:latest','-R',f'{os.getuid()}:{os.getgid()}','/output'],check=True,stdout=subprocess.DEVNULL)
            break
    return {'exit_code':code,'normal_exit':code==0,'wall_seconds':time.time()-start,'command':cmd}

def job(scene,batch):
    ep=batch/f'task{scene["task_id"]:02d}_s{scene["seed"]}'
    result_path=ep.with_suffix('.summary.json')
    if result_path.exists():return read(result_path)
    # Never rerun a partially executed seed or replace its failure with a fresh attempt.
    if ep.exists() or ep.with_suffix('.log').exists():
        raise RuntimeError('Incomplete attempt needs inspection: '+str(ep))
    if shutil.disk_usage(ROOT).free<40*1024**3:raise RuntimeError('Less than 40 GiB free; collection stopped')
    pre=batch/'preflight'/ep.name;pre.parent.mkdir(parents=True,exist_ok=True)
    if pre.exists() or pre.with_suffix('.log').exists():raise RuntimeError('Incomplete preflight needs inspection: '+str(pre))
    pre_status=execute('collect.py',['--seed',str(scene['seed']),'--out','/app/'+str(pre.relative_to(REPO)),'--preflight'],pre.with_suffix('.log'),900)
    write(pre.with_suffix('.status.json'),pre_status)
    init=read(pre/'initialization.json') if (pre/'initialization.json').exists() else {}
    if not pre_status['normal_exit'] or not init.get('pass',False):
        r={'scene':scene,'capture':None,'initialization_process':pre_status,'initialization':init,
           'accepted':False,'development_pass':False,'teacher_not_run':'initialization admission failed'}
        write(result_path,r)
        print(json.dumps({'task':scene['task_id'],'seed':scene['seed'],'split':scene['split'],'teacher_not_run':r['teacher_not_run'],'errors':init.get('errors')},ensure_ascii=False),flush=True)
        return r
    rel='/app/'+str(ep.relative_to(REPO))
    status=execute('collect.py',['--seed',str(scene['seed']),'--out',rel],ep.with_suffix('.log'),3600)
    write(ep.with_suffix('.status.json'),status)
    r={'scene':scene,'capture':status,'accepted':False,'development_pass':False}
    if status['normal_exit'] and (ep/'teacher/result.json').exists() and (ep/'capture.json').exists():
        write(ep/'teacher.status.json',{**status,'job':{'profile':'eco65_no_extra_pd','seed':scene['seed'],'case':'bowl'}})
        cap=read(ep/'capture.json')
        if cap['transitions']:
            replay=execute('replay.py',['--episode',rel,'--out',rel+'/replay.json'],ep/'replay.log',1800)
            write(ep/'replay.status.json',replay)
            r['replay']=replay
        audit=execute(VERSION/'scripts/audit.py',['--episode',rel,'--collection-version',COLLECTION.name],ep/'audit.log',900)
        write(ep/'audit.status.json',audit)
        r['audit_process']=audit
        if (ep/'audit.json').exists():
            report=read(ep/'audit.json');r.update(accepted=report['accepted'],development_pass=report['development_pass'],audit=report)
    write(result_path,r)
    print(json.dumps({'task':scene['task_id'],'seed':scene['seed'],'split':scene['split'],'accepted':r['accepted'],'development_pass':r['development_pass'],'exit':status['exit_code']},ensure_ascii=False),flush=True)
    return r

def main():
    global COLLECTION
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['regression','formal'],required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--supplement',action='store_true');a=p.parse_args()
    if a.supplement:
        assert a.stage=='formal'
        COLLECTION=ROOT/'versions/collection_20261006_r1'
    assert 1<=a.workers<=6
    verify_sources()
    reference=read(ROOT/'runs/reference_preflight_20261005_r3/summary.json')
    assert reference['pass'],'reference initialization/instrumentation gate failed'
    if a.stage=='formal':assert read(ROOT/'runs/regression_20261005_r1/summary.json')['pass'],'new-scene teacher regression gate failed'
    wanted={'regression'} if a.stage=='regression' else {'train','validation'}
    jobs=[s for s in read(COLLECTION/'scenes.json')['jobs'] if s['split'] in wanted]
    if a.stage=='formal':
        eligible=read(ROOT/'runs/regression_20261005_r1/summary.json')['eligible_tasks']
        jobs=[s for s in jobs if s['task_id'] in eligible]
    if a.supplement:jobs=[s for s in jobs if s['seed']>=650000]
    # Interleave tasks to expose per-task failures before finishing whole task blocks.
    jobs.sort(key=lambda s:(s['seed']%100,s['task_id'],s['split']))
    batch=ROOT/('runs/supplement_20261006_r1' if a.supplement else 'runs/'+a.stage+'_20261005_r1');batch.mkdir(parents=True,exist_ok=True)
    protocol={'stage':a.stage,'jobs':jobs,'workers':a.workers,'retries':0,'controller':'eco65_no_extra_pd'}
    if (batch/'protocol.json').exists():assert read(batch/'protocol.json')==protocol
    else:write(batch/'protocol.json',protocol)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(lambda s:job(s,batch),jobs))
    counts={}
    for tid in sorted({s['task_id'] for s in jobs}):
        per=[r for r in rows if r['scene']['task_id']==tid]
        counts[str(tid)]={split:{'attempted':sum(r['scene']['split']==split for r in per),'accepted':sum(r['scene']['split']==split and (r['development_pass'] if a.stage=='regression' else r['accepted']) for r in per)} for split in sorted(wanted)}
    if a.stage=='regression':
        eligible=[int(t) for t,v in counts.items() if v['regression']['accepted']>=2]
        passed=bool(eligible)
    else:
        eligible=[int(t) for t in counts]
        passed=all(v['train']['accepted']>=20 and v['validation']['accepted']>=4 for v in counts.values())
    hashes={};duplicates=[]
    for r in rows:
        if not (r['development_pass'] if a.stage=='regression' else r['accepted']):continue
        key=(r['scene']['task_id'],r['audit']['settled_qpos_rounded8_sha256'])
        if key in hashes:duplicates.append([hashes[key],r['scene']['seed']])
        hashes[key]=r['scene']['seed']
    if a.supplement:passed=any(r['accepted'] for r in rows)
    passed=passed and not duplicates
    write(batch/'summary.json',{'pass':passed,'stage':a.stage,'eligible_tasks':eligible,'excluded_tasks':sorted(set([0,2,3,5,6,8])-set(eligible)),'attempts':len(rows),'counts':counts,'duplicate_initializations':duplicates,
        'admission_threshold':'at least 2/3 regression successes per task' if a.stage=='regression' else 'at least 20 train and 4 validation accepted demonstrations per task',
        'records':[str((batch/f'task{r["scene"]["task_id"]:02d}_s{r["scene"]["seed"]}').relative_to(ROOT)) for r in rows]})
    subprocess.run(['docker','run','--rm','--network','none','-v',str(ROOT)+':/output','--entrypoint','chown','libero:latest','-R',f'{os.getuid()}:{os.getgid()}','/output'],check=True)
    print('STAGE COMPLETE',a.stage,passed,counts,flush=True)
    return int(not passed)

if __name__=='__main__':raise SystemExit(main())
