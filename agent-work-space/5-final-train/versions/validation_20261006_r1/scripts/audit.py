"""Formal source admission: unchanged r6 physical audit plus structural/init/replay gates."""
import argparse
import hashlib
import json
import pathlib
import sys
import h5py
import numpy as np

VERSION=pathlib.Path(__file__).resolve().parents[1]
ROOT=VERSION.parents[1]
COLLECTION=ROOT/'versions/collection_20261005_r4'
sys.path.insert(0,str(COLLECTION/'vendor'))
import verify_real_object_r6 as physical

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(p.read_text())

def audit(ep):
    errors=[]
    def check(ok,msg):
        if not ok:errors.append(msg)
    status=read(ep.with_suffix('.status.json'))
    capture=read(ep/'capture.json')
    result=read(ep/'teacher/result.json')
    init=read(ep/'initialization.json')
    numeric=physical.audit(ep/'teacher')
    replay=read(ep/'replay.json') if (ep/'replay.json').exists() else None
    replay_status=read(ep/'replay.status.json') if (ep/'replay.status.json').exists() else None
    scene=capture['scene']
    registered=next(s for s in read(COLLECTION/'scenes.json')['jobs'] if s['seed']==capture['seed'])
    check(scene==registered,'unregistered scene or split')
    check(status['exit_code']==0 and status['normal_exit'],'capture process failed')
    check(capture['teacher_finished'] and not capture['probe'],'incomplete/probe capture')
    check(numeric['audit_pass'],'r6 physical evidence integrity')
    rows=[json.loads(s) for s in (ep/'teacher/steps.jsonl').read_text().splitlines()]
    with h5py.File(ep/'trajectory.h5','r') as f:
        n=len(f['transitions/actions']) if 'transitions/actions' in f else 0
        check(n==len(rows)==capture['transitions'],'actual action count')
        check(bool(f.attrs['teacher_finished']),'HDF5 not complete')
        check(f.attrs['task_name']==scene['name']==result['task_name'],'task identity')
        check(f.attrs['language']==scene['benchmark_prompt'],'canonical prompt')
        check(f.attrs['scene_manifest_sha256']==sha(COLLECTION/'scenes.json'),'scene configuration drift')
        check(json.loads(f.attrs['coverage_scene'])['scene']==scene,'HDF5 scene identity')
        check(len(f['preparation/actions'])==120,'preparation count')
        check(all(len(f['preparation_images/'+key])==120 for key in ('image','wrist_image')),'preparation RGB count')
        if n:
            obs=f['observations'];actions=f['transitions/actions'][...]
            check(actions.shape==(n,7) and np.isfinite(actions).all() and abs(actions).max()<=1,'action shape/finiteness/range')
            check(all(len(d)==n+1 for d in obs.values()),'T+1 observation boundary')
            check(obs['state'].shape==(n+1,8),'state contract')
            for key in ('state','qpos','qvel','qacc_warmstart','ctrl','gripper_qpos'):
                check(np.isfinite(obs[key][...]).all(),'finite '+key)
            check(np.allclose(np.diff(obs['time']),.05,rtol=0,atol=1e-9),'20Hz times')
            for field in ('qpos','time'):
                check(np.array_equal(obs[field][:-1],f['transitions/'+field+'_before'][...]),'pre-action '+field)
                check(np.array_equal(obs[field][1:],f['transitions/'+field+'_after'][...]),'post-action '+field)
            check(np.array_equal(actions,np.array([r['action'] for r in rows])),'executed action mismatch')
            check([s.decode().strip() for s in f['transitions/phase']]==[r['phase'] for r in rows],'phase mismatch')
            names=json.loads(f.attrs['gripper_joint_names'])
            ids=[next(i for i,name in enumerate(names) if name.endswith('_'+suffix)) for suffix in ('Left_1_Joint','Right_1_Joint')]
            check(np.allclose(obs['state'][:,-2:],obs['gripper_qpos'][...][:,ids],rtol=0,atol=1e-7),'named gripper state')
            for key in ('image','wrist_image'):
                check(obs[key].shape==(n+1,256,256,3) and obs[key].dtype==np.uint8,'RGB contract '+key)
                check(all(np.std(obs[key][i])>2 for i in {0,n//2,n}),'blank image '+key)
            check(not obs['native_success'][0],'initial goal already met')
            check(replay is not None and replay['pass'],'independent action replay')
            check(replay_status is not None and replay_status['exit_code']==0,'replay process exit')
        else:
            check(not result['success'] and result['failure']['stage']=='planning','empty trajectory not planning rejection')
        reset_hash=hashlib.sha256(f['reset/qpos'][...].tobytes()).hexdigest()
        start_hash=hashlib.sha256(np.round(f['preparation/qpos_after'][-1],8).tobytes()).hexdigest()
        positions=json.loads(f.attrs.get('settled_object_states','{}'))
    admission=[]
    if errors:admission.append('source integrity checks failed')
    if not init['pass']:admission.append('initialization audit failed')
    if not result['success']:admission.append('teacher task failure')
    if not n:admission.append('no teacher action trajectory')
    if scene['split'] not in ('train','validation'):admission.append('development/reference is not training data')
    relevant=['trajectory.h5','capture.json','initialization.json','teacher/result.json','teacher/steps.jsonl','teacher/recipient_contacts.jsonl','teacher.status.json']
    if replay is not None:relevant+=['replay.json','replay.status.json']
    return {'accepted':not admission,'development_pass':not errors and init['pass'] and bool(result['success']) and n>0,
        'integrity_pass':not errors,'errors':errors,'admission_rejections':admission,'seed':capture['seed'],
        'task_id':scene['task_id'],'split':scene['split'],'frames':n,'scene':scene,
        'teacher_success':bool(result['success']),'teacher_failure':result.get('failure'),
        'initialization_pass':init['pass'],'initialization_errors':init['errors'],'physical_audit':numeric,
        'reset_qpos_sha256':reset_hash,'settled_qpos_rounded8_sha256':start_hash,'settled_objects':positions,
        'files':{name:sha(ep/name) for name in relevant},'scope':'demonstration admission, not policy success'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--episode',type=pathlib.Path,required=True);p.add_argument('--collection-version',choices=['collection_20261005_r4','collection_20261006_r1'],default='collection_20261005_r4');a=p.parse_args();COLLECTION=ROOT/'versions'/a.collection_version
    result=audit(a.episode)
    with (a.episode/'audit.json').open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,default=lambda x:x.item())
    print(json.dumps({k:result[k] for k in ('accepted','development_pass','integrity_pass','seed','task_id','teacher_success','initialization_pass','errors','admission_rejections')},ensure_ascii=False),flush=True)
    raise SystemExit(0 if result['integrity_pass'] else 1)
