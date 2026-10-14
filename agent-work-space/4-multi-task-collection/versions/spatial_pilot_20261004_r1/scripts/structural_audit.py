"""Read-only structural checks on all attempted trajectories; no data release."""
import pathlib,json,hashlib
import h5py,numpy as np
ROOT=pathlib.Path('/app/agent-work-space/4-multi-task-collection')
batch=ROOT/'runs/spatial_pilot_20261004_r1'
records=[]
for tid in range(10):
    ep=batch/f'task{tid:02d}_s{57100+tid}';errors=[]
    check=lambda ok,msg:errors.append(msg) if not ok else None
    status=json.loads(ep.with_suffix('.status.json').read_text());result=json.loads((ep/'teacher/result.json').read_text());cap=json.loads((ep/'capture.json').read_text())
    physics=json.loads((ep/'physical_audit.json').read_text());replay_path=ROOT/'runs/spatial_replay_20261004_r1'/f'task{tid:02d}.json';replay=json.loads(replay_path.read_text()) if replay_path.exists() else None
    rows=[json.loads(s) for s in (ep/'teacher/steps.jsonl').read_text().splitlines()]
    with h5py.File(ep/'trajectory.h5','r') as f:
        check(status['exit_code']==0,'capture process exit');check(f.attrs['teacher_finished'],'teacher completion')
        n=len(f['transitions/actions']) if 'transitions/actions' in f else 0
        check(n==len(rows)==cap['transitions'],'action count')
        prep=len(f['preparation/actions']);check(len(f['preparation_images/image'])==len(f['preparation_images/wrist_image'])==prep,'preparation video frame count')
        spec=json.loads(f.attrs['coverage_scene'])['scene'];check(spec['task_id']==tid and spec['name']==result['task_name']==f.attrs['task_name'],'task mapping')
        check(spec['benchmark_prompt']==f.attrs['language'],'canonical prompt')
        if n:
            obs=f['observations'];a=f['transitions/actions'][...]
            check(a.shape==(n,7) and np.isfinite(a).all() and abs(a).max()<=1,'action validity')
            check(all(len(d)==n+1 for d in obs.values()),'T+1 observations')
            check(obs['state'].shape==(n+1,8),'state shape')
            for k in ('state','qpos','qvel','ctrl','gripper_qpos'):check(np.isfinite(obs[k][...]).all(),'finite '+k)
            check(np.allclose(np.diff(obs['time']),.05,rtol=0,atol=1e-9),'20Hz')
            check(np.array_equal(obs['qpos'][:-1],f['transitions/qpos_before'][...]),'pre action state')
            check(np.array_equal(obs['qpos'][1:],f['transitions/qpos_after'][...]),'post action state')
            check(np.array_equal(obs['time'][:-1],f['transitions/time_before'][...]),'pre action time')
            check(np.array_equal(obs['time'][1:],f['transitions/time_after'][...]),'post action time')
            check(np.array_equal(a,np.array([r['action'] for r in rows])),'actual action equality')
            check([s.decode().strip() for s in f['transitions/phase']]==[r['phase'] for r in rows],'phase equality')
            names=json.loads(f.attrs['gripper_joint_names']);idx=[next(i for i,name in enumerate(names) if name.endswith('_'+s)) for s in ('Left_1_Joint','Right_1_Joint')]
            check(np.allclose(obs['state'][:,-2:],obs['gripper_qpos'][...][:,idx],rtol=0,atol=1e-7),'named gripper joints')
            for k in ('image','wrist_image'):
                check(obs[k].shape==(n+1,256,256,3) and obs[k].dtype==np.uint8,'RGB '+k)
                check(all(np.std(obs[k][i])>2 for i in {0,n//2,n}),'blank RGB '+k)
            check(replay is not None and replay['pass'],'replay')
            check(not obs['native_success'][0],'initial native success')
        else:check(result['failure'] is not None and result['failure']['stage']=='planning' and not result['success'],'empty planning failure')
        initial={'reset_object_states':json.loads(f.attrs['reset_object_states']),'reset_contacts_raw':json.loads(f.attrs['reset_contacts']),'preparation_max_contact_penetration_mm':(float(f.attrs['preparation_non_support_penetration_mm']) if 'preparation_non_support_penetration_mm' in f.attrs else None),'settled_object_states':json.loads(f.attrs.get('settled_object_states','{}')),'note':'Raw contact classifier counts object-object support; diagnostic only, not an initialization acceptance gate.'}
    records.append({'task_id':tid,'seed':57100+tid,'teacher_success':result['success'],'failure':result['failure'],'physical_audit_pass':physics['audit_pass'],'physical_errors':physics['errors'],'structural_pass':not errors,'structural_errors':errors,'replay_pass':replay['pass'] if replay else None,'frames':n,'preparation_frames':prep,'initialization':initial,'task_name':spec['name'],'bddl_sha256':spec['bddl_sha256'],'dataset_released':False,'hdf5_sha256':hashlib.sha256((ep/'trajectory.h5').read_bytes()).hexdigest()})
report={'attempts':len(records),'teacher_successes':sum(r['teacher_success'] for r in records),'physical_evidence_passes':sum(r['physical_audit_pass'] for r in records),'structural_passes':sum(r['structural_pass'] for r in records),'replay_passes':sum(r['replay_pass'] is True for r in records),'dataset_released':False,'scope':'one native development attempt per task; not policy evaluation or formal LIBERO benchmark','records':records}
with (batch/'structural_audit.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
print(json.dumps({k:v for k,v in report.items() if k!='records'}))
