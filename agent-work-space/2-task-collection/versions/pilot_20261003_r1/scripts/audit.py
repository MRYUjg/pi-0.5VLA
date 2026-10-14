"""Dataset release audit; preserve r6 physical criteria and independently check temporal alignment."""
import argparse,hashlib,json,pathlib,sys
import h5py,numpy as np
from PIL import Image,ImageDraw
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/'vendor'))
import verify_real_object_r6 as upstream

def audit(ep):
    errors=[]
    def check(condition,msg):
        if not condition:errors.append(msg)
    status=json.loads(ep.with_suffix('.status.json').read_text())
    replay_status=json.loads((ep/'replay.status.json').read_text())
    replay=json.loads((ep/'replay.json').read_text())
    check(status['normal_exit'] and status['exit_code']==0,'capture exit')
    check(replay_status['exit_code']==0 and replay['pass'],'replay failed')
    r=upstream.audit(ep/'teacher');check(r['audit_pass'],'physical evidence: '+str(r['errors']))
    check(r['success'],'teacher failed')
    rows=[json.loads(x) for x in (ep/'teacher/steps.jsonl').read_text().splitlines()]
    with h5py.File(ep/'trajectory.h5','r') as f:
        a=f['transitions/actions'][...];o=f['observations'];n=len(a)
        check(n==len(rows) and n>0,'transition count')
        check(a.shape==(n,7) and np.isfinite(a).all() and np.max(abs(a))<=1,'action validity')
        check(all(len(ds)==n+1 for ds in o.values()),'T+1 observations')
        check(o['state'].shape==(n+1,8),'state shape')
        check(np.allclose(np.diff(o['time'][...]),.05,atol=1e-9,rtol=0),'20 Hz time')
        check(np.array_equal(o['qpos'][:-1],f['transitions/qpos_before'][...]),'o_t before action')
        check(np.array_equal(o['qpos'][1:],f['transitions/qpos_after'][...]),'o_t+1 after action')
        check(np.array_equal(o['time'][:-1],f['transitions/time_before'][...]),'time before action')
        check(np.array_equal(o['time'][1:],f['transitions/time_after'][...]),'time after action')
        for name in ('state','qpos','qvel','ctrl','gripper_qpos'):
            check(np.isfinite(o[name][...]).all(),'nonfinite '+name)
        phases=[s.decode().strip() for s in f['transitions/phase'][...]]
        check(phases==[r['phase'] for r in rows],'phase alignment')
        check(np.array_equal(a,np.array([r['action'] for r in rows])),'actual executed action equality')
        check(np.allclose(o['qpos'][:-1],np.array([r['before']['qpos'] for r in rows]),rtol=0,atol=0),'physics pre-state equality')
        names=json.loads(f.attrs['gripper_joint_names']);idx=[next(i for i,nm in enumerate(names) if nm.endswith('_'+s)) for s in ('Left_1_Joint','Right_1_Joint')]
        check(np.allclose(o['state'][:,-2:],o['gripper_qpos'][...][:,idx],rtol=0,atol=1e-7),'named gripper state')
        check(np.allclose(o['state'][:-1,:3],np.array([r['before']['grip_site'] for r in rows]),rtol=0,atol=1e-7),'TCP state')
        for name in ('image','wrist_image'):
            ds=o[name];check(ds.shape==(n+1,256,256,3) and ds.dtype==np.uint8,'RGB shape/dtype '+name)
            check(all(float(np.std(ds[i]))>2 for i in (0,n//2,n)),'blank RGB '+name)
            check(np.any(ds[0]!=ds[n//2]),'static RGB '+name)
        unique_indices=[]
        for phase in ('approach','close','hold','transport','release','post_release_window'):
            ids=[i for i,p in enumerate(phases) if p==phase]
            if ids:unique_indices.append((phase,ids[len(ids)//2]))
        sheet=Image.new('RGB',(256*len(unique_indices),540),'#222222');draw=ImageDraw.Draw(sheet)
        for col,(phase,i) in enumerate(unique_indices):
            sheet.paste(Image.fromarray(o['image'][i]),(col*256,25));sheet.paste(Image.fromarray(o['wrist_image'][i]),(col*256,282));draw.text((col*256+6,6),phase+' '+str(i),fill='white')
        sheet.save(ep/'preview.jpg')
        stats={'frames':n,'sim_seconds':n/20,'language':str(f.attrs['language']),'initial_scene_sha256':hashlib.sha256(o['qpos'][0].tobytes()).hexdigest(),
            'initial_object_bbox':r['initial_object_bbox'],'max_action_abs':float(abs(a).max()),'phase_counts':{p:phases.count(p) for p in dict.fromkeys(phases)}}
    files={str(p.relative_to(ep)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ep.rglob('*') if p.is_file() and p.name not in ('audit.json',)}
    return {'accepted':not errors,'errors':errors,'teacher_numeric_audit':r,'replay':replay,'stats':stats,'files':files}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);a=ap.parse_args();batch=pathlib.Path(a.batch);records=[]
    protocol=json.loads((batch/'protocol.json').read_text())
    for seed in protocol['seeds']:
        ep=batch/('can_s'+str(seed))
        try:r=audit(ep)
        except Exception as e:r={'accepted':False,'errors':[type(e).__name__+': '+str(e)]}
        with (ep/'audit.json').open('x') as f:json.dump(r,f,indent=2,ensure_ascii=False,default=lambda x:x.item())
        records.append({'seed':seed,'episode':str(ep),'accepted':r['accepted'],'errors':r['errors'],'stats':r.get('stats')})
    accepted=[r for r in records if r['accepted']]
    summary={'version':'pilot_20261003_r1','attempts':len(protocol['seeds']),'accepted':len(accepted),'records':records,'scope':'pipeline pilot only; not formal policy acceptance','unique_scene_hashes':len({r['stats']['initial_scene_sha256'] for r in accepted}),'unique_target_bboxes':len({json.dumps(r['stats']['initial_object_bbox']) for r in accepted})}
    with (batch/'dataset_manifest.json').open('x') as f:json.dump(summary,f,indent=2,ensure_ascii=False)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    return int(len(accepted)!=len(records))
if __name__=='__main__':raise SystemExit(main())
