#!/usr/bin/env python3
"""Independent post-run validation of saved geometry, actions, process status and gates."""
import argparse,json,pathlib,hashlib
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
b=pathlib.Path(a.batch);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
status=json.loads((b/'process_status.json').read_text());summary=[];issues=[]
for st in status:
    job=st['job'];name='%s_%s_s%d_f%g'%(job['gripper'],job['mode'],job['seed'],job['force']);p=b/name
    assert st['exit_code']==0 and st['normal_exit'],name+' process exit'
    r=json.loads((p/'result.json').read_text());plan=r.get('safe_path',{});rs=[json.loads(x) for x in (p/'steps.jsonl').read_text().splitlines()] if (p/'steps.jsonl').exists() else []
    cs=[json.loads(x) for x in (p/'basket_contacts.jsonl').read_text().splitlines()] if (p/'basket_contacts.jsonl').exists() else []
    assert not any(r['physics_warning_counts']),name+' physics warnings'
    assert len(cs)==25*len(rs),name+' substep contact records'
    for row in rs:
        assert row['physics_samples']==25
        assert len(row['action'])==7 and np.max(np.abs(row['action']))<=1
        for key in ('qpos','qvel','grip_site','object_pos'):assert np.isfinite(row['after'][key]).all()
    if rs:
        with np.load(p/'physics.npz') as f:assert len(f['values'])==25*len(rs) and np.isfinite(f['values']).all()
    success=bool(r.get('safe_teacher_success'));phases=list(dict.fromkeys(x['phase'] for x in rs));geo=[x for x in rs if 'safe_geometry' in x]
    across=[x for x in geo if x['phase']=='transport'];place=[x for x in geo if x['phase']=='place']
    rim=plan.get('measured_loaded_geometry',{}).get('basket',[[0,0,0],[0,0,0]])[1][2]
    clearance=min([min(x['safe_geometry']['object'][0][2],x['safe_geometry']['gripper_outer'][0][2])-rim for x in across],default=None)
    initial=json.loads((p/'safe_path.json').read_text()).get('initial_geometry')
    shift=max([float(np.linalg.norm(np.asarray(x['safe_geometry']['basket_pos'])-initial['basket_pos'])) for x in geo],default=0)
    peak_robot=max([x['robot_basket_force_N'] for x in cs if x['phase'] in ('clearance_lift','transport','place')],default=0)
    peak_payload=max([x['payload_basket_force_N'] for x in cs if x['phase'] in ('clearance_lift','transport','place')],default=0)
    if success:
        assert r['result']['success_n4_style'] and all(r['result']['gates'].values()) and not r['result']['notes']
        required=['hold','clearance_lift','transport','place','release','post_release_window']
        assert all(ph in phases for ph in required) and [phases.index(ph) for ph in required]==sorted(phases.index(ph) for ph in required)
        for ph in ('approach','descend','lift','clearance_lift','transport','place'):
            v=plan['stage_arrivals'][ph];assert v['ok'] and v['error'][0]<v['position_tolerance_m'] and v['error'][1]<.03
            assert abs(np.linalg.norm(np.asarray(v['actual'])-v['target'])-v['error'][0])<1e-10
        assert clearance>=.01 and shift<=.005 and peak_robot<=.1 and peak_payload<=.1
        assert all(c['pass'] for c in plan['ik_candidates'][-1]['checks'])
        assert all(c['pass'] for c in plan['corrected_horizontal_ik']+plan['corrected_place_ik'])
        post=[x for x in rs if x['phase']=='post_release_window'];assert len(post)==20 and all(x['after']['native_success'] for x in post)
    summary.append({'seed':job['seed'],'success':success,'failure':r['result'].get('notes') or r['result'].get('reason'),'phases':phases,'tilt_deg':plan.get('tilt_deg'),'safe_tcp_height_m':plan.get('vertical_target',[None,None,None])[2],'horizontal_clearance_min_mm':None if clearance is None else clearance*1000,'basket_displacement_max_mm':shift*1000,'robot_basket_force_peak_before_release_N':peak_robot,'payload_basket_force_peak_before_release_N':peak_payload,'stage_arrivals':plan.get('stage_arrivals',{}),'steps':len(rs),'physics_samples':len(cs)})
res={'verdict':'PASS','scope':'Saved evidence integrity and declared safe-teacher criteria; not LIBERO-wide or hardware acceptance','attempts':len(summary),'successes':sum(x['success'] for x in summary),'runs':summary}
(out/'verification.json').write_text(json.dumps(res,ensure_ascii=False,indent=2)+'\n');print(json.dumps(res,ensure_ascii=False,indent=2))
