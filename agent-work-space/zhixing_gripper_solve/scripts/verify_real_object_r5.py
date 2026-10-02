#!/usr/bin/env python3
"""Read-only evidence audit; recompute numeric gates from recorded states/physics."""
import argparse,collections,hashlib,json,pathlib
import numpy as np

def audit(p):
 r=json.loads((p/'result.json').read_text());rows=[json.loads(x) for x in (p/'steps.jsonl').read_text().splitlines()]
 extras=[json.loads(x) for x in (p/'recipient_contacts.jsonl').read_text().splitlines()]
 if not rows:
  errors=[]
  status=json.loads(p.with_name(p.name+'.status.json').read_text())
  if not (r['failure'] and r['failure']['stage']=='planning' and not r['success'] and not extras and not (p/'physics.npz').exists() and status['normal_exit']):errors.append('invalid empty preflight failure')
  for name,sha in r['sources'].items():
   target=pathlib.Path('/mnt/data/vla-policy/openpi')/pathlib.Path(name).relative_to('/app')
   if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=sha:errors.append('source changed: '+name)
  return {'case':r['case'],'seed':r['seed'],'success':False,'failure':r['failure'],'audit_pass':not errors,'errors':errors,'gates':r['gates'],'initial_state_sha256':r['initial_state_sha256'],'initial_object_bbox':r['initial_geometry']['object'],'mass_kg':r['object_mass_kg'],'physics_samples':0,'bilateral_hold':0.,'hold_force_mean_N':None,'hold_force_balance_mean_N':None,'hold_other_force_peak_N':None,'penetration_peak_mm':None,'max_step_disp_mm':0.,'recipient_shift_mm':0.,'min_horizontal_clearance_mm':None,'unsafe_contact_samples':0,'robot_nontarget_contacts':0,'hold_relative_slip_mm':None,'post_drift_mm':None,'sources':r['sources'],'audit_note':'Planning rejection before first commanded action; absence of physics trace is expected. Recorded initial predicate cannot be independently checked from empty steps. No success asserted.'}
 v=np.load(p/'physics.npz')['values'];errors=[]
 def check(ok,msg):
  if not ok:errors.append(msg)
 check(np.isfinite(v).all(),'nonfinite physics')
 check(len(v)==sum(x['physics_samples'] for x in rows)==len(extras),'physics count mismatch')
 by={ph:[x for x in rows if x['phase']==ph] for ph in dict.fromkeys(x['phase'] for x in rows)}
 for x in rows:
  raw=v[v[:,0]==x['step']]
  check(len(raw)==x['physics_samples'],'step sample count')
  check(np.allclose(raw[:,2:4].mean(0),x['normal_force_mean_N'],atol=1e-9),'force summary mismatch')
  check(np.allclose(raw[:,4:6].max(0),x['penetration_peak_mm'],atol=1e-9),'penetration summary mismatch')
 h=by.get('hold',[]);post=by.get('post_release_window',[]);hids=[x['step'] for x in h];hv=v[np.isin(v[:,0],hids)]
 height=lambda x:float(np.mean(np.array(x['geometry']['object'])[:,2])-np.mean(np.array(r['initial_geometry']['object'])[:,2]))
 bil=float(np.mean((hv[:,6]>0)&(hv[:,7]>0))) if len(hv) else 0
 maxdisp=max(np.linalg.norm(np.array(x['after']['grip_site'])-x['before']['grip_site']) for x in rows)
 gates={'initial_goal_false':not rows[0]['before']['native_success'],'all_stages_complete':r['failure'] is None and all(ph in by for ph in ['approach','descend','close','lift','hold','clearance_lift','transport','place','release','post_release_settle','post_release_window']),
 'lift_30mm':bool(h and height(h[0])>=.030),'hold_40steps':len(h)==40,'bilateral_hold':bool(h and bil>=.95),'unsupported_hold':bool(h and hv[:,8].max()<1e-5),
 'penetration_2mm':float(v[:,4:6].max())<=2,'max_step_disp_20mm':maxdisp<.02,
 'native_success_window':len(post)==20 and all(x['after']['native_success'] for x in post),
 'post_drift_5mm':len(post)==20 and np.linalg.norm(np.array(post[-1]['after']['object_pos'])-post[0]['before']['object_pos'])<.005,
 'target_support_contact':len(post)==20 and all(any(e['step']==x['step'] and e['payload_basket_force_N']>1e-5 for e in extras) for x in post),
 'no_physics_warnings':not any(r['physics_warnings'])}
 if 'closure_bilateral' in r['gates']:
  ids=[x['step'] for x in by.get('close',[])[-10:]];cv=v[np.isin(v[:,0],ids)];gates['closure_bilateral']=len(ids)==10 and float(np.mean((cv[:,6]>0)&(cv[:,7]>0)))>=.95
 if 'hold_height' in r['gates']:gates['hold_height']=bool(h and all(height(x)>=.025 for x in h))
 check(gates==r['gates'],'reported gate mismatch')
 check(all(gates.values())==r['success'],'reported success mismatch')
 initial=np.array(r['initial_geometry']['basket_pos']);shift=max(np.linalg.norm(np.array(x['geometry']['basket_pos'])-initial) for x in rows)
 transport=by.get('transport',[]);clearance=min((min(x['geometry']['object'][0][2],x['geometry']['gripper_outer'][0][2])-r['initial_geometry']['basket'][1][2] for x in transport),default=None)
 unsafe=[e for e in extras if e['gripper_external_force_N']>.1 or e['robot_basket_force_N']>.1 or (e['phase'] in ('clearance_lift','transport','place') and e['payload_basket_force_N']>.1)]
 robot_scene=[c for e in extras for c in e.get('all_robot_external_contacts',[]) if not c['target_object'] and c['normal_N']>.1]
 if r['success']:
  check(not robot_scene,'robot touched non-target scene')
  check(not unsafe,'successful run has unsafe contact');check(shift<=.005,'recipient displacement');check(clearance is not None and clearance>=.010,'horizontal clearance')
  for phase,a in r['plan']['stage_arrivals'].items():
   last=by[phase][-1]['geometry'];err=np.linalg.norm(np.array(last['tcp'])-a['target'])
   angle=np.arccos(np.clip((np.trace(np.array(a['target_rotation']).T@np.array(last['R']))-1)/2,-1,1))
   check(err<a['position_tolerance_m'] and angle<.03,'arrival '+phase)
 src={}
 for name,sha in r['sources'].items():
  target=pathlib.Path('/mnt/data/vla-policy/openpi')/pathlib.Path(name).relative_to('/app')
  check(target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest()==sha,'source changed: '+name);src[name]=sha
 status=json.loads(p.with_name(p.name+'.status.json').read_text());check(status['normal_exit'] and status['exit_code']==0,'process exit')
 return {'case':r['case'],'seed':r['seed'],'success':r['success'],'failure':r['failure'],'audit_pass':not errors,'errors':errors,'gates':gates,'initial_state_sha256':r['initial_state_sha256'],'initial_object_bbox':r['initial_geometry']['object'],'mass_kg':r['object_mass_kg'],'physics_samples':len(v),'bilateral_hold':bil,'hold_force_mean_N':hv[:,2:4].mean(0).tolist() if len(hv) else None,'hold_force_balance_mean_N':(hv[:,-3:].mean(0)+[0,0,-9.81*r['object_mass_kg']]).tolist() if len(hv) else None,'hold_other_force_peak_N':float(hv[:,8].max()) if len(hv) else None,'penetration_peak_mm':float(v[:,4:6].max()),'max_step_disp_mm':float(maxdisp*1000),'recipient_shift_mm':float(shift*1000),'min_horizontal_clearance_mm':None if clearance is None else float(clearance*1000),'unsafe_contact_samples':len(unsafe),'robot_nontarget_contacts':len(robot_scene),'hold_relative_slip_mm':float(max((np.linalg.norm(np.array(x['after']['object_in_grip'])-h[0]['before']['object_in_grip']) for x in h),default=0)*1000),'post_drift_mm':float(np.linalg.norm(np.array(post[-1]['after']['object_pos'])-post[0]['before']['object_pos'])*1000) if post else None,'sources':src}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();batch=pathlib.Path(a.batch);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
 records=[audit(p.parent) for p in sorted(batch.glob('*/result.json'))];protocol=json.loads((batch/'protocol.json').read_text());complete=len(records)==len(protocol['jobs'])
 summary={c:{'success':sum(x['success'] for x in records if x['case']==c),'total':sum(x['case']==c for x in records),'unique_initial_states':len({x['initial_state_sha256'] for x in records if x['case']==c}),'unique_target_bboxes':len({json.dumps(x['initial_object_bbox']) for x in records if x['case']==c})} for c in sorted({r['case'] for r in records})}
 result={'version':'verify_real_object_r5','batch':str(batch),'pass':complete and all(x['audit_pass'] for x in records),'complete':complete,'summary':summary,'records':records,'limitation':'Native task predicates use recorded engine output; numeric gates recomputed independently from per-physics-step data. Audit PASS is evidence consistency, not task acceptance.'}
 (out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,default=lambda x:x.item())+'\n');print(json.dumps({'pass':result['pass'],'summary':summary},ensure_ascii=False));return 0 if result['pass'] else 1
if __name__=='__main__':raise SystemExit(main())
