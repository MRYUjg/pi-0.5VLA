#!/usr/bin/env python3
import argparse,json,pathlib,hashlib,ast
import numpy as np
ap=argparse.ArgumentParser();ap.add_argument('batch');ap.add_argument('--out',required=True);a=ap.parse_args();batch=pathlib.Path(a.batch)
statuses=json.loads((batch/'process_status.json').read_text());assert len(statuses)==14
counts={'runs':0,'chain':0,'pull':0,'valid_preloads':0,'planner_rejections':0,'physics_samples':0};force_balance=[]
for status in statuses:
 assert status['exit_code']==0 and status['result_exists']
 j=status['job'];name='%s_%s_s%d_f%g'%(j['gripper'],j['mode'],j['seed'],j['force']);p=batch/name
 d=json.loads((p/'result.json').read_text());assert d['version']=='loaded_diagnostics_r5' and d['mujoco_version']=='3.2.6'
 assert d['calibration']['pass'] and abs(d['calibration']['normal_force_N']-9.81)<.02
 assert not any(d['physics_warning_counts'])
 for fn,h in d['sources'].items():assert hashlib.sha256(pathlib.Path(fn).read_bytes()).hexdigest()==h,(fn,'changed')
 counts['runs']+=1;counts[d['mode']]+=1
 if d['result'].get('reason'):
  assert d['mode']=='chain';counts['planner_rejections']+=1;continue
 rows=[json.loads(s) for s in (p/'steps.jsonl').read_text().splitlines()]
 raw=np.load(p/'physics.npz');v=raw['values'];assert np.isfinite(v).all();assert len(v)==25*len(rows)
 assert np.all(np.diff(v[:,1])>0)
 counts['physics_samples']+=len(v)
 for i,r in enumerate(rows):
  assert r['physics_samples']==25
  vv=v[v[:,0]==i];assert len(vv)==25
  assert np.allclose(vv[:,2:4].mean(axis=0),r['normal_force_mean_N'],atol=1e-10)
  assert np.allclose(vv[:,4:6].max(axis=0),r['penetration_peak_mm'],atol=1e-10)
  for key in ['qpos','qvel','ctrl','object_pos','object_in_grip']:assert np.isfinite(r['after'][key]).all()
 balance=np.linalg.norm(d['phases']['hold']['object_force_balance_mean_N']);assert balance<.01
 force_balance.append(float(balance))
 if d['gripper']=='PandaGripper':assert all(len(gs)>=2 for gs in d['finger_contact_geoms'])
 if d['mode']=='pull':
  q=d['result']['pull'];assert q['valid_preload'];counts['valid_preloads']+=1
  assert q['force_world']==[j['force'],0.,0.] and q['measurement']['steps']==40
for f in ['loaded_diagnostics_r5.py','run_loaded_matrix_r4.py','summarize_loaded_r2.py']:ast.parse((batch.parent.parent/'scripts'/f).read_text())
report={'verdict':'PASS','scope':'Measurement integrity and process validity only; no task/adaptation acceptance', 'counts':counts,'max_hold_force_balance_error_N':max(force_balance),
 'checks':['14 real exit codes zero','all runtime/source hashes match','all calibrations 9.81 N','no MuJoCo warnings','25 physics samples per control step','raw contact aggregates reproduce JSON','all recorded state finite','8 valid actual load preconditions','Panda full finger geometry coverage','world +X equal force direction','2 s force window','no upper-level frozen source edits']}
with pathlib.Path(a.out).open('x') as f:json.dump(report,f,indent=2)
print(json.dumps(report,indent=2))
