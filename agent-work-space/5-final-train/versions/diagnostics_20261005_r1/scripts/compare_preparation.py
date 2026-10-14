import pathlib,json,h5py,numpy as np
R=pathlib.Path('/app/agent-work-space/5-final-train');rows=[]
for t in [0,2,3,5,6,8]:
 name=f'task{t:02d}_s{57100+t}'
 with h5py.File(R/'runs/reference_baseline_20261005_r1'/name/'preparation.h5','r') as f,h5py.File(R/'runs/reference_preflight_20261005_r2'/name/'preparation.h5','r') as g:
  errors={k:float(np.max(abs(f['preparation/'+k][...]-g['preparation/'+k][...]))) for k in ('actions','qpos_before','qpos_after','time_before','time_after')}
  for k in ('image','wrist_image'):errors[k]=max(int(np.max(abs(f['preparation_images/'+k][i].astype(int)-g['preparation_images/'+k][i].astype(int)))) for i in (0,119))
 rows.append({'task':t,'errors':errors,'pass':all(v==0 for v in errors.values())})
out={'pass':all(x['pass'] for x in rows),'same_host_control_without_new_auditor':True,'records':rows}
with (R/'reports/same_host_initialization_observer_comparison_r1.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
