import json,os,pathlib
import numpy as np
from collect import T
out=pathlib.Path('/app/agent-work-space/2-task-collection/runs/coverage_initialization_probe_r2');out.mkdir(exist_ok=False)
os.environ['EXTRA_JOINT_PD_PROFILE']='legacy';os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
rows=[]
for seed in [53100,53101,54100]:
 env,task=T.D.build('libero_object',0,seed=seed)
 old=env.env._reset_internal; samples=[]
 def reset_internal():
  sampler=env.env.placement_initializer;orig=sampler.sample
  def sample(*a,**kw):
   result=orig(*a,**kw)
   samples.append({k:{'pos':list(v[0]),'quat':list(v[1])} for k,v in result.items()})
   return result
  sampler.sample=sample
  return old()
 env.env._reset_internal=reset_internal
 env.reset();sim,m,d,gid=T.N.ids(env)
 objects=lambda:{n:d.xpos[T.N4.find_body(sim,n)[0]].tolist() for n in env.env.objects_dict}
 before=objects(); reset_q=d.qpos.copy()
 T.N4.quiet_hold(env,120,-1)
 rows.append({'seed':seed,'sampler_outputs':samples,'reset_objects':before,'settled_objects':objects(),'reset_qpos':reset_q.tolist()})
 env.close()
(out/'result.json').write_text(json.dumps(rows,indent=2,default=float))
print(json.dumps(rows,indent=2,default=float))
