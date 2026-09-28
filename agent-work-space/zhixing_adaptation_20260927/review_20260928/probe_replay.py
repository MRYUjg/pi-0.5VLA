import os,sys,pathlib,json
import numpy as np
import mujoco
sys.path.insert(0,'/app/agent-work-space/zhixing_adaptation_20260927/scripts')
import n4_collect as C
import n3_grasp_test as N
from libero.libero import benchmark,get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
O=C.ROOT/'review_20260928'
task=benchmark.get_benchmark_dict()['libero_object']().get_task(0)
bddl=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
f=sorted((C.ROOT/'raw/success').glob('*.npz'))[0]
z=np.load(f,allow_pickle=False);meta=json.loads(str(z['_meta_json']))
rows=[]
for speed,fixacc in [(0.2,False),(0.01,False),(0.01,True)]:
 os.environ['ZHIXING_GRIPPER_SPEED']=str(speed)
 e=ControlEnv(bddl_file_name=bddl,robots=['ECO65'],gripper_types='ZhixingGripper',use_camera_obs=False,has_offscreen_renderer=False)
 e.seed(meta['seed']);e.reset();s,m,d,g=N.ids(e);r=e.env.robots[0]
 d.qpos[:]=z['init_qpos'];d.qvel[:]=z['init_qvel'];d.qacc_warmstart[:]=z['init_qacc_warmstart'];mujoco.mj_forward(m,d)
 for meth in ['reset_goal','update']:
  fn=getattr(r.controller,meth)
  try:fn(force=True)
  except TypeError:fn()
 if fixacc:r.gripper.current_action=np.ones(1)
 rec={'speed':speed,'restore_open_accumulator':fixacc,'initial_accumulator':np.array(r.gripper.current_action).tolist(),'errors':{}}
 for t,a in enumerate(z['actions']):
  obs,reward,done,info=e.step(a)
  if t+1 in [1,2,5,10,20,40,60,100,200,400] and t+1<len(z['qpos']):
   er=np.abs(d.qpos-z['qpos'][t+1]);rec['errors'][str(t+1)]={'all':float(er.max()),'arm':float(er[:6].max()),'gripper':float(er[6:12].max()),'objects':float(er[12:].max())}
 rec['final_success']=bool(e.check_success());rec['last_step_done']=bool(done);rows.append(rec);e.close()
(O/'replay_probe.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2),flush=True)
