import os,sys,pathlib,json,inspect,textwrap,types
import numpy as np
os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
sys.path.insert(0,'/app/agent-work-space/zhixing_adaptation_20260927/scripts')
import n3_osc_full_chain as F
import n4_collect as C
import n3_grasp_test as N
from libero.libero import benchmark,get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.robots.single_arm import SingleArm
O=C.ROOT/'rework_review_20260928'
task=benchmark.get_benchmark_dict()['libero_object']().get_task(0)
bddl=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
orig=SingleArm.control
src=textwrap.dedent(inspect.getsource(orig));assert 'joint_kp = 3.0' in src
scope={};exec(src.replace('joint_kp = 3.0','joint_kp = 0.0'),orig.__globals__,scope)
no_pd=scope['control']
rows=[]
for pd,coupled in [(True,False),(False,False),(False,True)]:
 SingleArm.control=orig if pd else no_pd
 e=ControlEnv(bddl_file_name=bddl,robots=['ECO65'],gripper_types='ZhixingGripper',use_camera_obs=False,has_offscreen_renderer=False)
 # Images are unused placeholders only to reuse the recorder; no visual checks are performed.
 for method in ['reset','step']:
  fn=getattr(e,method)
  def wrap(*a,_fn=fn,_method=method,**kw):
   out=_fn(*a,**kw);obs=out if _method=='reset' else out[0]
   obs['agentview_image']=np.zeros((1,1,3),dtype=np.uint8);obs['robot0_eye_in_hand_image']=np.zeros((1,1,3),dtype=np.uint8)
   return out
  setattr(e,method,wrap)
 for move in ['ry015','yaw_plus90','yaw_minus90']:
  ep=F.ChainEpisode(e,None);e.seed(41700);ep.start();r=e.env.robots[0];r.controller.uncoupling=not coupled
  obs=C.quiet_hold(e,120,-1);ep.raw_steps+=120;ep.refresh_obs(obs,'probe');ep.act(np.r_[np.zeros(6),-1],'settle')
  p,R=N.gripsite(*N.ids(e));p=p.copy();R=R.copy()
  T=F.expm_so3(np.array([0,.15,0]))@R if move=='ry015' else R@F.rotz(np.pi/2*(1 if move=='yaw_plus90' else -1))
  ok,steps=ep.servo_limited(p,T,-1,'control_probe',300 if move=='ry015' else 400,tol=.005,rot_tol=.03 if move=='ry015' else .02,stall_patience=100)
  ep.track(p,T,-1,'hold',10);p1,R1=N.gripsite(*N.ids(e));c=r.controller
  rec={'extra_pd':pd,'uncouple_pos_ori':not coupled,'move':move,'steps':steps,'converged':ok,'pos_error_mm':float(np.linalg.norm(p1-p)*1000),'rot_error_rad':float(np.linalg.norm(N.rot_err_vec(T,R1))),'rot_error_deg':float(np.degrees(np.linalg.norm(N.rot_err_vec(T,R1)))),'runtime_controller_config_keys':sorted(r.controller_config.keys()),'kp':c.kp.tolist(),'q':np.array(r._joint_positions).tolist(),'jacobian_singular_values':np.linalg.svd(c.J_full,compute_uv=False).tolist()}
  rows.append(rec);print(json.dumps(rec),flush=True);(O/'control_probe.json').write_text(json.dumps(rows,indent=2));ep.frames=[]
 e.close()
SingleArm.control=orig
