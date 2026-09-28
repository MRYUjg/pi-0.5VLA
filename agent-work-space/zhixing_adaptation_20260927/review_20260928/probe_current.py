import os,sys,json,pathlib,hashlib
import numpy as np
import mujoco
sys.path.insert(0,'/app/agent-work-space/zhixing_adaptation_20260927/scripts')
import n4_collect as C
import n3_grasp_test as N
from libero.libero import benchmark,get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
OUT=pathlib.Path('/app/agent-work-space/zhixing_adaptation_20260927/review_20260928')
def build(robot,gripper,suite='libero_spatial'):
 task=benchmark.get_benchmark_dict()[suite]().get_task(0)
 p=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file
 e=ControlEnv(bddl_file_name=p,robots=[robot],gripper_types=gripper,use_camera_obs=False,has_offscreen_renderer=False)
 e.seed(7); e.reset();return e
out={}
poses={}
for robot,gripper in [('Panda','PandaGripper'),('ECO65','PandaGripper'),('ECO65','ZhixingGripper')]:
 e=build(robot,gripper);s,m,d,g=N.ids(e);r=e.env.robots[0]
 pose={'pos':d.site_xpos[g].copy().tolist(),'R':d.site_xmat[g].copy().reshape(3,3).tolist(),'init_qpos':r.robot_model.init_qpos.tolist(),'controller_output_max':r.controller.output_max.tolist()}
 if robot=='ECO65' and gripper=='PandaGripper':
  old=np.array([-0.128227,-0.080175,-1.482052,-0.064915,1.563529,-1.699308]);d.qpos[r._ref_joint_pos_indexes]=old;mujoco.mj_forward(m,d)
  pose['old_qpos_grip_pos']=d.site_xpos[g].copy().tolist();pose['old_qpos_grip_R']=d.site_xmat[g].copy().reshape(3,3).tolist()
 poses[robot+'_'+gripper]=pose;e.close()
p=np.array(poses['Panda_PandaGripper']['pos']);ec=poses['ECO65_PandaGripper']
out['poses']=poses
out['eco_panda_current_offset_mm']=float(np.linalg.norm(np.array(ec['pos'])-p)*1000)
out['eco_panda_previous_offset_mm']=float(np.linalg.norm(np.array(ec['old_qpos_grip_pos'])-p)*1000)
out['eco_panda_orientation_delta_deg']=float(np.linalg.norm(N.rot_err_vec(np.array(poses['Panda_PandaGripper']['R']),np.array(ec['R'])))*180/np.pi)
e=build('ECO65','ZhixingGripper','libero_object');s,m,d,g=N.ids(e);r=e.env.robots[0]
out['gripper_joint_names']=r.gripper.joints
out['gripper_qpos_indices']=list(map(int,r._ref_gripper_joint_pos_indexes))
out['selected_names_1_5']=[r.gripper.joints[i] for i in [1,5]]
out['default_gripper_speed']=r.gripper.speed
out['control_timestep']=e.env.control_timestep
out['controller_output_max']=r.controller.output_max.tolist()
recs=[]
for f in sorted((C.ROOT/'raw/success').glob('*.npz')):
 with np.load(f,allow_pickle=False) as z:
  meta=json.loads(str(z['_meta_json']));q=z['qpos'];st=z['state'];actual=q[:,[r._ref_gripper_joint_pos_indexes[i] for i in [0,4]]]
  jmap={name:int(m.jnt_qposadr[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,name)]) for name in r.gripper.joints}
  # Restore last saved pre-action state then execute final action; not a full replay.
  d.qpos[:]=q[-1];d.qvel[:]=z['qvel'][-1];mujoco.mj_forward(m,d)
  recs.append({'file':f.name,'frames':len(q),'stored_state_tail_vs_declared_max_abs_rad':float(np.abs(st[:,-2:]-actual).max()),'state_tail_equals_support_joints':bool(np.array_equal(st[:,-2:],q[:,[7,11]])),'metadata_scales':[meta['osc_pos_scale'],meta['osc_rot_scale']],'metadata_speed':meta['gripper_speed'],'first_obs_vs_direct_grip_mm':float(np.linalg.norm(z['eef_pos'][0]-z['grip_site'][0])*1000),'last_pre_action_native_success':bool(e.check_success()),'has_full_xml':'xml' in z.files,'has_ctrl':'ctrl' in z.files,'has_gripper_current_action':'gripper_current_action' in z.files})
out['episodes']=recs;e.close()
(OUT/'current_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2),flush=True)
