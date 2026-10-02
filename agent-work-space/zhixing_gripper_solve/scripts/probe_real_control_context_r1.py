"""Read-only runtime/profile inspection at saved failure; no parameter changes."""
import json,pathlib,sys,numpy as np,mujoco
import real_object_teacher_r12 as T
from robosuite.models.robots.robot_model import extra_joint_pd_profile,extra_joint_pd_kp_for
p=pathlib.Path(sys.argv[1]);r=json.loads((p/'result.json').read_text());last=json.loads((p/'steps.jsonl').read_text().splitlines()[-1])['after'];env,_=T.D.build(r['suite'],0,seed=r['seed']);env.reset();robot=env.env.robots[0];sim,m,d,gid=T.N.ids(env)
d.qpos[:]=last['qpos'];d.qvel[:]=last['qvel'];d.ctrl[:]=last['ctrl'];mujoco.mj_forward(m,d)
js=[i for i in range(m.njnt) if T.S.name(m,mujoco.mjtObj.mjOBJ_JOINT,i).startswith('robot0_joint')];q=d.qpos[m.jnt_qposadr[js]];v=d.qvel[m.jnt_dofadr[js]];kp=extra_joint_pd_kp_for(type(robot.robot_model).__name__)
result={'version':'probe_real_control_context_r1','scope':'Read-only source/runtime and offline saved-state reconstruction; not an A/B causal test','profile':extra_joint_pd_profile(),'extra_joint_pd_kp':kp,'extra_joint_pd_kv':2*np.sqrt(kp),'robot_class':type(robot.robot_model).__name__,'controller_eef_name':robot.controller.eef_name,'teacher_site':T.S.name(m,mujoco.mjtObj.mjOBJ_SITE,gid),'osc_kp':np.asarray(robot.controller.kp),'osc_uncoupling':robot.controller.uncoupling,'position_output_max':np.asarray(robot.controller.output_max),'saved_joint_q':q,'init_joint_q':np.asarray(robot.init_qpos),'extra_pd_torque_Nm':kp*(robot.init_qpos-q)-2*np.sqrt(kp)*v,'torque_limits':[robot.torque_limits[0],robot.torque_limits[1]],'parameters_modified':False}
T.D.dump(p.parent/(p.name+'_control_context.json'),result);print(json.dumps(result,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()));env.close()
