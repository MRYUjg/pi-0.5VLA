"""Scratch MjData kinematic diagnosis; never writes the running episode."""
import pathlib,json,numpy as np,mujoco
import safe_teacher_r2 as S
root=pathlib.Path('/app/agent-work-space/zhixing_gripper_solve')
p=root/'evidence/safe_path_20261002_r1/ZhixingGripper_chain_s47100_f0'
m=mujoco.MjModel.from_xml_path(str(p/'environment.xml'));d=mujoco.MjData(m)
rows=[json.loads(x) for x in (p/'steps.jsonl').read_text().splitlines()]
state=next(x['after'] for x in reversed(rows) if x['phase']=='hold');d.qpos[:]=state['qpos'];mujoco.mj_forward(m,d)
gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'gripper0_grip_site')
js=[i for i in range(m.njnt) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) or '').startswith('robot0_joint')]
qa=m.jnt_qposadr[js];va=m.jnt_dofadr[js];q0=d.qpos[qa].copy();R=np.array(rows[0]['before']['grip_rotation']);plan=json.loads((p/'safe_path.json').read_text())['plan']
results=[]
for label,target in [('up',plan['vertical_target'])]+[('across_x%g'%dx,np.array(plan['horizontal_target'])+[dx,0,0]) for dx in [0,-.02,-.04]]:
 d.qpos[:]=state['qpos'];q=q0.copy();target=np.array(target)
 for k in range(1500):
  d.qpos[qa]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
  e=np.r_[target-d.site_xpos[gid],S.N.rot_err_vec(R,d.site_xmat[gid].reshape(3,3))]
  if np.linalg.norm(e)<1e-6:break
  jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,gid);J=np.vstack([jp[:,va],jr[:,va]])
  dq=J.T@np.linalg.solve(J@J.T+.0025*np.eye(6),e);q=np.clip(q+.6*dq,m.jnt_range[js,0],m.jnt_range[js,1])
 results.append({'label':label,'target':target.tolist(),'position_error_m':float(np.linalg.norm(e[:3])),'rotation_error_rad':float(np.linalg.norm(e[3:])),'q':q.tolist(),'iterations':k+1})
print(json.dumps(results,indent=2))
