import json,pathlib,sys,numpy as np,mujoco
import real_object_teacher_r5 as T
p=pathlib.Path(sys.argv[1]);r=json.loads((p/'result.json').read_text());last=json.loads((p/'steps.jsonl').read_text().splitlines()[-1])['after']
env,_=T.D.build(r['suite'],0,seed=r['seed']);env.reset();sim,m,d,gid=T.N.ids(env)
d.qpos[:]=last['qpos'];d.qvel[:]=last['qvel'];d.ctrl[:]=last['ctrl'];mujoco.mj_forward(m,d)
cs=[];f=np.zeros(6)
for i in range(d.ncon):
 c=d.contact[i];names=[T.S.name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)]
 if any(n.startswith(('robot0_','gripper0_')) for n in names):
  mujoco.mj_contactForce(m,d,i,f);cs.append({'pair':names,'normal_N':float(f[0]),'distance':float(c.dist)})
jo=[]
for j in range(m.njnt):
 name=T.S.name(m,mujoco.mjtObj.mjOBJ_JOINT,j)
 if name.startswith('robot0_joint'):
  v=m.jnt_dofadr[j];jo.append({'joint':name,'position':float(d.qpos[m.jnt_qposadr[j]]),'range':m.jnt_range[j].tolist(),'bias_Nm':float(d.qfrc_bias[v]),'constraint_Nm':float(d.qfrc_constraint[v]),'actuator_Nm':float(d.qfrc_actuator[v])})
result={'mode':'offline reconstructed mj_forward; not a live force trace','contacts':cs,'joints':jo};print(json.dumps(result,indent=2));T.D.dump(p.parent/(p.name+'_failure_probe.json'),result);env.close()
