#!/usr/bin/env python3
"""CPU-only diagnostic harness. No asset/controller/teacher/gate edits.
Uses legacy teacher with an observation-only recorder (no image/dataset output).
Each OS process runs one independently initialized trial; no forced exit.
"""
import argparse, datetime, hashlib, json, os, pathlib, sys, time
import numpy as np
import mujoco
from scipy.spatial import cKDTree
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eco65_expert_validation as EX
N, N3, N4 = EX.N, EX.N3, EX.N4
ROOT = HERE.parent

def dump(p, value):
    with pathlib.Path(p).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2,
                  default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())

def calibrate():
    # Analytic reference: settled 1 kg box on floor, sum of normal forces = mg.
    m = mujoco.MjModel.from_xml_string('<mujoco><option timestep="0.002"/><worldbody><geom type="plane" size="2 2 .1"/><body pos="0 0 .051"><freejoint/><geom type="box" size=".05 .05 .05" mass="1"/></body></worldbody></mujoco>')
    d = mujoco.MjData(m)
    for _ in range(1500): mujoco.mj_step(m,d)
    f = np.zeros(6,dtype=np.float64); normals=[]
    for i in range(d.ncon):
        mujoco.mj_contactForce(m,d,i,f);normals.append(float(f[0]))
    total=sum(normals)
    assert abs(total-9.81)<0.02, (total,normals)
    return {'known_mass_kg':1,'weight_N':9.81,'normal_force_N':total,'ncon':d.ncon,'pass':True}

def build(suite, task_id, robot='ECO65', gripper='ZhixingGripper', seed=7):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    task=benchmark.get_benchmark_dict()[suite]().get_task(task_id)
    env=ControlEnv(bddl_file_name=pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file,
                   robots=[robot],gripper_types=gripper,use_camera_obs=False,
                   has_renderer=False,has_offscreen_renderer=False)
    env.seed(seed)
    return env,task

def gap(env,pair=None):
    sim,m,d,_=N.ids(env);g=EX.pad_geoms_generic(sim)
    A=g['pad_left'] or g['all']['left']; B=g['pad_right'] or g['all']['right']
    ds=[]
    for a in A:
        pa=EX.surface_points(m,d,[a]); tree=cKDTree(pa)
        for b in B: ds.append(float(tree.query(EX.surface_points(m,d,[b]),k=1)[0].min())*1000)
    return min(ds) if pair=='min' else max(ds)

class PullComplete(BaseException): pass

class Monitor:
    def __init__(self,env):
        self.env=env;self.phase='init';self.rows=[];self.raw=[];self.buf=[];self.force=np.zeros(3)
        sim,self.m,self.d,self.gid=N.ids(env);m,d=self.m,self.d
        self.r=env.env.robots[0]; g=self.r.gripper
        self.bid,_=N4.find_body(sim,N4.goal_objects(env)[0]); self.objgeoms=set()
        def inside(b):
            while b:
                if b==self.bid:return True
                b=int(m.body_parentid[b])
            return False
        self.objgeoms={i for i in range(m.ngeom) if inside(int(m.geom_bodyid[i])) and (m.geom_contype[i] or m.geom_conaffinity[i])}
        pads=EX.pad_geoms_generic(sim)
        self.sides=[set(pads['all'][s]) for s in ('left','right')]
        self.aids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,n) for n in g.actuators]
        self.jids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in g.joints]
        self.qadr=[int(m.jnt_qposadr[j]) for j in self.jids];self.vadr=[int(m.jnt_dofadr[j]) for j in self.jids]
        self.mass=sum(float(m.body_mass[b]) for b in range(m.nbody) if inside(b))
        self.phase_pairs={};self.first={};self.last={};self.buffer_force=np.zeros(6,dtype=np.float64); self.current_world_force=np.zeros(3)
        self.step_orig=sim.step
        def step():
            d.xfrc_applied[self.bid,:3]=self.force
            self.step_orig()
            self.sample()
        sim.step=step
    def state(self):
        m,d=self.m,self.d;p=d.site_xpos[self.gid].copy();R=d.site_xmat[self.gid].reshape(3,3).copy()
        obj=d.xpos[self.bid].copy();relative=R.T@(obj-p)
        return {'time':float(d.time),'grip_site':p,'object_pos':obj,'object_in_grip':relative,
                'qpos':d.qpos.copy(),'qvel':d.qvel.copy(),'gripper_qpos':d.qpos[self.qadr].copy(),
                'ctrl':d.ctrl.copy(),'gripper_force':d.actuator_force[self.aids].copy(),
                'gripper_actuator_length':d.actuator_length[self.aids].copy(),
                'gripper_current_action':self.r.gripper.current_action.copy(),
                'object_rotation':d.xmat[self.bid].reshape(3,3).copy(),'grip_rotation':R,
                'native_success':bool(self.env.check_success())}
    def sample(self):
        m,d=self.m,self.d;fn=[0.,0.];contact=[0,0];pen=[0.,0.];otherfn=0.;world_force=np.zeros(3);pair=self.phase_pairs.setdefault(self.phase,{})
        for i in range(d.ncon):
            c=d.contact[i];g1,g2=int(c.geom1),int(c.geom2)
            if g1 not in self.objgeoms and g2 not in self.objgeoms:continue
            other=g2 if g1 in self.objgeoms else g1
            mujoco.mj_contactForce(m,d,i,self.buffer_force)
            world_force += (1 if g2 in self.objgeoms else -1)*np.asarray(c.frame).reshape(3,3).T@self.buffer_force[:3]
            normal=max(0.,float(self.buffer_force[0]));depth=max(0.,-float(c.dist))*1000
            side=next((s for s in (0,1) if other in self.sides[s]),None)
            if side is not None:
                fn[side]+=normal; contact[side]+=normal>1e-6;pen[side]=max(pen[side],depth)
            else:otherfn+=normal
            if normal>1e-6:
                name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,other)
                v=pair.setdefault(name,{'force_peak_N':0.,'penetration_peak_mm':0.,'active_samples':0})
                v['force_peak_N']=max(v['force_peak_N'],normal);v['penetration_peak_mm']=max(v['penetration_peak_mm'],depth);v['active_samples']+=1
        self.current_world_force=world_force.copy()
        row=[float(d.time),fn[0],fn[1],pen[0],pen[1],int(contact[0]>0),int(contact[1]>0),otherfn]+d.actuator_force[self.aids].tolist()+world_force.tolist()
        assert np.isfinite(row).all(), 'nonfinite contact/actuator data'
        self.buf.append(row);self.raw.append([len(self.rows)]+row)
    def finish_step(self,action,before):
        after=self.state();arr=np.asarray(self.buf);self.buf=[]
        row={'phase':self.phase,'step':len(self.rows),'action':np.asarray(action),'before':before,'after':after,
             'physics_samples':len(arr),'normal_force_mean_N':arr[:,1:3].mean(axis=0),
             'normal_force_peak_N':arr[:,1:3].max(axis=0),'penetration_peak_mm':arr[:,3:5].max(axis=0),
             'bilateral_fraction':float(np.mean((arr[:,5]>0)&(arr[:,6]>0))),
             'other_contact_force_peak_N':float(arr[:,7].max()),
             'actuator_force_mean':arr[:,8:8+len(self.aids)].mean(axis=0),'actuator_force_min':arr[:,8:8+len(self.aids)].min(axis=0),'actuator_force_max':arr[:,8:8+len(self.aids)].max(axis=0)}
        row['object_contact_resultant_mean_N']=arr[:,-3:].mean(axis=0)
        row['object_force_balance_mean_N']=row['object_contact_resultant_mean_N']+self.mass*self.m.opt.gravity+self.force
        self.rows.append(row); self.first.setdefault(self.phase,before);self.last[self.phase]=after

MON=None

def minimal_act(ep,action,phase):
    global MON
    if MON is None:MON=Monitor(ep.env)
    if MON.phase!=phase:print('PHASE',phase,'step',ep.n,flush=True)
    MON.phase=phase;before=MON.state();MON.buf=[]
    act=np.asarray(action,dtype=float);assert act.shape==(7,) and np.isfinite(act).all() and np.max(abs(act))<=1
    fr={'phase':phase,'grip_site':before['grip_site'],'qpos':before['qpos'],'action':act}
    obs,rew,done,info=ep.env.step(act)
    MON.finish_step(act,before)
    ep.raw_steps+=1;ep.obs=obs;ep.obs_raw_step=ep.raw_steps;ep.obs_origin='step';ep.done=bool(done)
    ep.t+=ep.dt;ep.n+=1;ep.frames.append(fr)
    return fr

def summarize(mon):
    result={}
    for ph in dict.fromkeys(r['phase'] for r in mon.rows):
        rs=[r for r in mon.rows if r['phase']==ph];s=rs[0]['before'];e=rs[-1]['after']
        rel=np.array([r['after']['object_in_grip'] for r in rs]);obj=np.array([r['after']['object_pos'] for r in rs])
        result[ph]={'steps':len(rs),'bilateral_fraction_mean':float(np.mean([r['bilateral_fraction'] for r in rs])),
          'normal_force_mean_N':np.mean([r['normal_force_mean_N'] for r in rs],axis=0),
          'object_contact_resultant_mean_N':np.mean([r['object_contact_resultant_mean_N'] for r in rs],axis=0),
          'object_force_balance_mean_N':np.mean([r['object_force_balance_mean_N'] for r in rs],axis=0),
          'other_contact_force_peak_N':max(r['other_contact_force_peak_N'] for r in rs),
          'actuator_force_mean':np.mean([r['actuator_force_mean'] for r in rs],axis=0),
          'penetration_peak_mm':np.max([r['penetration_peak_mm'] for r in rs],axis=0),
          'object_displacement_mm':float(np.linalg.norm(e['object_pos']-s['object_pos'])*1000),
          'object_in_grip_change_mm':float(np.linalg.norm(e['object_in_grip']-s['object_in_grip'])*1000),
          'object_in_grip_max_change_mm':float(np.linalg.norm(rel-s['object_in_grip'],axis=1).max()*1000),
          'object_z_min_m':float(obj[:,2].min()),'object_start':s['object_pos'],'object_end':e['object_pos'],
          'grip_start':s['grip_site'],'grip_end':e['grip_site'],
          'gripper_qpos_end':e['gripper_qpos'],'contacts':mon.phase_pairs.get(ph,{})}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--gripper',default='ZhixingGripper');ap.add_argument('--seed',type=int,default=47100)
    ap.add_argument('--mode',choices=['chain','pull','calibration'],default='chain');ap.add_argument('--force',type=float,default=0)
    ap.add_argument('--out',required=True);a=ap.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
    t0=time.time();cal=calibrate();print('FORCE_CALIBRATION',cal,flush=True)
    if a.mode=='calibration':dump(out/'result.json',cal);return
    os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
    N4.build=build;EX.jaw_gap_mm=gap;N4.Episode.act=minimal_act
    EX.install_generic_pad_geoms(print)
    env,_=build('libero_object',0,gripper=a.gripper,seed=a.seed)
    EX.install_generic_state_gripper(env,print)
    cat,_,_=N3.target_category(env);env.close()
    N3.install_object_substitution(cat,N3.block_object_class(.03,N3.BLOCK_DEPTH_M,N3.BLOCK_HEIGHT_M),print)
    env,_=build('libero_object',0,gripper=a.gripper,seed=a.seed)
    EX.install_generic_state_gripper(env,print)
    args=argparse.Namespace(prep_steps=120,grasp_margin_mm=4.,approach_height_m=.08,approach_max_steps=260,
          descend_max_steps=260,close_max_steps=700,close_min_steps=60,close_stable_checks=8,lift_m=.08,release_steps=60)
    pulled={};original=N3.ChainEpisode.servo_limited
    def servo(ep,target,R_target,grip_cmd,phase,max_steps,*pa,**kw):
        if phase=='transport' and a.mode=='pull':
            st=MON.state();hold=summarize(MON).get('hold',{})
            close=MON.first.get('close');liftmm=(st['object_pos'][2]-close['object_pos'][2])*1000
            valid=(hold.get('steps',0)==40 and hold.get('bilateral_fraction_mean',0)>.95 and
                   hold.get('object_in_grip_max_change_mm',1e9)<2 and liftmm>30 and hold.get('other_contact_force_peak_N',1e9)<1e-5)
            pulled.update({'valid_preload':bool(valid),'lift_before_pull_mm':liftmm,'hold_baseline':hold})
            if valid:
                # Horizontal tangent to closing direction. Applied at object COM; gravity remains active.
                pads=EX.pad_geoms_generic(N.ids(env)[0]);axis=MON.d.geom_xpos[pads['right']]-MON.d.geom_xpos[pads['left']]
                tangent=np.array([1.,0.,0.])
                assert abs(float(np.dot(axis/np.linalg.norm(axis),tangent)))<.05, 'world +X not tangential enough'
                pulled.update({'force_N':a.force,'force_world':a.force*tangent,'duration_s':2.,'criterion_relative_slip_mm':5.,'initial':st})
                MON.force=a.force*tangent
                ep.track(st['grip_site'],st['grip_rotation'],1.,'pull',40)
                MON.force[:]=0;MON.d.xfrc_applied[MON.bid,:]=0
                pulled['final']=MON.state();pull=summarize(MON)['pull'];pulled['measurement']=pull
                pulled['held_under_load']=bool(pull['object_in_grip_max_change_mm']<=5 and pull['bilateral_fraction_mean']>=.95)
            raise PullComplete()
        return original(ep,target,R_target,grip_cmd,phase,max_steps,*pa,**kw)
    N3.ChainEpisode.servo_limited=servo
    result={}
    try:
        result=EX.run_trial(env,args,{'seed':a.seed,'trial':1,'trial_label':'diagnostic_r1','width_mm':30},print)
    except PullComplete:result={'pull':pulled,'not_a_task_success_test':True}
    phases=summarize(MON) if MON else {}
    m=env.env.sim.model._model
    metadata={'version':'loaded_diagnostics_r3','generated_utc':datetime.datetime.utcnow().isoformat(),'gripper':a.gripper,'seed':a.seed,
      'mode':a.mode,'force_N':a.force,'calibration':cal,'rendering':False,'training_data':False,
      'teacher':'existing EX.run_trial, unchanged targets/gains/gates; N4 image recorder replaced by diagnostic numeric recorder',
      'physics_timestep_s':float(m.opt.timestep),'control_timestep_s':float(env.env.control_timestep),
      'object_mass_kg':MON.mass if MON else None,'mujoco_version':mujoco.__version__,
      'runtime':{'python':sys.version,'numpy':np.__version__,'mujoco_library':mujoco.__file__,'numba_disable_jit':os.environ.get('NUMBA_DISABLE_JIT')},
      'arm_qpos_indices':list(map(int,env.env.robots[0]._ref_joint_pos_indexes)),
      'arm_actuator_indices':list(map(int,env.env.robots[0]._ref_joint_actuator_indexes)),
      'joint_names':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,j) for j in range(m.njnt)],
      'joint_qposadr':m.jnt_qposadr,'joint_range':m.jnt_range,'joint_limited':m.jnt_limited,
      'actuator_names':[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,j) for j in range(m.nu)],
      'actuator_ctrlrange':m.actuator_ctrlrange,'actuator_forcerange':m.actuator_forcerange,
      'physics_warning_counts':env.env.sim.data._data.warning.number.copy(),
      'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [pathlib.Path(__file__),pathlib.Path(EX.__file__),pathlib.Path(N4.__file__),pathlib.Path(N3.__file__),pathlib.Path('/app/robosuite/models/assets/grippers/zhixing_gripper.xml'),pathlib.Path('/app/robosuite/models/grippers/zhixing_gripper.py')]},
      'wall_seconds':time.time()-t0,'result':result,'phases':phases}
    dump(out/'result.json',metadata)
    with (out/'environment.xml').open('x') as f:f.write(env.env.model.get_xml())
    if MON:
        with (out/'steps.jsonl').open('x') as f:
            for row in MON.rows:f.write(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
        np.savez_compressed(str(out/'physics.npz'),values=np.asarray(MON.raw),
             columns=np.array(['control_step','time','left_normal_N','right_normal_N','left_pen_mm','right_pen_mm','left_active','right_active','other_normal_N']+['actuator_force_'+str(i) for i in range(len(MON.aids))]+['object_contact_fx_N','object_contact_fy_N','object_contact_fz_N']))
    env.close()
    print('RESULT',out,'seconds',round(time.time()-t0,2),'phases',list(phases),flush=True)
if __name__=='__main__':main()
