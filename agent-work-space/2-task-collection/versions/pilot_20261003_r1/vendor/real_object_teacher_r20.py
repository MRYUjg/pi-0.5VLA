#!/usr/bin/env python3
"""Native LIBERO can/bowl teacher; immutable assets and numeric CPU evidence."""
import argparse,datetime,hashlib,json,os,pathlib,sys,time,traceback
import numpy as np
import mujoco
import safe_teacher_r8 as S
D,EX,N,N3,N4=S.D,S.EX,S.N,S.N3,S.N4
HERE=pathlib.Path(__file__).resolve().parent

class RealMonitor(S.SafeMonitor):
    def __init__(self,env):
        super().__init__(env)
        self.basket=set(S.geometry(env)['basket_geoms'])
    def sample(self):
        super().sample()
        contacts=[];f=np.zeros(6)
        for i in range(self.d.ncon):
            c=self.d.contact[i];a,b=int(c.geom1),int(c.geom2)
            other=b if a in self.robot and b not in self.robot else (a if b in self.robot and a not in self.robot else None)
            if other is not None:
                mujoco.mj_contactForce(self.m,self.d,i,f)
                if f[0]>1e-6:contacts.append({'geoms':[S.name(self.m,mujoco.mjtObj.mjOBJ_GEOM,j) for j in (a,b)],'target_object':other in self.objgeoms,'normal_N':float(f[0]),'penetration_mm':max(0.,-float(c.dist))*1000})
        S.CONTACT_ROWS[-1]['all_robot_external_contacts']=contacts
    def finish_step(self,action,before):
        super().finish_step(action,before)
        self.rows[-1]['geometry']=S.geometry(self.env)

def occupied_intervals(m,d,gids,origin,axis):
    intervals=[]
    for g in gids:
        if m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX:raise S.PathRejected('unsupported target collision geom')
        R=d.geom_xmat[g].reshape(3,3);o=R.T@(origin-d.geom_xpos[g]);v=R.T@axis;s=m.geom_size[g]
        lo,hi=-np.inf,np.inf
        for k in range(3):
            if abs(v[k])<1e-10:
                if abs(o[k])>s[k]:lo,hi=1.,0.;break
            else:
                a,b=sorted(((-s[k]-o[k])/v[k],(s[k]-o[k])/v[k]));lo=max(lo,a);hi=min(hi,b)
        if lo<=hi and np.isfinite([lo,hi]).all():intervals.append([float(lo),float(hi),[int(g)]])
    merged=[]
    for lo,hi,gs in sorted(intervals):
        if merged and lo<=merged[-1][1]+.0001:merged[-1][1]=max(merged[-1][1],hi);merged[-1][2]+=gs
        else:merged.append([lo,hi,gs])
    return merged

def open_approach_clearance(env,pin,R,qseed):
    sim,m,d,gid=N.ids(env);geo=S.geometry(env);scratch=mujoco.MjData(m);scratch.qpos[:]=d.qpos
    js=[i for i in range(m.njnt) if S.name(m,mujoco.mjtObj.mjOBJ_JOINT,i).startswith('robot0_joint')];qa=m.jnt_qposadr[js]
    fingers=set(geo['gripper_geoms']);robot={i for i in range(m.ngeom) if S.name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith(('robot0_','gripper0_'))}
    checks=[];q=qseed.copy()
    for dz in (.080,.060,.040,.020,0.):
        q,ik=S.ik(m,d,gid,pin+[0,0,dz],R,q)
        if not ik['pass']:return False,[{'dz':dz,'ik':ik}]
        scratch.qpos[qa]=q;mujoco.mj_forward(m,scratch);hits=[]
        for k in range(scratch.ncon):
            c=scratch.contact[k];a,b=int(c.geom1),int(c.geom2)
            if ((a in fingers and b not in robot) or (b in fingers and a not in robot)) and c.dist<-.00001:
                hits.append({'geoms':[S.name(m,mujoco.mjtObj.mjOBJ_GEOM,j) for j in (a,b)],'penetration_m':-float(c.dist)})
        checks.append({'dz':dz,'ik':ik,'collisions':hits})
        if hits:return False,checks
    return True,checks

def plan_grasp(env,case,use_surface=False):
    sim,m,d,gid=N.ids(env);geo=S.geometry(env);p,R=N.gripsite(sim,m,d,gid)
    closing=EX.closing_axis_world_generic(env);center=geo['object'].mean(0);lo,hi=geo['object'];opening=D.gap(env)/1000
    js=[i for i in range(m.njnt) if S.name(m,mujoco.mjtObj.mjOBJ_JOINT,i).startswith('robot0_joint')]
    base=d.xpos[m.jnt_bodyid[js[0]]];rad=center-base;tiltaxis=np.array([rad[1],-rad[0],0.])
    candidates=[]
    if use_surface:
        bid,_=N4.find_body(sim,N4.goal_objects(env)[0]);body_R=d.xmat[bid].reshape(3,3)
        local_centers={g:body_R.T@(d.geom_xpos[g]-d.xpos[bid]) for g in geo['object_geoms']}
        top_center=max(v[2] for v in local_centers.values())
        origins=[(None,d.geom_xpos[g]+body_R[:,2]*.008) for g,v in local_centers.items() if v[2]>top_center-.003]
    else:
        origins=[]
        for inset in ([.008,.016,.024] if case=='can' else [.006,.010,.014]):
            origin=center.copy();origin[2]=hi[2]-inset;origins.append((inset,origin))
    # Search whole-body grips for cans and individual material intervals for bowl rims.
    orientations=[(tilt,S.rotation(tiltaxis,np.deg2rad(tilt))@R,'world') for tilt in (0,15,25,35)]
    if use_surface:orientations += [(0,body_R@R,'object')]
    for tilt,Rc,frame in orientations:
        C=Rc@(R.T@closing);C=C/np.linalg.norm(C)
        for inset,origin in origins:
            sections=occupied_intervals(m,d,geo['object_geoms'],origin,C)
            choices=([[sections[0][0],sections[-1][1],sum((v[2] for v in sections),[])]] if case=='can' and sections else sections)
            for a,b,gids in choices:
                width=b-a
                if not .001<width<opening-.004:continue
                pin=origin+.5*(a+b)*C
                # Conservative nominal finger extension reserve, validated against scene contacts at execution.
                if pin[2]-lo[2]<.026:continue
                approach=pin+np.array([0,0,.08]);q=d.qpos[m.jnt_qposadr[js]].copy();checks=[]
                for tgt in (approach,pin):
                    q,ch=S.ik(m,d,gid,tgt,Rc,q);checks.append(ch)
                    if not ch['pass']:break
                candidate={'pin':pin,'R':Rc,'opening_m':opening,'material_width_m':width,'intervals':sections,'selected_interval':[a,b],'selected_geom_ids':gids,'tilt_deg':tilt,'top_inset_m':inset,'origin_world':origin,'surface_fallback':use_surface,'orientation_frame':frame,'ik':checks,'strategy':'whole_body' if case=='can' else 'rim_pinch'}
                candidates.append(candidate)
                if len(checks)==2 and all(ch['pass'] for ch in checks):
                    if use_surface:
                        clear,scan=open_approach_clearance(env,pin,Rc,d.qpos[m.jnt_qposadr[js]].copy());candidate['open_approach_collision_checks']=scan
                        if not clear:continue
                    candidate['score']=np.linalg.norm(pin-p)+.001*tilt
    feasible=[c for c in candidates if 'score' in c]
    if not feasible:
        if case=='bowl' and not use_surface:
            S.PLAN['primary_grasp_candidates']=candidates
            return plan_grasp(env,case,True)
        S.PLAN['grasp_candidates']=candidates;raise S.PathRejected('no feasible native-object grasp')
    best=min(feasible,key=lambda c:c['score']);S.PLAN['grasp_candidates']=candidates;S.PLAN['grasp_selected']=best
    return best

# Teacher-level bounded residual correction. Controller gains/action scales remain unchanged.
BIAS=np.zeros(3)
BOWL_MODE=False
SURFACE_MODE=False
def residual_action(ep,target,R,grip,phase,max_step_pos=.012,max_step_rot=.04):
    global BIAS
    no_pd = BOWL_MODE and os.environ.get('EXTRA_JOINT_PD_PROFILE') == 'eco65_no_extra_pd'
    if BOWL_MODE:max_step_rot=max(max_step_rot,.12 if phase in ('clearance_lift','transport','place','release','post_release_settle','post_release_window') else (.12 if SURFACE_MODE or phase in ('lift','hold') else .08));max_step_pos=max(max_step_pos,.030 if phase in ('clearance_lift','transport','place') else .020)
    sim,m,d,gid=N.ids(ep.env);p,actual_R=N.gripsite(sim,m,d,gid)
    err=np.asarray(target)-p;rot=N.rot_err_vec(R,actual_R)
    if np.linalg.norm(err)<(.050 if BOWL_MODE else .020) and np.linalg.norm(rot)<(.10 if BOWL_MODE else .03):
        BIAS=np.clip(BIAS+.05*err,-(.025 if BOWL_MODE else .012),.025 if BOWL_MODE else .012)
    if no_pd:
        BIAS[:]=0.;max_step_pos=.030 if phase in ('clearance_lift','transport','place') else .020;max_step_rot=.12
    dv=err+BIAS
    if np.linalg.norm(dv)>max_step_pos:dv*=max_step_pos/np.linalg.norm(dv)
    if np.linalg.norm(rot)>max_step_rot:rot*=max_step_rot/np.linalg.norm(rot)
    ap=np.clip(dv/N3.TEACHER_GAIN_POS_M,-max_step_pos/N3.OSC_POS_UNIT_M,max_step_pos/N3.OSC_POS_UNIT_M)
    ar=np.clip(rot/(.10 if no_pd else N3.TEACHER_GAIN_ROT_RAD),-max_step_rot/N3.OSC_ROT_UNIT_RAD,max_step_rot/N3.OSC_ROT_UNIT_RAD)
    ep.act(np.r_[ap,ar,grip],phase)
    D.MON.rows[-1]['teacher_bias_m']=BIAS.copy()
    return N.gripsite(sim,m,d,gid)

def residual_servo(ep,target,R,grip,phase,max_steps,tol=.004,rot_tol=.03,max_step_pos=.012,max_step_rot=.04,**kw):
    settled=0
    for k in range(max_steps):
        if ep.n>=N3.FRAME_LIMIT:return False,k
        p,Ra=residual_action(ep,target,R,grip,phase,max_step_pos,max_step_rot)
        row=D.MON.rows[-1]
        slow=np.linalg.norm(np.asarray(row['after']['grip_site'])-row['before']['grip_site'])<.0005
        inside=np.linalg.norm(np.asarray(target)-p)<tol and np.linalg.norm(N.rot_err_vec(R,Ra))<rot_tol
        settled=settled+1 if inside and slow else 0
        if (settled>=5 if BOWL_MODE else inside):return True,k+1
    return False,max_steps

def residual_track(ep,target,R,grip,phase,steps,*args,**kw):
    for k in range(steps):
        if ep.n>=N3.FRAME_LIMIT:raise S.PathRejected('teacher frame limit')
        residual_action(ep,target,R,grip,phase)


def main():
    global BOWL_MODE,SURFACE_MODE
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['can','bowl'],required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    requested_profile=os.environ.get('EXTRA_JOINT_PD_PROFILE','legacy');requested_kp=os.environ.get('EXTRA_JOINT_PD_KP','')
    os.environ['EXTRA_JOINT_PD_PROFILE']='legacy';os.environ['EXTRA_JOINT_PD_KP']=''
    BOWL_MODE=a.case=='bowl'
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time();os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
    calibration=D.calibrate();suite='libero_object' if a.case=='can' else 'libero_spatial'
    env,task=D.build(suite,0,seed=a.seed);EX.install_generic_pad_geoms(print);EX.install_generic_state_gripper(env,print)
    D.Monitor=RealMonitor;N4.Episode.act=S.act;N3.ChainEpisode.servo_limited=S.servo;N3.ChainEpisode.track=S.track
    S.orig_servo=residual_servo;S.orig_track=residual_track
    ep=N3.ChainEpisode(env,None);ep.start();obs=N4.quiet_hold(env,120,-1);ep.raw_steps+=120;ep.refresh_obs(obs,'native_reset_settled')
    os.environ['EXTRA_JOINT_PD_PROFILE']=requested_profile;os.environ['EXTRA_JOINT_PD_KP']=requested_kp
    sim,m,d,gid=N.ids(env);initial=S.geometry(env);initial_qpos=d.qpos.copy();initial_obj=initial['object'].mean(0);initial_native=bool(env.check_success());failure=None;stage='planning';gates={};plan=None
    try:
        if initial_native:raise S.PathRejected('native goal already true at initialization')
        plan=plan_grasp(env,a.case);SURFACE_MODE=plan.get('surface_fallback',False);pin,R=plan['pin'],plan['R'];print('GRASP',a.case,plan['strategy'],pin,'tilt',plan['tilt_deg'],'width',plan['material_width_m'],flush=True)
        stage='approach';ep.servo_limited(pin+[0,0,.08],R,-1,stage,260,tol=.004)
        stage='descend';ep.servo_limited(pin,R,-1,stage,260,tol=.001 if a.case=='bowl' else .004)
        stage='close';ep.track(pin,R,1,stage,80)
        close_rows=[row for row in D.MON.rows if row['phase']=='close'][-10:];gates['closure_bilateral']=len(close_rows)==10 and np.mean([row['bilateral_fraction'] for row in close_rows])>=.95
        if not gates['closure_bilateral']:raise S.PathRejected('closure did not establish sustained bilateral contact')
        stage='lift';lift=pin+[0,0,.08];ep.servo_limited(lift,R,1,stage,260,tol=.004)
        if S.geometry(env)['object'].mean(0)[2]-initial_obj[2]<.030:raise S.PathRejected('object lift <30 mm')
        stage='hold';ep.track(lift,R,1,stage,40)
        h=[r for r in D.MON.rows if r['phase']=='hold'];gates['hold_height']=all(r['geometry']['object'].mean(0)[2]-initial_obj[2]>=.025 for r in h)
        if not gates['hold_height']:raise S.PathRejected('hold height <25 mm')
        stage='transport';ep.servo_limited(np.zeros(3),R,1,stage,220,tol=.008)
        stage='place';ep.servo_limited(S.TARGET,R,1,stage,160,tol=.008)
        stage='release';ep.track(S.TARGET,R,-1,stage,60)
        stage='post_release_settle';ep.track(S.TARGET,R,-1,stage,40)
        stage='post_release_window';ep.track(S.TARGET,R,-1,stage,20)
    except Exception as e:failure={'stage':stage,'type':type(e).__name__,'reason':str(e),'traceback':traceback.format_exc()};print('FAIL',failure['reason'],flush=True)
    phases=D.summarize(D.MON) if D.MON else {};rows=D.MON.rows if D.MON else []
    h=[r for r in rows if r['phase']=='hold'];post=[r for r in rows if r['phase']=='post_release_window']
    gates.update({'initial_goal_false':not initial_native,'all_stages_complete':failure is None,'lift_30mm':bool(h and h[0]['geometry']['object'].mean(0)[2]-initial_obj[2]>=.03),'hold_40steps':len(h)==40,'bilateral_hold':bool(h and np.mean([r['bilateral_fraction'] for r in h])>=.95),'unsupported_hold':bool(h and max(r['other_contact_force_peak_N'] for r in h)<1e-5),'penetration_2mm':max((max(r['penetration_peak_mm']) for r in rows),default=999)<=2.,'max_step_disp_20mm':ep.max_step_disp<.02,'native_success_window':len(post)==20 and all(r['after']['native_success'] for r in post),'post_drift_5mm':len(post)==20 and np.linalg.norm(post[-1]['after']['object_pos']-post[0]['before']['object_pos'])<.005,'target_support_contact':len(post)==20 and all(r['payload_basket_force_peak_N']>1e-5 for r in post),'no_physics_warnings':not np.any(d.warning.number)})
    sources=[pathlib.Path(__file__),pathlib.Path(S.__file__),pathlib.Path(D.__file__),pathlib.Path(EX.__file__),pathlib.Path(N.__file__),pathlib.Path(N3.__file__),pathlib.Path(N4.__file__),pathlib.Path('/app/robosuite/models/assets/grippers/zhixing_gripper.xml'),pathlib.Path('/app/robosuite/models/grippers/zhixing_gripper.py')]
    assets=pathlib.Path('/app/third_party/libero/libero/libero/assets');asset=assets/('stable_hope_objects/alphabet_soup/alphabet_soup.xml' if a.case=='can' else 'stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml');sources.append(asset)
    result={'version':'real_object_teacher_r20','case':a.case,'suite':suite,'task_name':task.name,'seed':a.seed,'object_name':N4.goal_objects(env)[0],'recipient':N4.goal_objects(env)[1],'original_assets':True,'object_substitution':False,'initial_qpos':initial_qpos,'initial_geometry':initial,'initial_state_sha256':hashlib.sha256(initial_qpos.tobytes()).hexdigest(),'success':all(gates.values()),'gates':gates,'failure':failure,'plan':S.PLAN,'phases':phases,'calibration':calibration,'object_mass_kg':D.MON.mass if D.MON else None,'mujoco_version':mujoco.__version__,'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},'physics_warnings':d.warning.number.copy(),'control_contract':N4.controller_action_contract(env),'max_step_disp_m':ep.max_step_disp,'wall_seconds':time.time()-start,'teacher_residual_correction':{'gain_per_step':.05,'axis_cap_m':.025 if BOWL_MODE else .012,'activation_position_m':.050 if BOWL_MODE else .020,'activation_rotation_rad':.10 if BOWL_MODE else .03,'bowl_rotation_increment_floor_rad':{'carry_and_release':.12,'grasp':.08,'lift_hold':.12,'surface_fallback_grasp':.12},'bowl_translation_increment_floor_m':{'clearance_lift_transport_place':.030,'other':.020}},'guard_failures':S.GUARD_FAILURES,'scope':'native can/bowl task diagnostics; CPU no rendering or training data'}
    from robosuite.models.robots.robot_model import extra_joint_pd_profile,extra_joint_pd_kp_for
    result['runtime_profile']={'initialization_profile':'legacy','profile_applied_after_quiet_reset':True,'extra_joint_pd_profile':extra_joint_pd_profile(),'extra_joint_pd_kp':extra_joint_pd_kp_for(type(env.env.robots[0].robot_model).__name__),'controller_eef_name':env.env.robots[0].controller.eef_name}
    result['teacher_runtime_revision']={'applies_to':'bowl only; can retains r15 teacher behavior','no_pd_action_scaling':'position error / 0.02 m; rotation error / 0.10 rad','no_pd_integral_enabled':False,'no_pd_increment_caps':{'carry_place_m':.030,'other_m':.020,'rotation_rad':.12},'arrival_consecutive_steps':5,'arrival_max_translation_per_step_m':.0005,'acceptance_thresholds_changed':False}
    D.dump(out/'result.json',result);(out/'environment.xml').write_text(env.env.model.get_xml())
    for filename,data in [('steps.jsonl',rows),('recipient_contacts.jsonl',S.CONTACT_ROWS)]:
        with (out/filename).open('x') as f:
            for row in data:f.write(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
    if D.MON:np.savez_compressed(out/'physics.npz',values=np.asarray(D.MON.raw),columns=np.array(['control_step','time_s','left_normal_N','right_normal_N','left_penetration_mm','right_penetration_mm','left_contact','right_contact','other_normal_N']+['actuator_force_'+str(i) for i in D.MON.aids]+['world_force_x_N','world_force_y_N','world_force_z_N']))
    env.close();print('RESULT',a.case,a.seed,result['success'],'failed',[k for k,v in gates.items() if not v],flush=True)
if __name__=='__main__':main()
