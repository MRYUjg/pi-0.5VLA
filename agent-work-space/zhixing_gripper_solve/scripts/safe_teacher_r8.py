#!/usr/bin/env python3
"""Versioned geometry-aware teacher. Only process-local diagnostic routing changes."""
import json, pathlib, sys, hashlib
import numpy as np
import mujoco
import loaded_diagnostics_r5 as D
EX,N,N3,N4=D.EX,D.N,D.N3,D.N4
PLAN={}; TARGET=None; INITIAL=None; ACTIVE=False
CONTACT_ROWS=[]; GUARD_FAILURES=[]
MARGIN=.040
R_CARRY=None
class PathRejected(RuntimeError):pass

def name(m,kind,i):return mujoco.mj_id2name(m,kind,i) or ''
def bbox(m,d,ids):
    points=[]
    for g in ids:
        R=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g];typ=m.geom_type[g];s=m.geom_size[g]
        if typ==mujoco.mjtGeom.mjGEOM_MESH:
            mid=m.geom_dataid[g];start=m.mesh_vertadr[mid];v=m.mesh_vert[start:start+m.mesh_vertnum[mid]]
            points.append(v@R.T+c)
        elif typ==mujoco.mjtGeom.mjGEOM_BOX:
            v=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])*s
            points.append(v@R.T+c)
        else:
            # Conservative analytic enclosing world AABB for primitive shapes.
            if typ==mujoco.mjtGeom.mjGEOM_SPHERE:extent=np.repeat(s[0],3)
            elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=np.abs(R[:,2])*s[1]+s[0]
            elif typ==mujoco.mjtGeom.mjGEOM_CYLINDER:extent=np.abs(R[:,2])*s[1]+s[0]*np.sqrt(np.maximum(0,1-R[:,2]**2))
            elif typ==mujoco.mjtGeom.mjGEOM_ELLIPSOID:extent=np.sqrt((R*R)@(s*s))
            else:raise PathRejected('unsupported envelope geom '+str(typ))
            points.append(np.array([c-extent,c+extent]))
    a=np.vstack(points);return np.array([a.min(0),a.max(0)])
def geometry(env):
    sim,m,d,gid=N.ids(env);p,R=N.gripsite(sim,m,d,gid)
    obj,_=N4.find_body(sim,N4.goal_objects(env)[0]);basket,_=N4.find_body(sim,N4.goal_objects(env)[1])
    def belongs(b,root):
        while b:
            if b==root:return True
            b=int(m.body_parentid[b])
        return False
    gids=[i for i in range(m.ngeom) if name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith('gripper0_')]
    og=[i for i in range(m.ngeom) if belongs(int(m.geom_bodyid[i]),obj) and (m.geom_contype[i] or m.geom_conaffinity[i])]
    bg=[i for i in range(m.ngeom) if belongs(int(m.geom_bodyid[i]),basket) and (m.geom_contype[i] or m.geom_conaffinity[i])]
    cg=[i for i in gids if m.geom_contype[i] or m.geom_conaffinity[i]]
    mount=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'gripper0_right_gripper')
    mount_R=d.xmat[mount].reshape(3,3)
    return {'tcp':p,'R':R,'tcp_in_mount':mount_R.T@(p-d.xpos[mount]),'tcp_rotation_in_mount':mount_R.T@R,'object':bbox(m,d,og),'gripper_outer':bbox(m,d,gids),'gripper_collision':bbox(m,d,cg),'basket':bbox(m,d,bg),'basket_pos':d.xpos[basket].copy(),'geom_names':[name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in gids], 'basket_geoms':bg,'gripper_geoms':cg,'object_geoms':og}

orig_servo=N3.ChainEpisode.servo_limited
orig_track=N3.ChainEpisode.track
orig_act=D.minimal_act

def act(ep,action,phase):
    global INITIAL
    result=orig_act(ep,action,phase)
    if INITIAL is None:INITIAL=geometry(ep.env)
    if D.MON.rows[-1].get('gripper_external_force_peak_N',0)>.1:
        GUARD_FAILURES.append({'step':ep.n,'phase':phase,'reason':'gripper contacted non-target environment'})
        raise PathRejected('gripper contacted non-target environment')
    if PLAN.get('vertical_target') is not None:
        geo=geometry(ep.env);row=D.MON.rows[-1]
        row['safe_geometry']={k:geo[k] for k in ('tcp','object','gripper_outer','basket','basket_pos')}
        shift=np.linalg.norm(geo['basket_pos']-INITIAL['basket_pos'])
        if shift>.005:raise PathRejected('basket moved >5 mm')
        if row['robot_basket_force_peak_N']>.1 or (phase in ('clearance_lift','transport','place') and row['payload_basket_force_peak_N']>.1):
            GUARD_FAILURES.append({'step':ep.n,'phase':phase,'reason':'unexpected basket contact'})
            raise PathRejected('robot or carried payload contacted basket before release')
        if phase=='transport' and min(geo['object'][0,2],geo['gripper_outer'][0,2])<INITIAL['basket'][1,2]+.010:
            raise PathRejected('actual envelope lost overhead clearance')
    return result

class SafeMonitor(D.Monitor):
    def __init__(self,env):
        super().__init__(env)
        m=self.m
        self.basket={i for i in range(m.ngeom) if name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith('basket_1_') and (m.geom_contype[i] or m.geom_conaffinity[i])}
        self.robot={i for i in range(m.ngeom) if name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith(('robot0_','gripper0_')) and (m.geom_contype[i] or m.geom_conaffinity[i])}
        self.gripper={i for i in self.robot if name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith('gripper0_')}
        self.extra=[]
    def sample(self):
        super().sample();rf=pf=gf=0.;pairs=[];f=np.zeros(6)
        for i in range(self.d.ncon):
            c=self.d.contact[i];a,b=int(c.geom1),int(c.geom2)
            if (a in self.gripper and b not in self.objgeoms and b not in self.robot) or (b in self.gripper and a not in self.objgeoms and a not in self.robot):
                mujoco.mj_contactForce(self.m,self.d,i,f);gf=max(gf,float(f[0]))
            other=b if a in self.basket else (a if b in self.basket else None)
            if other is None:continue
            mujoco.mj_contactForce(self.m,self.d,i,f)
            if other in self.robot:rf=max(rf,float(f[0]))
            if other in self.objgeoms:pf=max(pf,float(f[0]))
            if f[0]>.000001 and (other in self.robot or other in self.objgeoms):pairs.append([name(self.m,mujoco.mjtObj.mjOBJ_GEOM,a),name(self.m,mujoco.mjtObj.mjOBJ_GEOM,b),float(f[0]),float(c.dist)])
        row={'step':len(self.rows),'phase':self.phase,'time':float(self.d.time),'robot_basket_force_N':rf,'payload_basket_force_N':pf,'gripper_external_force_N':gf,'pairs':pairs}
        self.extra.append(row);CONTACT_ROWS.append(row)
    def finish_step(self,action,before):
        super().finish_step(action,before)
        self.rows[-1]['robot_basket_force_peak_N']=max(r['robot_basket_force_N'] for r in self.extra)
        self.rows[-1]['payload_basket_force_peak_N']=max(r['payload_basket_force_N'] for r in self.extra)
        self.rows[-1]['gripper_external_force_peak_N']=max(r['gripper_external_force_N'] for r in self.extra)
        self.extra=[]

ENV=None
original_build=D.build

def tracked_build(*a,**kw):
    global ENV
    ENV,task=original_build(*a,**kw)
    return ENV,task

def box_grasp_plan(sim,bodies,closing_axis=1,opening_mm=89.96,margin_mm=4.,band_m=.006):
    """Exact box-plane intersection; diagnostic boxes only, not a general object planner."""
    m,d=sim.model._model,sim.data._data;geo=geometry(ENV)
    if any(m.geom_type[g]!=mujoco.mjtGeom.mjGEOM_BOX for g in geo['object_geoms']):
        raise PathRejected('box-plane grasp planner requires box geometry')
    lo,hi=geo['object'];z=float(hi[2]-.004);cross=[]
    signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
    for g in geo['object_geoms']:
        v=(signs*m.geom_size[g])@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
        for i in range(8):
            for j in range(i+1,8):
                if np.count_nonzero(signs[i]!=signs[j])!=1:continue
                za,zb=v[i,2],v[j,2]
                if abs(zb-za)<1e-12:
                    if abs(z-za)<1e-9:cross.extend([v[i],v[j]])
                elif min(za,zb)-1e-10<=z<=max(za,zb)+1e-10:
                    cross.append(v[i]+(v[j]-v[i])*((z-za)/(zb-za)))
    if len(cross)<3:return {'best':None,'scan':[],'reason':'no_box_plane_section'}
    cross=np.array(cross);low=cross.min(0);high=cross.max(0);width=float(high[closing_axis]-low[closing_axis]);center=(low+high)/2
    # Nominal open-to-closed shape sweep in a scratch data object; add 10 mm
    # for load-dependent linkage deflection. Live simulation qpos is never overwritten.
    scratch=mujoco.MjData(m);scratch.qpos[:]=d.qpos
    grip=ENV.env.robots[0].gripper
    joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in grip.joints]
    qa=m.jnt_qposadr[joints];closed=np.asarray(grip.init_qpos)
    outer=[i for i in range(m.ngeom) if name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith('gripper0_')]
    depths=[]
    for fraction in np.linspace(0,1,11):
        scratch.qpos[qa]=fraction*closed
        mujoco.mj_kinematics(m,scratch)
        depths.append(float(geo['tcp'][2]-bbox(m,scratch,outer)[0,2]))
    tip_depth=max(depths)+.010
    clearance=float(z-tip_depth-lo[2]);ok=bool(width*1000<=opening_mm-margin_mm and clearance>.002)
    PLAN['grasp_geometry']={'method':'box edge / horizontal plane intersection','section_points':cross,'z':z,'width_m':width,'open_geometry':geo,'nominal_sweep_lower_depths_m':depths,'linkage_deflection_reserve_m':.010,'lower_envelope_reserve_m':tip_depth,'floor_clearance_m':clearance,'top_inset_m':.004}
    return {'best':(z,width,clearance,float(center[0]),float(center[1])) if ok else None,'scan':[[z,width*1000,clearance*1000,ok]],'object_z_range':[float(lo[2]),float(hi[2])],'table_z':float(lo[2]),'opening_budget_mm':opening_mm-margin_mm}

def rotation(axis,angle):
    axis=axis/np.linalg.norm(axis);K=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)

def ik(m,source,gid,target,R,qseed):
    # Separate MjData: no changes to live physics, no resetting executed arm joints.
    d=mujoco.MjData(m);d.qpos[:]=source.qpos
    js=[i for i in range(m.njnt) if name(m,mujoco.mjtObj.mjOBJ_JOINT,i).startswith('robot0_joint')]
    qa=m.jnt_qposadr[js];va=m.jnt_dofadr[js];q=np.array(qseed).copy()
    jp=np.zeros((3,m.nv));jr=jp.copy()
    for k in range(600):
        d.qpos[qa]=q;mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d)
        e=np.r_[target-d.site_xpos[gid],N.rot_err_vec(R,d.site_xmat[gid].reshape(3,3))]
        if np.linalg.norm(e[:3])<.001 and np.linalg.norm(e[3:])<.005:break
        mujoco.mj_jacSite(m,d,jp,jr,gid);J=np.vstack([jp[:,va],jr[:,va]])
        dq=J.T@np.linalg.solve(J@J.T+.001*np.eye(6),e)
        q=np.clip(q+.6*dq,m.jnt_range[js,0]+.005,m.jnt_range[js,1]-.005)
    return q,{'position_error_m':float(np.linalg.norm(e[:3])),'rotation_error_rad':float(np.linalg.norm(e[3:])),'pass':bool(np.linalg.norm(e[:3])<.002 and np.linalg.norm(e[3:])<.01)}

def corners(box):return np.array([[x,y,z] for x in box[:,0] for y in box[:,1] for z in box[:,2]])

def make_plan(env,R):
    geo=geometry(env);sim,m,d,gid=N.ids(env);p=geo['tcp'];top=geo['basket'][1,2]
    js=[i for i in range(m.njnt) if name(m,mujoco.mjtObj.mjOBJ_JOINT,i).startswith('robot0_joint')]
    base=d.xpos[m.jnt_bodyid[js[0]]];rad=geo['basket'].mean(0)-base;axis=np.array([rad[1],-rad[0],0.])
    local={k:(corners(geo[k])-p)@geo['R'] for k in ('object','gripper_outer')}
    trials=[]
    for degree in (0,15,25,35,45,55):
        Rc=rotation(axis,np.deg2rad(degree))@R
        obj=local['object']@Rc.T;outer=local['gripper_outer']@Rc.T
        safe_z=top+MARGIN-min(obj[:,2].min(),outer[:,2].min())
        up=p.copy();up[2]=max(p[2],safe_z)
        across=up.copy();across[:2]=geo['basket'].mean(0)[:2]-obj.mean(0)[:2]
        place=across.copy();place[2]=top+.015-obj[:,2].min()
        q=d.qpos[m.jnt_qposadr[js]].copy();checks=[]
        # Validate both endpoints and intermediate horizontal poses, not only the destination.
        targets=[up]+[up+(across-up)*u for u in np.linspace(.125,1,8)]+[place]
        for target in targets:
            q,check=ik(m,d,gid,target,Rc,q);checks.append(check)
            if not check['pass']:break
        trials.append({'tilt_deg':degree,'up':up.copy(),'across':across.copy(),'place':place.copy(),'checks':checks})
        if len(checks)==len(targets) and all(c['pass'] for c in checks):
            return geo,Rc,up,across,place,trials
    PLAN['ik_candidates']=trials
    raise PathRejected('no kinematically feasible safe-height carry pose')

def arrived(ep,target,R,phase,reported_ok,tol):
    _,_,d,gid=N.ids(ep.env);p=np.array(d.site_xpos[gid]);actual_R=d.site_xmat[gid].reshape(3,3)
    errors=[float(np.linalg.norm(np.asarray(target)-p)),float(np.linalg.norm(N.rot_err_vec(R,actual_R)))]
    valid=bool(reported_ok and errors[0]<tol and errors[1]<.03)
    PLAN.setdefault('stage_arrivals',{})[phase]={'ok':valid,'servo_reported_ok':bool(reported_ok),'target':np.asarray(target).copy(),'target_rotation':R.copy(),'actual':p,'error':errors,'position_tolerance_m':tol,'rotation_tolerance_rad':.03}
    if not valid:raise PathRejected(phase+' did not converge at actual post-action state')
    return True

def recheck_path(env,start,end,R):
    sim,m,d,gid=N.ids(env);js=[i for i in range(m.njnt) if name(m,mujoco.mjtObj.mjOBJ_JOINT,i).startswith('robot0_joint')]
    q=d.qpos[m.jnt_qposadr[js]].copy();checks=[]
    for u in np.linspace(0,1,9):
        q,c=ik(m,d,gid,np.asarray(start)+(np.asarray(end)-start)*u,R,q);checks.append(c)
        if not c['pass']:raise PathRejected('corrected target/path failed IK preflight')
    return checks

def servo(ep,target,R,grip,phase,max_steps,*args,**kw):
    global ACTIVE,TARGET,R_CARRY
    if phase=='transport':
        hold=D.summarize(D.MON).get('hold',{})
        if hold.get('steps',0)!=40 or hold.get('bilateral_fraction_mean',0)<.95 or hold.get('other_contact_force_peak_N',1)>1e-5:
            raise PathRejected('invalid loaded hold before carry')
        geo,Rc,up,across,place,trials=make_plan(ep.env,R);top=geo['basket'][1,2]
        PLAN.update({'measured_loaded_geometry':geo,'margin_m':MARGIN,'vertical_target':up,'horizontal_target':across,'release_target':place,'carry_rotation':Rc,'ik_candidates':trials,'tilt_deg':trials[-1]['tilt_deg']})
        ACTIVE=True;R_CARRY=Rc
        ok,k=orig_servo(ep,up,Rc,grip,'clearance_lift',400,tol=.008)
        PLAN['vertical_reached']=arrived(ep,up,Rc,'clearance_lift',ok,.008)
        actual=geometry(ep.env)
        if min(actual['object'][0,2],actual['gripper_outer'][0,2])<top+.010:raise PathRejected('vertical clearance not achieved')
        # Re-measure payload offsets after tilt/lift. X/Y target compensates observed grasp slip.
        across=across.copy()
        across[:2]=actual['basket'].mean(0)[:2]-(actual['object'].mean(0)-actual['tcp'])[:2]
        PLAN['horizontal_target']=across.copy()
        PLAN['corrected_horizontal_ik']=recheck_path(ep.env,actual['tcp'],across,Rc)
        ok,k=orig_servo(ep,across,Rc,grip,'transport',max_steps,*args,**kw)
        PLAN['horizontal_reached']=arrived(ep,across,Rc,'transport',ok,.008)
        actual=geometry(ep.env)
        place=across.copy();place[2]=top+.015+(actual['tcp'][2]-actual['object'][0,2])
        PLAN['corrected_place_ik']=recheck_path(ep.env,actual['tcp'],place,Rc)
        PLAN['release_target']=place
        PLAN['place_started']=True
        ACTIVE=False;TARGET=place
        return ok,k
    if phase=='place':
        if TARGET is None:raise PathRejected('place without verified transport')
        target=TARGET;R=R_CARRY
    if phase in ('approach','descend'):
        kw['max_step_rot']=.06
        kw['max_step_pos']=.020
    ok,k=orig_servo(ep,target,R,grip,phase,max_steps,*args,**kw)
    arrived(ep,target,R,phase,ok,float(kw.get('tol',.004)))
    if phase=='place':PLAN['place_reached']=True
    return ok,k

def track(ep,target,R,grip,phase,steps,*args,**kw):
    if phase in ('release','post_release_settle','post_release_window'):
        if TARGET is None:raise PathRejected('release without safe path')
        target=TARGET;R=R_CARRY
    return orig_track(ep,target,R,grip,phase,steps,*args,**kw)

if __name__=='__main__':
    D.build=tracked_build;N.grasp_plan=box_grasp_plan
    D.Monitor=SafeMonitor
    N3.ChainEpisode.servo_limited=servo;N3.ChainEpisode.track=track;D.minimal_act=act
    # main caches the current servo, so geometry-aware wrapper remains in the chain.
    original_dump=D.dump
    def annotated_dump(path,value):
        if pathlib.Path(path).name=='result.json':
            value['teacher']='safe_teacher_r8 geometry-aware staged candidate; no controller/asset changes'
            value['sources'][N.__file__]=hashlib.sha256(pathlib.Path(N.__file__).read_bytes()).hexdigest()
            value['sources'][__file__]=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
            value['safe_path']=PLAN
            value['teacher_protocol']={'carry_margin_m':MARGIN,'actual_clearance_floor_m':.01,'max_basket_shift_m':.005,'pregrasp_max_rotation_increment_rad':.06,'other_max_rotation_increment_rad':.04,'pregrasp_max_translation_increment_m':.020,'other_max_translation_increment_m':.012,'position_gain_scale_m':EX.TEACHER_GAIN_POS_M,'rotation_gain_scale_rad':EX.TEACHER_GAIN_ROT_RAD,'upstream_controller_changed':False,'ik_position_tolerance_m':.002,'ik_rotation_tolerance_rad':.01}
            pe=value['result'].get('phase_end',{})
            if 'horizontal_target' in PLAN:pe['transport_target']=PLAN['horizontal_target']
            if 'release_target' in PLAN:pe['place_point']=PLAN['release_target']
            for phase,target_key in [('transport','horizontal_target'),('place','release_target')]:
                if phase in PLAN.get('stage_arrivals',{}):
                    arrival=PLAN['stage_arrivals'][phase]
                    pe[phase+'_residual_m']=arrival['error'][0]
                    trace=value['result'].get('phase_trace',{}).get(phase,{})
                    trace['target']=arrival['target'];trace['err_end_m']=arrival['error'][0]
                    if phase=='transport':
                        e=np.asarray(arrival['target'])-np.asarray(arrival['actual']);pe['transport_err_world_m']=e;pe['transport_err_horizontal_m']=float(np.linalg.norm(e[:2]))
                    trace['err_end_horizontal_m']=float(np.linalg.norm((np.asarray(arrival['target'])-arrival['actual'])[:2]));trace['err_end_vertical_m']=float(arrival['target'][2]-arrival['actual'][2])
            value['version']='safe_teacher_r8'
            value['safe_teacher_success']=bool(value['result'].get('success_n4_style') and PLAN.get('vertical_reached') and PLAN.get('horizontal_reached') and PLAN.get('place_reached') and not GUARD_FAILURES and not value['result'].get('notes'))
            value['guard_failures']=GUARD_FAILURES
        return original_dump(path,value)
    D.dump=annotated_dump
    D.main()
    out=pathlib.Path(sys.argv[sys.argv.index('--out')+1])
    with (out/'basket_contacts.jsonl').open('x') as f:
        for row in CONTACT_ROWS:f.write(json.dumps(row)+'\n')
    D.dump(out/'safe_path.json',{'version':'safe_teacher_r8','source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'plan':PLAN,'initial_geometry':INITIAL,'note':'Teacher candidate, unchanged assets/controller and upstream gates.'})
