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
    if PLAN.get('vertical_target') is not None:
        geo=geometry(ep.env);row=D.MON.rows[-1]
        row['safe_geometry']={k:geo[k] for k in ('tcp','object','gripper_outer','basket','basket_pos')}
        shift=np.linalg.norm(geo['basket_pos']-INITIAL['basket_pos'])
        if shift>.005:raise PathRejected('basket moved >5 mm')
        if row['robot_basket_force_peak_N']>.1 or (phase in ('clearance_lift','transport','place') and row['payload_basket_force_peak_N']>.1):
            GUARD_FAILURES.append({'step':ep.n,'phase':phase,'reason':'unexpected basket contact'})
            raise PathRejected('robot or carried payload contacted basket before release')
        if phase=='transport' and min(geo['object'][0,2],geo['gripper_outer'][0,2])<INITIAL['basket'][1,2]+.005:
            raise PathRejected('actual envelope lost overhead clearance')
    return result

class SafeMonitor(D.Monitor):
    def __init__(self,env):
        super().__init__(env)
        m=self.m
        self.basket={i for i in range(m.ngeom) if name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith('basket_1_') and (m.geom_contype[i] or m.geom_conaffinity[i])}
        self.robot={i for i in range(m.ngeom) if name(m,mujoco.mjtObj.mjOBJ_GEOM,i).startswith(('robot0_','gripper0_')) and (m.geom_contype[i] or m.geom_conaffinity[i])}
        self.extra=[]
    def sample(self):
        super().sample();rf=pf=0.;pairs=[];f=np.zeros(6)
        for i in range(self.d.ncon):
            c=self.d.contact[i];a,b=int(c.geom1),int(c.geom2)
            other=b if a in self.basket else (a if b in self.basket else None)
            if other is None:continue
            mujoco.mj_contactForce(self.m,self.d,i,f)
            if other in self.robot:rf=max(rf,float(f[0]))
            if other in self.objgeoms:pf=max(pf,float(f[0]))
            if f[0]>.000001 and (other in self.robot or other in self.objgeoms):pairs.append([name(self.m,mujoco.mjtObj.mjOBJ_GEOM,a),name(self.m,mujoco.mjtObj.mjOBJ_GEOM,b),float(f[0]),float(c.dist)])
        row={'step':len(self.rows),'phase':self.phase,'time':float(self.d.time),'robot_basket_force_N':rf,'payload_basket_force_N':pf,'pairs':pairs}
        self.extra.append(row);CONTACT_ROWS.append(row)
    def finish_step(self,action,before):
        super().finish_step(action,before)
        self.rows[-1]['robot_basket_force_peak_N']=max(r['robot_basket_force_N'] for r in self.extra)
        self.rows[-1]['payload_basket_force_peak_N']=max(r['payload_basket_force_N'] for r in self.extra)
        self.extra=[]

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

def servo(ep,target,R,grip,phase,max_steps,*args,**kw):
    global ACTIVE,TARGET,R_CARRY
    if phase=='transport':
        geo,Rc,up,across,place,trials=make_plan(ep.env,R);top=geo['basket'][1,2]
        PLAN.update({'measured_loaded_geometry':geo,'margin_m':MARGIN,'vertical_target':up,'horizontal_target':across,'release_target':place,'carry_rotation':Rc,'ik_candidates':trials,'tilt_deg':trials[-1]['tilt_deg']})
        ACTIVE=True;R_CARRY=Rc
        ok,k=orig_servo(ep,up,Rc,grip,'clearance_lift',400,tol=.008)
        PLAN['vertical_reached']=bool(ok);PLAN['vertical_error']=ep.last_err
        if not ok:raise PathRejected('vertical clearance target did not converge')
        actual=geometry(ep.env)
        if min(actual['object'][0,2],actual['gripper_outer'][0,2])<top+.010:raise PathRejected('vertical clearance not achieved')
        # Re-measure payload offsets after tilt/lift. X/Y target compensates observed grasp slip.
        across=across.copy()
        across[:2]=actual['basket'].mean(0)[:2]-(actual['object'].mean(0)-actual['tcp'])[:2]
        PLAN['horizontal_target']=across.copy()
        ok,k=orig_servo(ep,across,Rc,grip,'transport',max_steps,*args,**kw)
        PLAN['horizontal_reached']=bool(ok);PLAN['horizontal_error']=ep.last_err
        if not ok:raise PathRejected('horizontal target did not converge')
        actual=geometry(ep.env)
        place=across.copy();place[2]=top+.015+(actual['tcp'][2]-actual['object'][0,2])
        PLAN['release_target']=place
        PLAN['place_started']=True
        ACTIVE=False;TARGET=place
        return ok,k
    if phase=='place':
        if TARGET is None:raise PathRejected('place without verified transport')
        target=TARGET;R=R_CARRY
    if phase in ('approach','descend'):
        kw['max_step_rot']=.06
    ok,k=orig_servo(ep,target,R,grip,phase,max_steps,*args,**kw)
    sim,m,d,gid=N.ids(ep.env);actual_p,actual_R=N.gripsite(sim,m,d,gid)
    actual_error=[float(np.linalg.norm(np.array(target)-actual_p)),float(np.linalg.norm(N.rot_err_vec(R,actual_R)))]
    PLAN.setdefault('stage_arrivals',{})[phase]={'ok':bool(ok),'target':np.array(target),'actual':actual_p,'error':actual_error}
    if not ok:raise PathRejected(phase+' did not converge')
    if phase=='place':PLAN['place_reached']=True
    return ok,k

def track(ep,target,R,grip,phase,steps,*args,**kw):
    if phase in ('release','post_release_settle','post_release_window'):
        if TARGET is None:raise PathRejected('release without safe path')
        target=TARGET;R=R_CARRY
    return orig_track(ep,target,R,grip,phase,steps,*args,**kw)

if __name__=='__main__':
    D.Monitor=SafeMonitor
    N3.ChainEpisode.servo_limited=servo;N3.ChainEpisode.track=track;D.minimal_act=act
    # main caches the current servo, so geometry-aware wrapper remains in the chain.
    original_dump=D.dump
    def annotated_dump(path,value):
        if pathlib.Path(path).name=='result.json':
            value['teacher']='safe_teacher_r6 geometry-aware staged candidate; no controller/asset changes'
            value['sources'][__file__]=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()
            value['safe_path']=PLAN
            value['version']='safe_teacher_r6'
            value['safe_teacher_success']=bool(value['result'].get('success_n4_style') and PLAN.get('vertical_reached') and PLAN.get('horizontal_reached') and PLAN.get('place_reached') and not GUARD_FAILURES and not value['result'].get('notes'))
            value['guard_failures']=GUARD_FAILURES
        return original_dump(path,value)
    D.dump=annotated_dump
    D.main()
    out=pathlib.Path(sys.argv[sys.argv.index('--out')+1])
    with (out/'basket_contacts.jsonl').open('x') as f:
        for row in CONTACT_ROWS:f.write(json.dumps(row)+'\n')
    D.dump(out/'safe_path.json',{'version':'safe_teacher_r6','source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'plan':PLAN,'initial_geometry':INITIAL,'note':'Teacher candidate, unchanged assets/controller and upstream gates.'})
