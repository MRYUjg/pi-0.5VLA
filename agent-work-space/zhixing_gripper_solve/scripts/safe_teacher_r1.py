#!/usr/bin/env python3
"""Versioned geometry-aware teacher. Only process-local diagnostic routing changes."""
import json, pathlib, sys, hashlib
import numpy as np
import mujoco
import loaded_diagnostics_r5 as D
EX,N,N3,N4=D.EX,D.N,D.N3,D.N4
PLAN={}; TARGET=None; INITIAL=None; ACTIVE=False
MARGIN=.020
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
    return {'tcp':p,'R':R,'object':bbox(m,d,og),'gripper_outer':bbox(m,d,gids),'gripper_collision':bbox(m,d,cg),'basket':bbox(m,d,bg),'basket_pos':d.xpos[basket].copy(),'geom_names':[name(m,mujoco.mjtObj.mjOBJ_GEOM,i) for i in gids], 'basket_geoms':bg,'gripper_geoms':cg,'object_geoms':og}

orig_servo=N3.ChainEpisode.servo_limited
orig_track=N3.ChainEpisode.track
orig_act=D.minimal_act

def act(ep,action,phase):
    global INITIAL
    result=orig_act(ep,action,phase)
    if INITIAL is None:INITIAL=geometry(ep.env)
    if ACTIVE:
        geo=geometry(ep.env);row=D.MON.rows[-1]
        row['safe_geometry']={k:geo[k] for k in ('tcp','object','gripper_outer','basket','basket_pos')}
        shift=np.linalg.norm(geo['basket_pos']-INITIAL['basket_pos'])
        if shift>.005:raise PathRejected('basket moved >5 mm')
        sim,m,d,_=N.ids(ep.env)
        moving=set(geo['gripper_geoms']+geo['object_geoms']);basket=set(geo['basket_geoms'])
        for c in d.contact:
            if ((int(c.geom1) in moving and int(c.geom2) in basket) or (int(c.geom2) in moving and int(c.geom1) in basket)) and c.dist<-.0005:
                raise PathRejected('payload/gripper contacted basket during clearance motion')
        if phase=='transport' and min(geo['object'][0,2],geo['gripper_outer'][0,2])<INITIAL['basket'][1,2]+.005:
            raise PathRejected('actual envelope lost overhead clearance')
    return result

def servo(ep,target,R,grip,phase,max_steps,*args,**kw):
    global ACTIVE,TARGET
    if phase=='transport':
        geo=geometry(ep.env);p=geo['tcp'];top=geo['basket'][1,2]
        lowest=min(geo['object'][0,2],geo['gripper_outer'][0,2])
        safe_z=max(p[2],top+MARGIN+(p[2]-lowest))
        center=geo['object'].mean(0);basketcenter=geo['basket'].mean(0)
        up=p.copy();up[2]=safe_z
        across=up.copy();across[:2]=basketcenter[:2]-(center[:2]-p[:2])
        # Release while object bottom is 15 mm above rim: avoids inserting wide hand into basket.
        place=across.copy();place[2]=top+.015+(p[2]-geo['object'][0,2])
        if place[2]>safe_z:raise PathRejected('release target above clearance plane')
        PLAN.update({'measured_loaded_geometry':geo,'margin_m':MARGIN,'vertical_target':up,'horizontal_target':across,'release_target':place,'object_offset_world':center-p,'minimum_envelope_z':lowest})
        ACTIVE=True
        ok,k=orig_servo(ep,up,R,grip,'clearance_lift',260,tol=.004)
        PLAN['vertical_reached']=bool(ok)
        if not ok:raise PathRejected('vertical clearance target did not converge')
        actual=geometry(ep.env)
        if min(actual['object'][0,2],actual['gripper_outer'][0,2])<top+.010:raise PathRejected('vertical clearance not achieved')
        ok,k=orig_servo(ep,across,R,grip,'transport',max_steps,*args,**kw)
        PLAN['horizontal_reached']=bool(ok)
        if not ok:raise PathRejected('horizontal target did not converge')
        ACTIVE=False;TARGET=place
        return ok,k
    if phase=='place':
        if TARGET is None:raise PathRejected('place without verified transport')
        target=TARGET
    ok,k=orig_servo(ep,target,R,grip,phase,max_steps,*args,**kw)
    if not ok:raise PathRejected(phase+' did not converge')
    return ok,k

def track(ep,target,R,grip,phase,steps,*args,**kw):
    if phase in ('release','post_release_settle','post_release_window'):
        if TARGET is None:raise PathRejected('release without safe path')
        target=TARGET
    return orig_track(ep,target,R,grip,phase,steps,*args,**kw)

if __name__=='__main__':
    N3.ChainEpisode.servo_limited=servo;N3.ChainEpisode.track=track;D.minimal_act=act
    # main caches the current servo, so geometry-aware wrapper remains in the chain.
    D.main()
    out=pathlib.Path(sys.argv[sys.argv.index('--out')+1])
    D.dump(out/'safe_path.json',{'version':'safe_teacher_r1','source_sha256':hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),'plan':PLAN,'initial_geometry':INITIAL,'note':'Teacher candidate, unchanged assets/controller and upstream gates.'})
