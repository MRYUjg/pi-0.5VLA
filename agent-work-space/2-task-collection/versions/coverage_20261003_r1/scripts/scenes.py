"""Explicit scene placements ONLY inside reset's native placement sampler."""
import hashlib,json,pathlib
import numpy as np
import mujoco
VERSION=pathlib.Path(__file__).resolve().parents[1]
MANIFEST=VERSION/'scenes.json'

def spec_for(seed):
    specs=json.loads(MANIFEST.read_text())['scenes']
    return next(s for s in specs if s['seed']==seed)

def quat_product(a,b):
    w,x,y,z=a;v,i,j,k=b
    return np.array([w*v-x*i-y*j-z*k,w*i+x*v+y*k-z*j,w*j-x*k+y*v+z*i,w*k+x*j-y*i+z*v])

def object_states(env):
    m,d=env.sim.model._model,env.sim.data._data;out={}
    for name,obj in env.env.objects_dict.items():
        bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,obj.root_body)
        out[name]={'position':d.xpos[bid].tolist(),'quaternion_wxyz':d.xquat[bid].tolist()}
    return out

def contact_check(env):
    m,d=env.sim.model._model,env.sim.data._data
    labels={}
    for name,obj in env.env.objects_dict.items():
        root=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,obj.root_body)
        for b in range(m.nbody):
            x=b
            while x:
                if x==root:labels[b]=name;break
                x=int(m.body_parentid[x])
    pairs=[];maximum=0.
    for c in d.contact:
        a,b=int(c.geom1),int(c.geom2);la=labels.get(int(m.geom_bodyid[a]));lb=labels.get(int(m.geom_bodyid[b]))
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,k) or '' for k in (a,b)]
        # Floor is legitimate support during native settling; record non-support collisions.
        robot_a=names[0].startswith(('robot0_','gripper0_'));robot_b=names[1].startswith(('robot0_','gripper0_'))
        relevant=(la and lb and la!=lb) or (la and robot_b) or (lb and robot_a)
        if relevant:
            mm=max(0.,-float(c.dist))*1000;maximum=max(maximum,mm)
            if mm>0:pairs.append({'geoms':names,'penetration_mm':mm})
    return {'non_support_penetration_mm':maximum,'pairs':pairs}

def install(env,scene):
    old=env.env._reset_internal
    env.coverage_scene=scene;env.coverage_record={}
    def reset_internal():
        sampler=env.env.placement_initializer;sample=sampler.sample
        def placed(*args,**kwargs):
            values=sample(*args,**kwargs)
            before={k:{'position':list(map(float,v[0])),'quat':list(map(float,v[1]))} for k,v in values.items()}
            for name,xy,yaw in [('alphabet_soup_1',scene['target_xy_m'],scene['target_yaw_offset_deg']),('basket_1',scene['basket_xy_m'],0)]:
                pos,quat,obj=values[name];pos=np.asarray(pos).copy();pos[:2]=xy
                angle=np.deg2rad(yaw)/2;quat=quat_product([np.cos(angle),0,0,np.sin(angle)],quat)
                values[name]=(pos,quat,obj)
            env.coverage_record={'scene':scene,'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'native_sampler_before_override':before,
               'configured_placements':{k:{'position':list(map(float,v[0])),'quat':list(map(float,v[1]))} for k,v in values.items()},
               'method':'Override sampler placements during reset only; preserve all objects/assets/goal/controller. No runtime object state injection.'}
            return values
        sampler.sample=placed
        try:return old()
        finally:sampler.sample=sample
    env.env._reset_internal=reset_internal

def wrap_build(native):
    def build(suite,task_id,robot='ECO65',gripper='ZhixingGripper',seed=7):
        scene=spec_for(seed);env,task=native(suite,task_id,robot,gripper,seed)
        install(env,scene);return env,task
    return build
