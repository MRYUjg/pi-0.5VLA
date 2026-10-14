"""Native Spatial task selection. No placement or runtime state overrides."""
import hashlib,json,pathlib
import numpy as np
import mujoco
VERSION=pathlib.Path(__file__).resolve().parents[1]
MANIFEST=VERSION/'scenes.json'
def spec_for(seed):
    return next(s for s in json.loads(MANIFEST.read_text())['jobs'] if s['seed']==seed)
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


def wrap_build(native):
    def build(suite,task_id,robot='ECO65',gripper='ZhixingGripper',seed=7):
        spec=spec_for(seed)
        assert suite=='libero_spatial'
        repo=VERSION.parents[3]
        assert hashlib.sha256((repo/spec['bddl_path']).read_bytes()).hexdigest()==spec['bddl_sha256']
        env,task=native(suite,spec['task_id'],robot,gripper,seed)
        assert task.name==spec['name'] and task.language==spec['benchmark_prompt']
        env.coverage_record={'scene':spec,'manifest_sha256':hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),'method':'native reset(seed); no placement or state overrides'}
        return env,task
    return build
