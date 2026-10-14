"""Observation-only adapter around frozen r20 teacher; one episode per process."""
import argparse, hashlib, json, os, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'vendor'))
import numpy as np
import h5py
import mujoco
import real_object_teacher_r20 as T
from robosuite.utils import transform_utils as TU
from robosuite import macros
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING


def dump(path, value):
    with pathlib.Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def append(group, key, value):
    value = np.asarray(value)
    if key not in group:
        compression = {'compression': 'gzip', 'compression_opts': 1, 'shuffle': True} if value.ndim else {}
        group.create_dataset(key, shape=(0,) + value.shape, maxshape=(None,) + value.shape,
                             chunks=(1,) + value.shape, dtype=value.dtype, **compression)
    ds = group[key]; ds.resize(ds.shape[0] + 1, axis=0); ds[-1] = value


def snapshot(env):
    sim,m,d,gid = T.N.ids(env)
    r = env.env.robots[0]
    names = list(r.gripper.joints)
    selected = [next(n for n in names if n.endswith('_' + s)) for s in ('Left_1_Joint','Right_1_Joint')]
    jids = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in names]
    gp = d.qpos[m.jnt_qposadr[jids]].copy()
    quat = TU.mat2quat(d.site_xmat[gid].reshape(3,3))
    state = np.r_[d.site_xpos[gid], TU.quat2axisangle(quat), [gp[names.index(n)] for n in selected]]
    return {'state': state.astype(np.float32), 'qpos':d.qpos.copy(), 'qvel':d.qvel.copy(),
            'qacc_warmstart':d.qacc_warmstart.copy(), 'ctrl':d.ctrl.copy(),
            'gripper_qpos':gp, 'gripper_target':r.gripper.current_action.copy(),
            'time':np.float64(d.time), 'native_success':np.bool_(env.check_success())}


def render(env):
    sim,m,d,gid = T.N.ids(env)
    if sim._render_context_offscreen is None:
        MjRenderContextOffscreen(sim, -1, max_width=256, max_height=256)
    # Same convention and 180-degree preprocessing as examples/libero/main.py.
    convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    return {key:np.ascontiguousarray(sim.render(256,256,camera_name=name)[::convention][::-1,::-1])
            for key,name in [('image','agentview'),('wrist_image','robot0_eye_in_hand')]}


class Recorder:
    def __init__(self, path, probe=False):
        self.file=h5py.File(path,'x');self.path=path;self.env=None;self.phase='preparation';self.teacher=False
        self.probe=probe;self.started=False;self.observations=0;self.count=0;self.wall_start=time.time()
    def obs(self):
        values=snapshot(self.env); before=values['qpos'].copy();values.update(render(self.env))
        assert np.array_equal(before,T.N.ids(self.env)[2].qpos), 'render changed qpos'
        for k,v in values.items():append(self.file.require_group('observations'),k,v)
        self.observations+=1
    def install(self, env, task):
        self.env=env; self.file.attrs['language']=task.language; self.file.attrs['task_name']=task.name
        self.file.attrs['schema']='eco65_zhixing_pilot_v1';self.file.attrs['fps']=20
        self.file.attrs['camera_preprocess']='robosuite IMAGE_CONVENTION then [::-1, ::-1]; RGB uint8 256x256'
        self.file.attrs['image_convention']=macros.IMAGE_CONVENTION
        self.file.attrs['state_fields']=json.dumps(['grip_site_xyz_m','grip_site_axisangle_rad','Left_1_Joint_rad','Right_1_Joint_rad'])
        self.file.attrs['gripper_joint_names']=json.dumps(list(env.env.robots[0].gripper.joints))
        step=env.step; reset=env.reset
        def reset_record():
            obs=reset()
            for k,v in snapshot(env).items():self.file.require_group('reset').create_dataset(k,data=v)
            self.file.attrs['environment_xml']=env.env.model.get_xml()
            return obs
        def step_record(action):
            values=snapshot(env);group=self.file.require_group('preparation' if not self.teacher else 'transitions')
            if self.teacher and not self.started:
                self.obs();self.started=True
            before_time=float(values['time'])
            result=step(action)
            append(group,'actions',np.asarray(action,dtype=np.float64))
            append(group,'time_before',before_time);append(group,'time_after',T.N.ids(env)[2].time)
            append(group,'qpos_before',values['qpos']);append(group,'qpos_after',T.N.ids(env)[2].qpos.copy())
            append(group,'phase',np.bytes_(self.phase.ljust(40)))
            append(group,'reward',float(result[1]));append(group,'environment_done',bool(result[2]))
            if self.teacher:
                self.obs();self.count+=1
            self.file.flush()
            return result
        env.reset=reset_record;env.step=step_record
    def close(self, completed):
        self.file.attrs['teacher_finished']=completed;self.file.attrs['transitions']=self.count
        self.file.attrs['wall_seconds']=time.time()-self.wall_start
        self.file.flush();self.file.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--probe',action='store_true');a=ap.parse_args()
    out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=False)
    rec=Recorder(out/'trajectory.partial.h5',a.probe)
    original_build=T.D.build; original_act=T.S.orig_act
    def build(*args,**kwargs):
        env,task=original_build(*args,**kwargs);rec.install(env,task);return env,task
    def act(ep,action,phase):
        rec.teacher=True;rec.phase=phase
        return original_act(ep,action,phase)
    T.D.build=build;T.S.orig_act=act
    complete=False
    try:
        if a.probe:
            os.environ['EXTRA_JOINT_PD_PROFILE']='legacy';os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
            env,task=build('libero_object',0,seed=a.seed);env.reset()
            for _ in range(3):env.step(np.r_[np.zeros(6),-1.])
            rec.teacher=True;rec.phase='probe'
            for _ in range(12):env.step(np.r_[np.zeros(6),-1.])
            env.close()
        else:
            sys.argv=['real_object_teacher_r20.py','--case','can','--seed',str(a.seed),'--out',str(out/'teacher')]
            T.main()
        complete=True
    finally:
        rec.close(complete)
    os.rename(out/'trajectory.partial.h5',out/'trajectory.h5')
    dump(out/'capture.json',{'seed':a.seed,'teacher_finished':complete,'transitions':rec.count,
          'observations':rec.observations,'probe':a.probe,'process_exit_pending':True,
          'note':'Data release additionally requires real process exit, teacher numeric audit and replay.'})

if __name__=='__main__':main()
