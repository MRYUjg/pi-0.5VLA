"""Frozen capture adapter plus an observation-only initialization auditor."""
import argparse
import json
import os
import pathlib
import sys
import h5py
import numpy as np
import capture_core as core
from initialization import Initialization


class AuditedRecorder(core.Recorder):
    def install(self, env, task):
        super().install(env, task)
        reset = env.reset
        step = env.step
        def observed_reset():
            result = reset()
            self.initialization = Initialization(env)
            sim_step = env.sim.step
            def observed_substep():
                sim_step()
                if not self.teacher:
                    self.initialization.substep()
            env.sim.step = observed_substep
            return result
        def observed_step(action):
            result = step(action)
            if not self.teacher:
                self.initialization.control_step()
            return result
        env.reset = observed_reset
        env.step = observed_step

    def close(self, completed):
        if hasattr(self, 'initialization'):
            report = self.initialization.result()
            core.dump(pathlib.Path(self.path).parent/'initialization.json', report)
        super().close(completed)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    scene=core.scenes.spec_for(a.seed)
    assert scene['split'] in ('reference','regression','train','validation'), 'test collection prohibited'
    core.Recorder=AuditedRecorder
    if not a.preflight:
        sys.argv=[__file__,'--seed',str(a.seed),'--out',str(a.out)]
        return core.main()
    a.out.mkdir(parents=True,exist_ok=False)
    os.environ['EXTRA_JOINT_PD_PROFILE']='legacy'
    os.environ['EXTRA_JOINT_PD_KP']=''
    os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
    core.T.D.calibrate()
    env,task=core.T.D.build('libero_spatial',scene['task_id'],seed=a.seed)
    core.T.EX.install_generic_pad_geoms(print)
    core.T.EX.install_generic_state_gripper(env,print)
    rec=AuditedRecorder(a.out/'preparation.h5')
    rec.install(env,task)
    completed=False
    try:
        env.reset()
        core.T.N4.quiet_hold(env,120,-1)
        completed=True
    finally:
        rec.close(completed)
        env.close()
    report={'seed':a.seed,'scene':scene,'preparation_completed':completed}
    if scene['split']=='reference':
        old=pathlib.Path('/app/agent-work-space/4-multi-task-collection/runs/spatial_pilot_20261004_r1')/f'task{scene["task_id"]:02d}_s{a.seed}'/'trajectory.h5'
        with h5py.File(old,'r') as f,h5py.File(a.out/'preparation.h5','r') as g:
            errors={k:float(np.max(np.abs(f['preparation/'+k][...]-g['preparation/'+k][...]))) for k in ('actions','qpos_before','qpos_after','time_before','time_after')}
            for k in ('image','wrist_image'):
                errors[k]=max(int(np.max(np.abs(f['preparation_images/'+k][i].astype(int)-g['preparation_images/'+k][i].astype(int)))) for i in (0,119))
        report.update(reference_errors=errors,reference_unchanged=all(v==0 for v in errors.values()))
        control=pathlib.Path('/app/agent-work-space/5-final-train/runs/reference_baseline_20261005_r1')/f'task{scene["task_id"]:02d}_s{a.seed}'/'preparation.h5'
        with h5py.File(control,'r') as f,h5py.File(a.out/'preparation.h5','r') as g:
            same={k:float(np.max(np.abs(f['preparation/'+k][...]-g['preparation/'+k][...]))) for k in ('actions','qpos_before','qpos_after','time_before','time_after')}
            for k in ('image','wrist_image'):
                same[k]=max(int(np.max(np.abs(f['preparation_images/'+k][i].astype(int)-g['preparation_images/'+k][i].astype(int)))) for i in (0,119))
        report.update(same_host_control_errors=same,same_host_observer_unchanged=all(x==0 for x in same.values()))
    core.dump(a.out/'preflight.json',report)
    print(json.dumps(report,ensure_ascii=False),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
