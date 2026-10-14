"""Recreate reset and replay executed actions; never inject saved qpos."""
import argparse,json,os,pathlib
import h5py,numpy as np
import sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'spatial_pilot_20261004_r1/scripts'))
from collect import T, snapshot, render, dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--episode',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    ep=pathlib.Path(a.episode);cap=json.loads((ep/'capture.json').read_text());out=pathlib.Path(a.out);assert not out.exists()
    os.environ['EXTRA_JOINT_PD_PROFILE']='legacy';os.environ['EXTRA_JOINT_PD_KP']='';os.environ['ZHIXING_GRIPPER_SPEED']='0.01'
    errors=[];max_q=0.;max_v=0.;max_state=0.;max_image=0;max_time=0.;native_mismatch=0
    env,task=T.D.build('libero_spatial',0,seed=cap['seed']);T.EX.install_generic_pad_geoms(print);T.EX.install_generic_state_gripper(env,print)
    env.reset()
    with h5py.File(ep/'trajectory.h5','r') as f:
        reset_err=float(np.max(abs(snapshot(env)['qpos']-f['reset/qpos'][...])));max_q=reset_err
        for prep_index,action in enumerate(f['preparation/actions']):
            env.step(action)
            if prep_index==0:
                # Offscreen context constructor invokes sim.forward(); align with capture.
                for k,v in render(env).items():
                    max_image=max(max_image,int(np.max(abs(v.astype(int)-f['preparation_images/'+k][0].astype(int)))))
        os.environ['EXTRA_JOINT_PD_PROFILE']='eco65_no_extra_pd'
        n=len(f['transitions/actions']);frames={0,n//4,n//2,3*n//4,n}
        # Physics monitor is unnecessary; it only writes zero applied force in this teacher.
        for i in range(n+1):
            s=snapshot(env)
            max_q=max(max_q,float(np.max(abs(s['qpos']-f['observations/qpos'][i]))))
            max_v=max(max_v,float(np.max(abs(s['qvel']-f['observations/qvel'][i]))))
            max_state=max(max_state,float(np.max(abs(s['state']-f['observations/state'][i]))))
            max_time=max(max_time,abs(float(s['time'])-float(f['observations/time'][i])))
            native_mismatch+=int(s['native_success']!=f['observations/native_success'][i])
            if i in frames:
                for k,v in render(env).items():max_image=max(max_image,int(np.max(abs(v.astype(int)-f['observations/'+k][i].astype(int)))))
            if i<n:
                env.env.done=False
                env.step(f['transitions/actions'][i])
        if max_q>1e-8:errors.append('qpos replay drift')
        if max_v>1e-7:errors.append('qvel replay drift')
        if max_state>1e-6:errors.append('state replay drift')
        if max_time>1e-8:errors.append('time replay drift')
        if max_image>0:errors.append('RGB replay mismatch')
        if native_mismatch:errors.append('native success replay mismatch')
    env.close()
    dump(out,{'pass':not errors,'errors':errors,'seed':cap['seed'],'steps':n,'reset_qpos_error':reset_err,'max_qpos_error':max_q,'max_qvel_error':max_v,'max_state_error':max_state,'max_time_error':max_time,'max_rgb_pixel_error':max_image,'native_mismatches':native_mismatch,'rendered_observation_indices':sorted(frames),'method':'reset(seed), replay recorded preparation and teacher actions; initialize renderer after first preparation action exactly as capture; no state injection','renderer_first_preparation_frame_checked':True,'process_exit_pending':True})
    print('REPLAY',cap['seed'],not errors,errors,flush=True)
    return int(bool(errors))
if __name__=='__main__':raise SystemExit(main())
