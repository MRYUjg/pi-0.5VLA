"""Independent Panda-only developer trajectories; original collection files stay read-only."""
import argparse
import base64
import collections
import hashlib
import json
import pathlib
import time
import urllib.request

import numpy as np
from libero.libero import benchmark,get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from robosuite.utils.transform_utils import quat2axisangle

p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True);p.add_argument('--task',type=int,required=True)
p.add_argument('--target',type=int,default=256);p.add_argument('--port',type=int,default=18877);a=p.parse_args()
a.out.mkdir(parents=True,exist_ok=False)
suite=benchmark.get_benchmark_dict()['libero_spatial']();task=suite.get_task(a.task)
states=suite.get_task_init_states(a.task)
assert len(states)>=50
formal_indices=list(range(len(states)-20,len(states)))
developer_indices=list(range(3,min(73,len(states)-20)))
rows=[];episodes=[];begin=time.time()
# Reserve the last 20 official scenes for a future matched Panda formal test.
# ECO65 formal seeds are separately frozen by stage 5 and never sampled here.
registration={'task':a.task,'developer_indices':developer_indices,'reserved_formal_indices':formal_indices,
              'seed_base':920000,'observation_stride_steps':2,'target_observations':a.target,
              'robot':'Panda','gripper':'PandaGripper','policy_denoise_steps':10,'executed_steps_before_replan':5,
              'max_policy_steps':220,'wait_steps':10,'retains_failed_trajectories':True,'purpose':'calibration_only'}
(a.out/'registration.json').write_text(json.dumps(registration,indent=2))


def record(obs):
    state=np.concatenate([obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),obs['robot0_gripper_qpos']]).astype(np.float32)
    views={k:image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs[source][::-1,::-1]),224,224))
           for k,source in [('image','agentview_image'),('wrist_image','robot0_eye_in_hand_image')]}
    return {**views,'state':state,'prompt':task.language}


try:
    for index in developer_indices:
        if len(rows)>=a.target:
            break
        seed=920000+a.task*100+index
        np.random.seed(seed)
        env=OffScreenRenderEnv(bddl_file_name=str(pathlib.Path(get_libero_path('bddl_files'))/task.problem_folder/task.bddl_file),
            robots=['Panda'],gripper_types='PandaGripper',camera_heights=256,camera_widths=256)
        try:
            env.seed(seed);env.reset();obs=env.set_init_state(states[index])
            assert env.env.control_freq==20
            assert type(env.env.robots[0].gripper).__name__=='PandaGripper'
            for _ in range(10):obs,_,_,_=env.step([0.]*6+[-1.])
            assert not env.env._check_success(),'initially successful scene'
            queue=collections.deque();steps=[];success=False;first_row=len(rows)
            for step in range(220):
                rec=record(obs)
                if step%2==0:
                    filename=f'obs_{len(rows):05d}.npz'
                    np.savez_compressed(a.out/filename,**rec)
                    rgb_sha=hashlib.sha256(rec['image'].tobytes()+rec['wrist_image'].tobytes()).hexdigest()
                    rows.append({'file':filename,'task':a.task,'index':index,'seed':seed,'step':step,
                        'phase':'early' if step<20 else ('middle' if step<60 else 'late'),
                        'image_sha256':rgb_sha,'noise_seed':seed*1000+step,'policy_replan':step%5==0})
                if not queue:
                    obj={k:base64.b64encode(rec[k].tobytes()).decode() for k in ['image','wrist_image']}
                    obj.update(state=rec['state'].tolist(),prompt=rec['prompt'],noise_seed=seed*1000+step//5)
                    req=urllib.request.Request(f'http://127.0.0.1:{a.port}/infer',data=json.dumps(obj).encode(),headers={'Content-Type':'application/json'})
                    reply=json.loads(urllib.request.urlopen(req,timeout=300).read())
                    actions=np.asarray(reply['actions'],np.float32)
                    assert actions.shape==(10,7) and np.isfinite(actions).all()
                    queue.extend(actions[:5])
                action=queue.popleft();executed=np.clip(action,-1,1)
                obs,_,_,_=env.step(executed)
                success=bool(env.env._check_success());steps.append({'step':step,'raw_action':action.tolist(),'success':success})
                if success:break
            ep={'task':a.task,'init_index':index,'seed':seed,'success':success,'steps':len(steps),
                'initial_sha256':hashlib.sha256(np.asarray(states[index]).tobytes()).hexdigest()}
            for row in rows[first_row:]:
                fraction=row['step']/max(1,len(steps))
                row['phase']='early' if fraction<1/3 else ('middle' if fraction<2/3 else 'late')
                row['phase_definition']='relative thirds of complete episode'
            episodes.append(ep)
            (a.out/f'episode_i{index:02d}.json').write_text(json.dumps({'episode':ep,'steps':steps},indent=2))
            print('EPISODE',a.task,index,success,len(steps),'OBS',len(rows),flush=True)
        finally:
            env.close()
    assert len(rows)>=a.target,(a.task,len(rows))
    (a.out/'observations.json').write_text(json.dumps(rows,indent=2))
    (a.out/'summary.json').write_text(json.dumps({'task':a.task,'observations':len(rows),'episodes':episodes,
        'seconds':time.time()-begin,'calibration_only':True,'closed_loop_acceptance':False},indent=2))
except Exception as exc:
    (a.out/'failure.json').write_text(json.dumps({'error':repr(exc),'observations_saved':len(rows)}));raise
