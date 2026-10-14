"""Matched Panda episodes; raw chunks, executed controls and dual-camera video."""
import argparse
import base64
import collections
import hashlib
import json
import pathlib
import time
import traceback
import urllib.request

import imageio.v2 as imageio
import numpy as np
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from robosuite.utils.transform_utils import quat2axisangle


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def jsonable(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    return repr(x)


def robot_metadata(env):
    robot=env.env.robots[0];c=robot.controller
    keys=['config','input_min','input_max','output_min','output_max','kp','damping_ratio',
        'control_delta','position_limits','orientation_limits','policy_freq','joint_indexes']
    return {'robot':type(robot.robot_model).__name__,'gripper':type(robot.gripper).__name__,
        'controller':type(c).__name__,'control_freq':env.env.control_freq,
        'controller_parameters':{k:getattr(c,k) for k in keys if hasattr(c,k)}}


def view(obs):
    return np.ascontiguousarray(np.concatenate([obs['agentview_image'][::-1,::-1],
        obs['robot0_eye_in_hand_image'][::-1,::-1]],axis=1))


def record(obs,language):
    state=np.concatenate([obs['robot0_eef_pos'],quat2axisangle(obs['robot0_eef_quat'].copy()),
        obs['robot0_gripper_qpos']]).astype(np.float32)
    views={k:image_tools.convert_to_uint8(image_tools.resize_with_pad(
        np.ascontiguousarray(obs[source][::-1,::-1]),224,224)) for k,source in [
        ('image','agentview_image'),('wrist_image','robot0_eye_in_hand_image')]}
    return {**views,'state':state,'prompt':language}


def episode(out,row,scene_folder,variant,port,reference_controller):
    seed=row['seed'];initial=scene_folder/row['initial_file']
    assert sha(initial)==row['initial_file_sha256'] and sha(row['bddl'])==row['bddl_sha256']
    np.random.seed(seed)
    env=OffScreenRenderEnv(bddl_file_name=row['bddl'],robots=['Panda'],gripper_types='PandaGripper',
        camera_heights=256,camera_widths=256)
    tag=f'init{row["init_index"]:02d}';video=out/(tag+'.mp4');steps=[];chunks=[];frames=0
    writer=None;start=time.monotonic()
    try:
        env.seed(seed);env.reset();obs=env.set_init_state(np.load(initial,allow_pickle=False))
        camera_started=time.perf_counter()
        for _ in range(10):
            camera_started=time.perf_counter();obs,_,_,_=env.step([0.]*6+[-1.])
        robot=robot_metadata(env)
        controller=json.loads(json.dumps(robot,default=jsonable))
        assert controller==reference_controller,(controller,reference_controller)
        assert not env.env._check_success(),'initially successful episode'
        state=np.asarray(env.env.sim.get_state().flatten())
        pose=record(obs,row['language'])
        initialization={'simulation_state_sha256':hashlib.sha256(state.tobytes()).hexdigest(),
            'initial_observation_sha256':hashlib.sha256(pose['image'].tobytes()+pose['wrist_image'].tobytes()+pose['state'].tobytes()).hexdigest(),
            'xml_sha256':hashlib.sha256(env.env.sim.model.get_xml().encode()).hexdigest(),
            'controller_sha256':hashlib.sha256(json.dumps(controller,sort_keys=True).encode()).hexdigest(),
            'controller':controller,'state8':pose['state'].tolist()}
        writer=imageio.get_writer(video,fps=20,codec='libx264',quality=7,ffmpeg_log_level='error',ffmpeg_params=['-threads','1'])
        writer.append_data(view(obs));frames+=1
        queue=collections.deque();success=False
        for step in range(220):
            dispatch_timing=None
            if not queue:
                prep_start=time.perf_counter();raw=record(obs,row['language'])
                obj={k:base64.b64encode(raw[k].tobytes()).decode() for k in ['image','wrist_image']}
                noise_seed=seed*1000+step//5
                obj.update(state=raw['state'].tolist(),prompt=raw['prompt'],noise_seed=noise_seed,
                    client_id=f'closedloop_{variant}_task{row["task"]}_init{row["init_index"]}_step{step}')
                payload=json.dumps(obj).encode();sent=time.perf_counter()
                req=urllib.request.Request(f'http://127.0.0.1:{port}/infer',data=payload,headers={'Content-Type':'application/json'})
                reply=json.loads(urllib.request.urlopen(req,timeout=300).read())
                received=time.perf_counter();actions=np.asarray(reply['actions'],np.float32)
                assert reply['variant']==variant and actions.shape==(10,7) and np.isfinite(actions).all()
                dispatch_timing={'image_state_preparation_and_encoding_ms':(sent-prep_start)*1000,
                    'client_round_trip_ms':(received-sent)*1000,
                    'observation_ready_to_action_ms':(received-prep_start)*1000,
                    'camera_step_to_action_ms':(received-camera_started)*1000,'server':reply['timing']}
                chunks.append({'step':step,'noise_seed':noise_seed,'raw_actions':actions.tolist(),
                    'state8':raw['state'].tolist(),'timing':dispatch_timing,'server_request':reply['request']})
                queue.extend(actions[:5])
            action=queue.popleft();executed=np.clip(action,-1,1)
            camera_started=time.perf_counter();obs,_,done,_=env.step(executed)
            camera_ended=time.perf_counter()
            success=bool(env.env._check_success())
            writer.append_data(view(obs));frames+=1
            steps.append({'step':step,'raw_action':action.tolist(),'executed_action':executed.tolist(),
                'clipped_dimensions':(np.abs(action)>1).tolist(),'success':success,'env_done':bool(done),
                'environment_step_and_render_ms':(camera_ended-camera_started)*1000,
                'eef_pos':obs['robot0_eef_pos'].tolist(),'gripper_qpos':obs['robot0_gripper_qpos'].tolist()})
            if success:break
        writer.close();writer=None
        reader=imageio.get_reader(video);assert reader.get_data(0).shape[:2]==(256,512);reader.close()
        metadata={'variant':variant,'task':row['task'],'init_index':row['init_index'],'seed':seed,
            'initial_state_sha256':row['initial_state_sha256'],'success':success,'steps':len(steps),
            'requests':len(chunks),'seconds':time.monotonic()-start,'initialization':initialization,
            'video':video.name,'video_frames':frames,'video_sha256':sha(video),
            'video_fps_is_simulated_control_rate':20,'formal_acceptance':False}
        with (out/(tag+'.json')).open('x') as f:json.dump({'episode':metadata,'chunks':chunks,'steps':steps},f,indent=2)
        print('EPISODE',variant,row['task'],row['init_index'],success,len(steps),flush=True)
        return metadata
    except Exception:
        with (out/(tag+'_failure.json')).open('x') as f:json.dump({'traceback':traceback.format_exc(),
            'partial_steps':steps,'partial_chunks':chunks,'video_frames':frames},f,indent=2)
        raise
    finally:
        if writer is not None:writer.close()
        env.close()


def main(out,scenes,task,variant,port):
    out.mkdir(parents=True,exist_ok=False)
    rows=[r for r in json.loads(scenes.read_text()) if r['task']==task]
    assert len(rows)==3
    reference=json.loads((scenes.parent/'controller.json').read_text());reference.pop('XML_sha256')
    episodes=[episode(out,r,scenes.parent,variant,port,reference) for r in rows]
    with (out/'summary.json').open('x') as f:json.dump({'task':task,'variant':variant,'episodes':episodes,
        'successes':sum(r['success'] for r in episodes),'trials':3,'all_videos_decoded':True},f,indent=2)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True)
    p.add_argument('--scenes',type=pathlib.Path,required=True);p.add_argument('--task',type=int,required=True)
    p.add_argument('--variant',required=True);p.add_argument('--port',type=int,default=18879)
    a=p.parse_args();main(a.out,a.scenes,a.task,a.variant,a.port)
