"""Native reset, image-language policy rollout, observation-only audits and dual-view MP4."""
import argparse
import base64
import collections
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback

import h5py
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite import macros
from robosuite.utils import transform_utils as TU
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
from robosuite.robots.robot import Robot
import urllib.request

V = pathlib.Path(__file__).resolve().parents[1]
ROOT = V.parents[1]
sys.path.insert(0, str(ROOT / 'versions/collection_20261005_r4/scripts'))
from initialization import Initialization
from geometry import apply_offset, Geometry
sys.path.insert(0, str(ROOT / 'versions/collection_20261005_r4/vendor'))
from n4_collect import quiet_hold


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--condition', required=True)
    p.add_argument('--offset', type=float, choices=[0., .05, .10], required=True)
    p.add_argument('--task', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=pathlib.Path, required=True)
    p.add_argument('--port', type=int, default=8872)
    p.add_argument('--video', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--preparation-only', action='store_true')
    a = p.parse_args()
    assert not a.preparation_only or a.smoke
    protocol = json.loads((V / 'configs/protocol.json').read_text())
    c = protocol['conditions'][a.condition]
    reserved = json.loads((ROOT / 'versions/rollout_20261008_r1/configs/test_seeds.json').read_text())['tasks']
    assert a.seed not in sum(reserved.values(), [])
    assert a.seed in protocol['development_seeds'][str(a.task)]
    apply_offset(a.offset)
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    task = benchmark.get_benchmark_dict()['libero_spatial']().get_task(a.task)
    bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    os.environ['EXTRA_JOINT_PD_PROFILE'] = 'legacy'
    os.environ['EXTRA_JOINT_PD_KP'] = ''
    os.environ['ZHIXING_GRIPPER_SPEED'] = '0.01'
    if c['robot'] == 'Panda':
        # Zero initialization noise still consumes one Gaussian per arm joint.
        # Align only that RNG consumption with canonical six-joint ECO65, so
        # the unchanged native object sampler receives the SAME random stream.
        original_robot_reset = Robot.reset
        def paired_robot_reset(self, deterministic=False):
            assert self.initialization_noise['magnitude'] == 0
            rng_before = np.random.get_state()
            value = original_robot_reset(self,deterministic=deterministic)
            if not deterministic:
                np.random.set_state(rng_before)
                np.random.randn(6)
            return value
        Robot.reset = paired_robot_reset
    env = ControlEnv(bddl_file_name=bddl, robots=[c['robot']], gripper_types=c['gripper'],
                     use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    env.seed(a.seed)
    result = {'condition': a.condition, 'task_id': a.task, 'seed': a.seed,
              'task_name': task.name, 'prompt': task.language, 'trained_task': a.task in [0,2,3,5,6],
              'smoke': a.smoke, 'primary_success': False, 'native_success_220': False,
              'native_success_extended': False, 'primary_budget': 220,
              'extended_budget': 1000,
              'base_translation_x_m': a.offset,
              'scope': 'paired validation development diagnostic; not held-out adaptation acceptance',
              'initialization_pass': False, 'failure': None, 'video': None,
              'physical_audit_scope': 'native initialization gate; execution contacts observed each physics substep; not full r6 teacher trajectory certification'}
    result['panda_zero_noise_rng_alignment'] = c['robot'] == 'Panda'
    trace = h5py.File(a.out / 'trajectory.h5', 'x')
    encoder = None
    frames = 0
    rows = []
    phase = 'preparation'
    try:
        env.reset()
        m, d = env.sim.model._model, env.sim.data._data
        robot = env.env.robots[0]
        gid = next(i for i in range(m.nsite) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_SITE,i) or '').endswith('grip_site'))
        joints = list(robot.gripper.joints)
        if c['gripper'] == 'ZhixingGripper':
            selected = [next(n for n in joints if n.endswith('_'+s)) for s in ('Left_1_Joint','Right_1_Joint')]
        else:
            selected = joints[:2]
        jids = [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,n) for n in selected]
        init = Initialization(env)
        geometry = Geometry(env, gid, init)
        result['reset_geometry'] = geometry.read()
        trace.attrs['environment_xml'] = env.env.model.get_xml()
        trace.attrs['camera_preprocess'] = 'IMAGE_CONVENTION then [::-1,::-1]; RGB256'
        trace.attrs['state_gripper_joints'] = json.dumps(selected)
        trace.attrs['protocol_sha256'] = hashlib.sha256((V/'configs/protocol.json').read_bytes()).hexdigest()
        trace.create_dataset('reset_qpos', data=d.qpos)
        trace.create_dataset('reset_qvel', data=d.qvel)
        reset_objects = {name: {'position': d.xpos[bid].tolist(), 'quaternion': d.xquat[bid].tolist()}
                         for name,bid in init.body_ids.items()}
        result['reset_objects'] = reset_objects
        result['reset_object_hash'] = digest({k: {field: np.round(val,9).tolist() for field,val in v.items()} for k,v in reset_objects.items()})
        def views():
            before = d.qpos.copy()
            if env.sim._render_context_offscreen is None:
                MjRenderContextOffscreen(env.sim,-1,max_width=256,max_height=256)
            conv = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
            value = {k:np.ascontiguousarray(env.sim.render(256,256,camera_name=name)[::conv][::-1,::-1])
                     for k,name in [('image','agentview'),('wrist_image','robot0_eye_in_hand')]}
            assert np.array_equal(before,d.qpos), 'render mutated qpos'
            return value

        force = np.zeros(6)
        object_names = set(env.env.objects_dict)
        relevant_entities = object_names | {'robot'}
        execution = {'max_cross_entity_penetration_mm': 0., 'max_robot_object_penetration_mm': 0.,
                     'max_normal_N': 0., 'worst_geoms': [], 'physics_substeps': 0}
        native_step = env.sim.step
        def watched_step():
            native_step()
            if phase == 'preparation':
                init.substep()
            else:
                execution['physics_substeps'] += 1
                for index, contact in enumerate(d.contact):
                    ia,ib = int(contact.geom1),int(contact.geom2)
                    la,lb = init.labels[ia],init.labels[ib]
                    if la == lb or not ({la,lb} & relevant_entities):
                        continue
                    depth = max(0.,-float(contact.dist))*1000
                    mujoco.mj_contactForce(m,d,index,force)
                    execution['max_normal_N'] = max(execution['max_normal_N'], float(force[0]))
                    if depth > execution['max_cross_entity_penetration_mm']:
                        execution['max_cross_entity_penetration_mm'] = depth
                        execution['worst_geoms'] = [init.geom_names[ia],init.geom_names[ib]]
                    if 'robot' in (la,lb) and ({la,lb} & object_names):
                        execution['max_robot_object_penetration_mm'] = max(execution['max_robot_object_penetration_mm'],depth)
        env.sim.step = watched_step
        prep = []
        original_control_step = env.step
        def prepared_step(action):
            value = original_control_step(action)
            prep.append(np.asarray(action).copy())
            # Frozen replay establishes the context immediately after step1.
            # Only context construction calls sim.forward(); later render calls
            # update the camera scene. Avoid rendering unused preparation frames.
            if len(prep)==1:views()
            init.control_step()
            return value
        env.step = prepared_step
        # Reuse the exact frozen helper, solely for proprioceptive settling.
        # No privileged teacher task planning or object-conditioned action is run.
        quiet_hold(env,120,-1)
        env.step = original_control_step
        initialization = init.result()
        (a.out/'initialization.json').write_text(json.dumps(initialization,indent=2))
        result['initialization_pass'] = initialization['pass']
        trace.create_dataset('preparation_actions',data=prep)
        trace.create_dataset('settled_qpos',data=d.qpos)
        result['settled_objects'] = {name:d.xpos[bid].tolist() for name,bid in init.body_ids.items()}
        os.environ['EXTRA_JOINT_PD_PROFILE'] = c['controller_profile']
        phase = 'policy'
        result['settled_geometry'] = geometry.read()
        limits = robot.controller
        result['controller'] = {'input_max': np.asarray(limits.input_max).tolist(),
                                'output_max': np.asarray(limits.output_max).tolist(),
                                'profile': c['controller_profile'], 'control_frequency': env.env.control_freq}
        assert np.allclose(limits.output_max,[.05,.05,.05,.5,.5,.5])
        assert env.env.control_freq == 20

        def state():
            quat = TU.mat2quat(d.site_xmat[gid].reshape(3,3))
            return np.r_[d.site_xpos[gid],TU.quat2axisangle(quat),d.qpos[m.jnt_qposadr[jids]]].astype(np.float32)

        mp4 = a.out/'rollout.mp4'
        if a.video:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            encoder = subprocess.Popen([ffmpeg,'-nostdin','-v','error','-f','rawvideo','-pixel_format','rgb24',
                                        '-video_size','512x288','-framerate','20','-i','pipe:0','-an',
                                        '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',str(mp4)],stdin=subprocess.PIPE)

        def video_frame(rgb,step,terminal=False):
            nonlocal frames
            if not encoder:
                return
            img = Image.new('RGB',(512,288),'#171c25')
            draw = ImageDraw.Draw(img)
            draw.text((5,3),f'base+{round(a.offset*100)}cm task{a.task:02d} seed{a.seed} step{step}',fill='white')
            if not result['initialization_pass']:
                caption = 'INITIALIZATION REJECTED | retained as failure'
            elif terminal:
                caption = 'PRIMARY SUCCESS' if result['primary_success'] else 'PRIMARY FAILURE: '+str(result['failure'])
            else:
                caption = 'agentview / eye_in_hand | budgets220/650/1000'
            draw.text((5,17),caption,fill='#d1d9ea')
            img.paste(Image.fromarray(rgb['image']),(0,32))
            img.paste(Image.fromarray(rgb['wrist_image']),(256,32))
            encoder.stdin.write(np.asarray(img).tobytes())
            frames += 1

        queue = collections.deque()
        query = 0
        rgb = views()
        for k,value in rgb.items():trace.create_dataset('first_policy_input/'+k,data=value,compression='gzip',compression_opts=1)
        Image.fromarray(np.concatenate([rgb['image'],rgb['wrist_image']],axis=1)).save(a.out/'initial_views.jpg')
        video_frame(rgb,0)
        if not initialization['pass']:
            result['failure'] = 'invalid_native_initialization'
            # A static second makes an invalid-initialization recording playable.
            for _ in range(19): video_frame(rgb,0)
        else:
            budget = 0 if a.preparation_only else 10 if a.smoke else result['extended_budget']
            for step in range(budget):
                if not queue:
                    if step and not a.video:rgb = views()
                    payload = {'seed':a.seed,'query':query,'state':state().tolist(),'prompt':task.language,
                               **{k:base64.b64encode(v.tobytes()).decode() for k,v in rgb.items()}}
                    request = urllib.request.Request(f'http://127.0.0.1:{a.port}/infer',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                    t = time.monotonic()
                    response = json.loads(urllib.request.urlopen(request,timeout=300).read())
                    raw = np.asarray(response['actions'],dtype=np.float64)
                    assert response['kind'] == c['model'] and raw.shape == (10,7) and np.isfinite(raw).all()
                    queue.extend(raw[:5])
                    group = trace.create_group(f'queries/{query:04d}')
                    group.attrs['step'] = step
                    group.attrs['noise_sha256'] = response['noise_sha256']
                    group.attrs['latency_seconds'] = time.monotonic()-t
                    group.create_dataset('state',data=state())
                    group.create_dataset('actions',data=raw)
                    for k,v in rgb.items(): group.create_dataset(k,data=v,compression='gzip',compression_opts=1)
                    query += 1
                raw = queue.popleft()
                action = np.clip(raw,-1.,1.)
                before = d.qpos.copy()
                env.env.done = False
                _,reward,done,info = env.step(action)
                native = bool(env.check_success())
                warnings = np.asarray(d.warning.number).tolist()
                finite = bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
                row = {'step':step+1,'time':float(d.time),'state':state().tolist(),
                       'raw_action':raw.tolist(),'executed_action':action.tolist(),
                       'geometry':geometry.read(),
                       'qpos_before':before.tolist(),'qpos_after':d.qpos.tolist(),
                       'qvel_after':d.qvel.tolist(),'native_success':native,'finite':finite,
                       'physics_warnings':warnings}
                rows.append(row)
                if a.video:
                    rgb = views()
                    video_frame(rgb,step+1)
                if native:
                    result['native_success_extended'] = True
                    result['first_success_step'] = step+1
                    if step+1 <= 220:
                        result['native_success_220'] = True
                        result['primary_success'] = finite and not any(warnings)
                    result['failure'] = None if result['primary_success'] else 'success_only_after_primary_budget'
                    break
                if not finite or any(warnings):
                    result['failure'] = 'physics_nonfinite_or_warning'
                    break
                if (step+1)%50 == 0:
                    print('POLICY STEP',a.condition,a.task,a.seed,step+1,flush=True)
            else:
                result['failure'] = 'technical_preparation_only' if a.preparation_only else 'time_budget_exhausted'
        result['steps'] = len(rows)
        result['queries'] = query
        result['clipped_action_frames'] = sum(not np.array_equal(x['raw_action'],x['executed_action']) for x in rows)
        result['execution_contacts'] = execution
        result['physics_warnings'] = np.asarray(d.warning.number).tolist()
        # One-second terminal caption, held on the actual last recorded state.
        if encoder:
            for _ in range(20):video_frame(rgb,len(rows),terminal=True)
        result['video_frames'] = frames
        if encoder:
            encoder.stdin.close()
            assert encoder.wait(timeout=60) == 0, 'video encoding failed'
            encoder = None
            result['video'] = 'rollout.mp4'
            result['video_sha256'] = hashlib.sha256(mp4.read_bytes()).hexdigest()
        final = views()
        Image.fromarray(np.concatenate([final['image'],final['wrist_image']],axis=1)).save(a.out/'final_views.jpg')
    except Exception:
        result['failure'] = 'execution_error'
        result['error'] = traceback.format_exc()
        raise
    finally:
        if encoder:
            encoder.stdin.close()
            result['encoder_exit'] = encoder.wait(timeout=60)
        trace.close()
        (a.out/'steps.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
        result['seconds'] = time.monotonic()-start
        (a.out/'result.json').write_text(json.dumps(result,indent=2))
        env.close()
    print('ROLLOUT RESULT',json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
