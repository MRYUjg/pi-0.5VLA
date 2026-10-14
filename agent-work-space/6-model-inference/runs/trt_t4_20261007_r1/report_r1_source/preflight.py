"""CPU-only registration of held-out development scenes and video rendering."""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import pathlib
import platform
import time

import imageio.v2 as imageio
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return repr(x)


def main(out):
    out.mkdir(parents=True, exist_ok=False)
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    registrations = []
    for task_id in range(10):
        task = suite.get_task(task_id)
        states = suite.get_task_init_states(task_id)
        assert len(states) >= 50
        bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
        old = pathlib.Path('/work/runs/trt_t0_t2_20261007_r1/panda_calibration_r1') / f'task{task_id:02d}/registration.json'
        old_registration = json.loads(old.read_text())
        assert not set([0,1,2]) & set(old_registration['developer_indices'])
        assert not set([0,1,2]) & set(old_registration['reserved_formal_indices'])
        for index in [0,1,2]:
            initial = np.asarray(states[index])
            filename = f'task{task_id:02d}_init{index:02d}.npy'
            np.save(out / filename, initial, allow_pickle=False)
            registrations.append({'task': task_id, 'init_index': index, 'seed': 940000 + task_id*100 + index,
                'initial_file': filename, 'initial_state_sha256': hashlib.sha256(initial.tobytes()).hexdigest(),
                'initial_file_sha256': sha(out / filename), 'shape': list(initial.shape), 'dtype': str(initial.dtype),
                'bddl': str(bddl), 'bddl_sha256': sha(bddl), 'language': task.language,
                'reserved_formal_indices': old_registration['reserved_formal_indices'],
                'calibration_registration_sha256': sha(old)})
    (out/'scenes.json').write_text(json.dumps(registrations,ensure_ascii=False,indent=2)+'\n')
    r = registrations[0]
    np.random.seed(r['seed'])
    env = OffScreenRenderEnv(bddl_file_name=r['bddl'], robots=['Panda'], gripper_types='PandaGripper',
        camera_heights=256, camera_widths=256)
    try:
        env.seed(r['seed']); env.reset()
        obs = env.set_init_state(np.load(out / r['initial_file']))
        for _ in range(10):
            obs, _, _, _ = env.step([0.]*6+[-1.])
        robot = env.env.robots[0]
        assert env.env.control_freq == 20
        assert type(robot.gripper).__name__ == 'PandaGripper'
        assert not env.env._check_success()
        controller = robot.controller
        keys = ['config','input_min','input_max','output_min','output_max','kp','damping_ratio',
                'control_delta','position_limits','orientation_limits','policy_freq','joint_indexes']
        cfg = {k: getattr(controller,k) for k in keys if hasattr(controller,k)}
        metadata = {'robot': type(robot.robot_model).__name__, 'gripper': type(robot.gripper).__name__,
            'controller': type(controller).__name__, 'control_freq': env.env.control_freq,
            'controller_parameters': cfg, 'XML_sha256': hashlib.sha256(env.env.sim.model.get_xml().encode()).hexdigest()}
        (out/'controller.json').write_text(json.dumps(metadata,indent=2,default=jsonable)+'\n')
        modules = [inspect.getmodule(type(env)), inspect.getmodule(type(controller)), inspect.getmodule(type(robot.gripper))]
        code = {m.__file__: sha(m.__file__) for m in modules if m is not None}
        probe = out / 'render_probe.mp4'
        with imageio.get_writer(probe, fps=20, codec='libx264', quality=7, ffmpeg_log_level='error') as writer:
            for _ in range(5):
                frame = np.concatenate([obs['agentview_image'][::-1,::-1],obs['robot0_eye_in_hand_image'][::-1,::-1]],axis=1)
                writer.append_data(np.ascontiguousarray(frame))
                obs, _, _, _ = env.step([0.]*6+[-1.])
        assert probe.stat().st_size > 1000
        reader = imageio.get_reader(probe)
        assert reader.get_data(0).shape[:2] == (256,512)
        reader.close()
    finally:
        env.close()
    import libero.libero.envs as envs
    import robosuite
    import mujoco
    roots = [pathlib.Path(envs.__file__).parent, pathlib.Path(robosuite.__file__).parent]
    code.update({str(p): sha(p) for root in roots for p in root.rglob('*.py') if p.is_file()})
    (out/'environment.json').write_text(json.dumps({'python': platform.python_version(),
        'mujoco': mujoco.__version__, 'robosuite': robosuite.__version__, 'source_files': code,
        'packages': dict(sorted((d.metadata['Name'],d.version) for d in importlib.metadata.distributions())),
        'renderer': 'OSMesa, LP_NUM_THREADS=1', 'controller_profile': 'legacy',
        'formal_test_scenes_used': False},indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps({'passed': True, 'scenes': 30, 'video_encode_and_decode': True,
        'robot': 'Panda', 'gripper': 'PandaGripper', 'scene_indices': [0,1,2]},indent=2)+'\n')
    print('PREFLIGHT PASS 30 HELD-OUT DEVELOPMENT SCENES AND VIDEO',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--out',type=pathlib.Path,required=True)
    main(p.parse_args().out)
