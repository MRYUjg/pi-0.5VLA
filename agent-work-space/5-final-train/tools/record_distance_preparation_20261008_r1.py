"""Visual reproduce two rejected development preparations, preserving frozen controls."""
import hashlib
import json
import pathlib
import subprocess
import sys

import h5py
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite import macros
from robosuite.utils.binding_utils import MjRenderContextOffscreen
from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'versions/workspace_distance_20261008_r2/scripts'))
from geometry import apply_offset
sys.path.insert(0, str(ROOT / 'versions/collection_20261005_r4/scripts'))
from initialization import Initialization
sys.path.insert(0, str(ROOT / 'versions/collection_20261005_r4/vendor'))
from n4_collect import quiet_hold


def main():
    task, seed, cm = map(int, sys.argv[1:])
    assert (task, seed, cm) in [(2, 630201, 10), (6, 630600, 5)]
    out = ROOT / 'videos/workspace_distance_preparation_20261008_r1' / f'task{task:02d}_s{seed}_base{cm:02d}cm'
    out.mkdir(parents=True, exist_ok=False)
    apply_offset(cm/100)
    os.environ['EXTRA_JOINT_PD_PROFILE'] = 'legacy'
    os.environ['EXTRA_JOINT_PD_KP'] = ''
    os.environ['ZHIXING_GRIPPER_SPEED'] = '0.01'
    item = benchmark.get_benchmark_dict()['libero_spatial']().get_task(task)
    bddl = pathlib.Path(get_libero_path('bddl_files')) / item.problem_folder / item.bddl_file
    env = ControlEnv(bddl_file_name=bddl, robots=['ECO65'], gripper_types='ZhixingGripper',
                     use_camera_obs=False, has_renderer=False, has_offscreen_renderer=False)
    env.seed(seed)
    env.reset()
    init = Initialization(env)
    actions = []
    native_step = env.sim.step
    def watched_step():
        native_step()
        init.substep()
    env.sim.step = watched_step
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = out / 'preparation.mp4'
    encoder = subprocess.Popen([ffmpeg, '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pixel_format', 'rgb24',
                                '-video_size', '512x288', '-framerate', '20', '-i', 'pipe:0', '-an',
                                '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-pix_fmt', 'yuv420p', str(mp4)], stdin=subprocess.PIPE)
    conv = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
    def frame(terminal=False):
        before = env.sim.data.qpos.copy()
        if env.sim._render_context_offscreen is None:
            MjRenderContextOffscreen(env.sim, -1, max_width=256, max_height=256)
        rgb = [np.ascontiguousarray(env.sim.render(256, 256, camera_name=n)[::conv][::-1, ::-1])
               for n in ['agentview', 'robot0_eye_in_hand']]
        assert np.array_equal(before, env.sim.data.qpos)
        image = Image.new('RGB', (512, 288), '#171c25')
        draw = ImageDraw.Draw(image)
        draw.text((5, 3), f'PREPARATION ONLY | base+{cm}cm task{task:02d} seed{seed} step{len(actions)}', fill='white')
        draw.text((5, 17), 'INITIAL GATE REJECTED | no policy actions' if terminal else 'Frozen quiet_hold / audit | real agentview + wrist', fill='#d1d9ea')
        for i, view in enumerate(rgb):
            image.paste(Image.fromarray(view), (i*256, 32))
        encoder.stdin.write(np.asarray(image).tobytes())
    native_control = env.step
    def prepared_step(action):
        result = native_control(action)
        actions.append(np.asarray(action).copy())
        # The first render context is created after step1, exactly as frozen replay.
        frame()
        init.control_step()
        return result
    env.step = prepared_step
    quiet_hold(env, 120, -1)
    assessment = init.result()
    assert not assessment['pass']
    reference = ROOT / 'runs/workspace_distance_20261008_r2/preflight' / f'base_{cm:02d}cm' / f'task{task:02d}_s{seed}'
    with h5py.File(reference / 'trajectory.h5', 'r') as f:
        qerror = float(abs(f['settled_qpos'][...]-env.sim.data.qpos).max())
        action_error = float(abs(f['preparation_actions'][...]-np.asarray(actions)).max())
    assert qerror <= 1e-8, qerror
    for _ in range(20):
        frame(terminal=True)
    encoder.stdin.close()
    assert encoder.wait(timeout=60) == 0
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(mp4))
    decoded = subprocess.run([ffmpeg, '-v', 'error', '-i', str(mp4), '-f', 'null', '-'], capture_output=True)
    assert decoded.returncode == 0 and frames == 140
    record = {'task': task, 'seed': seed, 'base_forward_cm': cm,
              'scope': 'preparation-only visual reproduction of rejected validation layout; not an additional policy test',
              'steps': 120, 'frames': frames, 'seconds': seconds, 'decode_pass': True,
              'sha256': hashlib.sha256(mp4.read_bytes()).hexdigest(),
              'settled_qpos_max_error_to_registered_preflight': qerror,
              'preparation_action_max_error': action_error,
              'initialization': assessment}
    with (out / 'record.json').open('x') as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    env.close()
    print('PREPARATION VIDEO VERIFIED', task, cm, qerror, flush=True)


if __name__ == '__main__':
    main()
