"""Static readback of original poses and real preparation images; no sim.step."""
import json
import pathlib
import sys

import h5py
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from robosuite.utils.binding_utils import MjSim
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.bddl_utils import robosuite_parse_problem

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'versions/workspace_distance_20261008_r2/scripts'))
from geometry import Geometry


def read_case(task):
    path = ROOT / 'runs/closed_loop_20261008_r1/eco65_zhixing_lora' / f'task{task:02d}_s{640000+task*100}'
    with h5py.File(path / 'trajectory.h5', 'r') as f:
        sim = MjSim.from_xml_string(f.attrs['environment_xml'])
    m, d = sim.model._model, sim.data._data
    g = Geometry.__new__(Geometry)
    g.m, g.d = m, d
    g.gid = next(i for i in range(m.nsite) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, i) or '').endswith('grip_site'))
    g.joints = [i for i in range(m.njnt) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) or '').startswith('robot0_')]
    assert len(g.joints) == 6
    g.qadr, g.dofs = m.jnt_qposadr[g.joints], m.jnt_dofadr[g.joints]
    g.base = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'robot0_base')
    b = benchmark.get_benchmark_dict()['libero_spatial']().get_task(task)
    parsed = robosuite_parse_problem(str(pathlib.Path(get_libero_path('bddl_files')) / b.problem_folder / b.bddl_file))
    target_name = parsed['goal_state'][0][1]
    g.target = next(i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or '') == target_name + '_main')
    g.jacp, g.jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    samples = []
    for line in (path / 'steps.jsonl').read_text().splitlines():
        row = json.loads(line)
        d.qpos[:] = row['qpos_after']
        d.qvel[:] = row['qvel_after']
        mujoco.mj_forward(m, d)
        metric = g.read()
        weighted = np.linalg.svd(np.vstack([g.jacp, .2*g.jacr])[:, g.dofs], compute_uv=False)
        metric.update(step=row['step'], weighted6d_smin=float(weighted[-1]),
                      weighted6d_condition=float(weighted[0]/max(weighted[-1], 1e-12)))
        samples.append(metric)
    closest = min(samples, key=lambda x: x['tcp_target_distance_m'])
    most_extended = max(samples, key=lambda x: x['elbow_extension_ratio'])
    return {'task': task, 'seed': 640000+task*100, 'steps': len(samples), 'first': samples[0],
            'closest_to_target': closest, 'most_extended': most_extended,
            'extension_over_0_98_steps': sum(s['elbow_extension_ratio'] > .98 for s in samples),
            'minimum_position_smin': min(s['jacobian_position_smin_m_per_rad'] for s in samples),
            'minimum_weighted6d_smin': min(s['weighted6d_smin'] for s in samples),
            'minimum_joint_margin_deg': min(s['minimum_joint_margin_deg'] for s in samples)}


def main():
    out = ROOT / 'reports/workspace_geometry_20261008_r1'
    out.mkdir(exist_ok=False)
    summary = {'scope': 'static reconstruction of recorded test poses; no policy rerun, simulation stepping, controller or source changes',
               'weighted6d_rotation_scale_m': .2, 'tasks': [read_case(t) for t in [2, 3, 6]]}
    (out / 'original_pose_analysis.json').write_text(json.dumps(summary, indent=2))
    sheet = Image.new('RGB', (1536, 320), '#171c25')
    draw = ImageDraw.Draw(sheet)
    for index, cm in enumerate([0, 5, 10]):
        source = ROOT / 'runs' / f'distance_entry_smoke_20261008_{cm:02d}cm_r2'
        result = json.loads((source / 'result.json').read_text())
        pos = result['settled_geometry']['base_world_m'][0]
        caption = f'base +{cm}cm | actual x={pos:.2f}m | initial gate ' + ('PASS' if result['initialization_pass'] else 'REJECT')
        draw.text((index*512+5, 7), caption, fill='white')
        draw.text((index*512+5, 23), 'REAL PREPARATION VIEWS | agentview + wrist | seed630201', fill='#d1d9ea')
        sheet.paste(Image.open(source / 'initial_views.jpg').convert('RGB'), (index*512, 64))
    sheet.save(out / 'preparation_comparison.png')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
