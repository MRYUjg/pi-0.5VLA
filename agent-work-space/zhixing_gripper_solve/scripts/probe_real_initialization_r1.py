"""Observe native reset and unchanged quiet_hold; never repair/resample the state."""
import argparse, hashlib, json, os, pathlib
import numpy as np
import mujoco
import real_object_teacher_r20 as T
ap = argparse.ArgumentParser()
ap.add_argument('--source', required=True)
ap.add_argument('--out', required=True)
a = ap.parse_args()
source = pathlib.Path(a.source)
reference = json.loads((source / 'result.json').read_text())
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=False)
os.environ['EXTRA_JOINT_PD_PROFILE'] = 'legacy'
os.environ['EXTRA_JOINT_PD_KP'] = ''
os.environ['ZHIXING_GRIPPER_SPEED'] = '0.01'
env, task = T.D.build(reference['suite'], 0, seed=reference['seed'])
T.EX.install_generic_pad_geoms(print)
T.EX.install_generic_state_gripper(env, print)
ep = T.N3.ChainEpisode(env, None)
ep.start()
sim, m, d, gid = T.N.ids(env)
obj, _ = T.N4.find_body(sim, T.N4.goal_objects(env)[0])
objgeoms = set(T.S.geometry(env)['object_geoms'])
force = np.zeros(6)
def sample():
    contacts = []
    for i in range(d.ncon):
        c = d.contact[i]
        if int(c.geom1) not in objgeoms and int(c.geom2) not in objgeoms:
            continue
        mujoco.mj_contactForce(m, d, i, force)
        if force[0] > 1e-6 or c.dist < 0:
            contacts.append({'geoms': [T.S.name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in (c.geom1, c.geom2)],
                             'normal_N': float(force[0]), 'penetration_mm': max(0., -float(c.dist)) * 1000})
    return {'time': float(d.time), 'object_pos': d.xpos[obj].copy(),
            'object_bbox': T.S.bbox(m, d, list(objgeoms)), 'contacts': contacts}
initial = sample()
rows = []
original_step = sim.step
def observed_step():
    original_step()
    rows.append(sample())
sim.step = observed_step
T.N4.quiet_hold(env, 120, -1)
final_hash = hashlib.sha256(d.qpos.tobytes()).hexdigest()
result = {'version': 'probe_real_initialization_r1', 'seed': reference['seed'], 'reset_state': initial,
          'after_quiet': sample(), 'quiet_control_steps': 120, 'physics_samples': len(rows),
          'final_qpos_sha256': final_hash, 'reference_qpos_sha256': reference['initial_state_sha256'],
          'exact_qpos_reproduction': final_hash == reference['initial_state_sha256'],
          'source_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
          'scope': 'Observation of reset-to-preteacher interval. No state writes, resampling or gate changes.'}
T.D.dump(out / 'result.json', result)
with (out / 'physics_contacts.jsonl').open('x') as f:
    for row in rows:
        f.write(json.dumps(row, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item()) + '\n')
(out / 'environment.xml').write_text(env.env.model.get_xml())
env.close()
print('EXACT REPRODUCTION', result['exact_qpos_reproduction'])
raise SystemExit(0 if result['exact_qpos_reproduction'] else 1)
