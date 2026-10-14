"""Read-only native initialization checks. No state injection or physics changes."""
import collections
import numpy as np
import mujoco


class Initialization:
    def __init__(self, env):
        self.env = env
        self.base = env.env
        self.m = env.sim.model._model
        self.d = env.sim.data._data
        self.predicates = self.base.parsed_problem['initial_state']
        self.entities = {**self.base.objects_dict, **self.base.fixtures_dict}
        self.body_ids = {}
        for name, obj in self.entities.items():
            bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, obj.root_body)
            if bid >= 0:
                self.body_ids[name] = bid
        self.geom_names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, i) or '' for i in range(self.m.ngeom)]
        self.labels = []
        for gid, name in enumerate(self.geom_names):
            label = None
            if name.startswith(('robot0_', 'gripper0_')):
                label = 'robot'
            else:
                b = int(self.m.geom_bodyid[gid])
                while b and label is None:
                    label = next((n for n, bid in self.body_ids.items() if bid == b), None)
                    b = int(self.m.body_parentid[b])
                if label is None:
                    label = 'main_table' if 'table' in name else ('floor' if 'floor' in name else 'world')
            self.labels.append(label)
        self.supports = set()
        for state in self.predicates:
            if len(state) == 3 and state[0] in ('on', 'in'):
                support = state[2]
                if support in self.base.object_sites_dict:
                    support = self.base.object_sites_dict[support].parent_name
                self.supports.add(tuple(sorted((state[1], support))))
        self.pairs = {}
        self.samples = 0
        self.frames = []
        self.force = np.zeros(6)

    def substep(self):
        self.samples += 1
        for i, c in enumerate(self.d.contact):
            a, b = int(c.geom1), int(c.geom2)
            la, lb = self.labels[a], self.labels[b]
            # Static scene and internal object/robot geometry are not object support pairs.
            if la == lb or not ({la, lb} & (set(self.base.objects_dict) | {'robot'})):
                continue
            pair = tuple(sorted((la, lb)))
            key = '|'.join(pair)
            row = self.pairs.setdefault(key, {'entities':list(pair), 'declared_support':pair in self.supports,
                'max_penetration_mm':0.0, 'max_normal_N':0.0, 'samples':0, 'worst_geoms':[]})
            depth = max(0.0, -float(c.dist)) * 1000
            mujoco.mj_contactForce(self.m, self.d, i, self.force)
            if depth >= row['max_penetration_mm']:
                row['max_penetration_mm'] = depth
                row['worst_geoms'] = [self.geom_names[a], self.geom_names[b]]
            row['max_normal_N'] = max(row['max_normal_N'], float(self.force[0]))
            row['samples'] += 1

    def control_step(self):
        predicates = []
        for state in self.predicates:
            try:
                value = bool(self.base._eval_predicate(state))
                error = None
            except Exception as exc:
                value = False
                error = type(exc).__name__ + ': ' + str(exc)
            predicates.append({'predicate':state, 'true':value, 'error':error})
        self.frames.append({'time':float(self.d.time), 'native_success':bool(self.env.check_success()),
            'positions':{name:self.d.xpos[bid].tolist() for name,bid in self.body_ids.items()},
            'quaternions':{name:self.d.xquat[bid].tolist() for name,bid in self.body_ids.items()},
            'finite':bool(np.isfinite(self.d.qpos).all() and np.isfinite(self.d.qvel).all()),
            'predicates':predicates})

    def result(self):
        errors = []
        if len(self.frames) != 120:
            errors.append('expected 120 native preparation control steps')
        final = self.frames[-20:]
        if len(final) < 20 or not all(f['finite'] for f in self.frames):
            errors.append('invalid or incomplete preparation state')
        if any(f['native_success'] for f in self.frames):
            errors.append('native goal already true during preparation')
        failed = [p for f in final for p in f['predicates'] if not p['true']]
        if failed:
            errors.append('native BDDL initial predicates not sustained during final 20 steps')
        drift = {}
        angles = {}
        if final:
            for name in self.body_ids:
                xyz = np.array([f['positions'][name] for f in final])
                qs = np.array([f['quaternions'][name] for f in final])
                drift[name] = float(np.linalg.norm(xyz - xyz[0], axis=1).max() * 1000)
                angles[name] = float(np.rad2deg(2*np.arccos(np.clip(np.abs(qs @ qs[0]),0,1))).max())
        if any(v > 5 for v in drift.values()):
            errors.append('object/fixture drift exceeds 5mm during final preparation second')
        if any(v > 5 for v in angles.values()):
            errors.append('object/fixture rotation exceeds 5deg during final preparation second')
        for pair, row in self.pairs.items():
            limit = 2.0 if row['declared_support'] else 1.0
            if row['max_penetration_mm'] > limit:
                errors.append('initial contact penetration exceeds '+str(limit)+'mm: '+pair)
            if 'robot' in row['entities'] and row['max_normal_N'] > .1:
                errors.append('robot contacts external scene during preparation: '+pair)
        return {'pass':not errors, 'errors':errors, 'preparation_steps':len(self.frames),
            'physics_substeps_observed':self.samples, 'declared_support_pairs':[list(p) for p in sorted(self.supports)],
            'contacts':self.pairs, 'native_predicate_failures_final_second':failed,
            'final_second_position_drift_mm':drift, 'final_second_rotation_drift_deg':angles,
            'final_native_predicates':final[-1]['predicates'] if final else [],
            'scope':'additional formal-data initialization gate; unchanged r6 teacher trajectory audit remains mandatory'}
