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
                    parent = self.base.object_sites_dict[support].parent_name
                    if parent is None:
                        assert support.startswith(self.base.workspace_name + '_'), ('unresolved support', support)
                        parent = self.base.workspace_name
                    support = parent
                self.supports.add(tuple(sorted((state[1], support))))
        self.pairs = {}
        self.settled_pairs = {}
        self.current_pairs = set()
        self.samples = 0
        self.frames = []
        self.force = np.zeros(6)
        self.reset_positions = {name:self.d.xpos[bid].copy() for name,bid in self.body_ids.items()}
        self.sampler_checks = []
        self.table_sampled = set()
        # Table placement uses radius-adjusted sampling endpoints, which can be
        # reversed for the benchmark's narrow named regions. The goal predicate
        # tests the unadjusted site, so it is not a valid sampler provenance check.
        for sampler in self.base.placement_initializer.samplers.values():
            for obj in sampler.mujoco_objects:
                if obj.name not in self.body_ids:
                    continue
                ranges=[]
                inverted=False
                radius=float(obj.horizontal_radius) if sampler.ensure_object_boundary_in_range else 0.0
                for xr,yr in zip(sampler.x_ranges,sampler.y_ranges):
                    x=[float(xr[0])+radius,float(xr[1])-radius]
                    y=[float(yr[0])+radius,float(yr[1])-radius]
                    inverted=inverted or x[0]>x[1] or y[0]>y[1]
                    ranges.append([sorted(x),sorted(y)])
                xy=self.reset_positions[obj.name][:2]-np.asarray(sampler.reference_pos)[:2]
                valid=any(x[0]-1e-9<=xy[0]<=x[1]+1e-9 and y[0]-1e-9<=xy[1]<=y[1]+1e-9 for x,y in ranges)
                self.sampler_checks.append({'object':obj.name,'reset_xy_relative':xy.tolist(),
                    'actual_native_sampling_ranges':ranges,'radius_adjustment':radius,
                    'reversed_native_endpoints':inverted,'pass':bool(valid)})
                self.table_sampled.add(obj.name)

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
            if self.force[0]>1e-5:
                self.current_pairs.add(pair)
            if len(self.frames)>=100:
                settled=self.settled_pairs.setdefault(key,{'entities':list(pair),'declared_support':pair in self.supports,
                    'max_penetration_mm':0.0,'max_normal_N':0.0,'samples':0})
                settled['max_penetration_mm']=max(settled['max_penetration_mm'],depth)
                settled['max_normal_N']=max(settled['max_normal_N'],float(self.force[0]))
                settled['samples']+=1

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
            'predicates':predicates,'contact_pairs':[list(p) for p in sorted(self.current_pairs)]})
        self.current_pairs.clear()

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
        if not all(s['pass'] for s in self.sampler_checks):
            errors.append('reset positions disagree with unchanged native sampler ranges')
        relational_failed=[p for p in failed if p['predicate'][1] not in self.table_sampled]
        if relational_failed:
            errors.append('object/fixture support predicates not sustained during final 20 steps')
        for state in self.predicates:
            if len(state)!=3 or state[1] not in self.base.objects_dict:
                continue
            support=state[2]
            if support in self.base.object_sites_dict:
                support=self.base.object_sites_dict[support].parent_name or self.base.workspace_name
            pair=list(sorted((state[1],support)))
            if not all(pair in f['contact_pairs'] for f in final):
                errors.append('declared object support lacks sustained physical contact: '+str(pair))
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
        horizontal_displacement={name:float(np.linalg.norm(np.array(final[-1]['positions'][name])[:2]-xyz[:2])*1000)
            for name,xyz in self.reset_positions.items()} if final else {}
        if any(v>5 for v in horizontal_displacement.values()):
            errors.append('reset-to-settled horizontal displacement exceeds 5mm')
        for pair, row in self.pairs.items():
            if not row['declared_support'] and row['max_penetration_mm']>1.0:
                errors.append('undeclared preparation contact penetration exceeds 1.0mm: '+pair)
            if 'robot' in row['entities'] and row['max_normal_N'] > .1:
                errors.append('robot contacts external scene during preparation: '+pair)
        for pair,row in self.settled_pairs.items():
            if row['max_penetration_mm']>2.0:
                errors.append('settled contact penetration exceeds 2.0mm: '+pair)
        return {'pass':not errors, 'errors':errors, 'preparation_steps':len(self.frames),
            'physics_substeps_observed':self.samples, 'declared_support_pairs':[list(p) for p in sorted(self.supports)],
            'contacts':self.pairs,'settled_contacts':self.settled_pairs,'native_sampling_checks':self.sampler_checks,
            'reset_to_settled_horizontal_displacement_mm':horizontal_displacement,
            'native_predicate_failures_final_second':failed,
            'final_second_position_drift_mm':drift, 'final_second_rotation_drift_deg':angles,
            'final_native_predicates':final[-1]['predicates'] if final else [],
            'scope':'native sampler provenance + settled support/stability + unsafe preparation contacts; site-goal predicate differs from native sampling range; unchanged r6 trajectory audit remains mandatory'}
