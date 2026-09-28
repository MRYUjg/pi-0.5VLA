"""Headless, model-free controller audit. All patches are process-local."""
import argparse
import collections
import csv
import hashlib
import inspect
import json
import pathlib
import textwrap
import time
import types

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
from robosuite.robots.single_arm import SingleArm
import robosuite.utils.transform_utils as T


ROOT = pathlib.Path(__file__).resolve().parent
ORIGINAL_CONTROL = SingleArm.control
CONTROL_SOURCE = textwrap.dedent(inspect.getsource(ORIGINAL_CONTROL))
DUMMY = np.array([0., 0., 0., 0., 0., 0., -1.])


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1))))


def patch_pd(enabled):
    # Fail closed if the implementation under audit has changed.
    assert CONTROL_SOURCE.count('joint_kp = 3.0') == 1
    if enabled:
        SingleArm.control = ORIGINAL_CONTROL
    else:
        ns = dict(ORIGINAL_CONTROL.__globals__)
        exec(compile(CONTROL_SOURCE.replace('joint_kp = 3.0', 'joint_kp = 0.0'),
                     '<controller-audit-pd-off>', 'exec'), ns)
        SingleArm.control = ns['control']


def pose(sim, site):
    return sim.data.site_xpos[site].copy(), sim.data.site_xmat[site].reshape(3, 3).copy()


def run_case(robot_name, pd_on, case, seed, duration, out):
    patch_pd(pd_on)
    np.random.seed(seed)
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    task = suite.get_task(0)
    bddl = pathlib.Path(get_libero_path('bddl_files')) / task.problem_folder / task.bddl_file
    env = ControlEnv(bddl_file_name=str(bddl), robots=[robot_name], gripper_types='PandaGripper',
                     use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False,
                     control_freq=20, ignore_done=True, initialization_noise=None)
    try:
        env.seed(seed)
        env.reset()
        r = env.env.robots[0]
        sim, c = env.env.sim, r.controller
        site = sim.model.site_name2id(c.eef_name)
        assert type(r.gripper).__name__ == 'PandaGripper'
        qidx = np.asarray(r._ref_joint_pos_indexes)
        jids = np.asarray([sim.model.joint_name2id(n) for n in r.robot_model.joints])
        lo, hi = sim.model.jnt_range[jids].T
        qreset = sim.data.qpos.copy()
        initial_p, initial_R = pose(sim, site)
        # Same reset and ten native no-op actions as the evaluation harness.
        for _ in range(10):
            env.step(DUMMY)
        sim.forward()
        start_p, start_R = pose(sim, site)
        target_p, target_R = start_p.copy(), start_R.copy()
        if case.startswith('pos_'):
            axis = 'xyz'.index(case[4]); sign = 1 if case.endswith('plus') else -1
            target_p[axis] += sign * .02
        elif case.startswith('rot_'):
            axis = 'xyz'.index(case[4]); sign = 1 if case.endswith('plus') else -1
            aa = np.zeros(3); aa[axis] = sign * .1
            target_R = T.quat2mat(T.axisangle2quat(aa)) @ start_R
        native = case.startswith('native_')
        set_goal = c.set_goal
        if not native:
            def fixed_goal(self, action, **kwargs):
                return set_goal(action, set_pos=target_p, set_ori=target_R)
            c.set_goal = types.MethodType(fixed_goal, c)

        stats = {'n': 0, 'sat': np.zeros(len(qidx)), 'peak_ratio': np.zeros(len(qidx)),
                 'min_util': np.ones(len(qidx)), 'max_util': np.zeros(len(qidx)),
                 'contacts': collections.Counter(), 'sigma_min': float('inf'),
                 'peak_pd_nm': 0., 'clip_verified': True}
        original_run = c.run_controller
        expected = [None]
        torque_lo, torque_hi = r.torque_limits
        def instrumented_run():
            osc = original_run()
            q, v = sim.data.qpos[qidx], sim.data.qvel[r._ref_joint_vel_indexes]
            pd = 3. * (r.init_qpos - q) - 2 * np.sqrt(3.) * v if pd_on else np.zeros_like(q)
            total = np.asarray(osc) + pd
            expected[0] = np.clip(total, torque_lo, torque_hi)
            stats['n'] += 1
            stats['sat'] += (total < torque_lo) | (total > torque_hi)
            stats['peak_ratio'] = np.maximum(stats['peak_ratio'], np.abs(total) / np.maximum(np.abs(torque_hi), 1e-9))
            util = (q - lo) / (hi - lo)
            stats['min_util'] = np.minimum(stats['min_util'], util)
            stats['max_util'] = np.maximum(stats['max_util'], util)
            stats['sigma_min'] = min(stats['sigma_min'], float(np.linalg.svd(c.J_full, compute_uv=False)[-1]))
            stats['peak_pd_nm'] = max(stats['peak_pd_nm'], float(np.max(np.abs(pd))))
            for k in range(sim.data.ncon):
                con = sim.data.contact[k]
                names = [sim.model.geom_id2name(int(g)) or str(g) for g in (con.geom1, con.geom2)]
                if any(n.startswith(('robot0_', 'gripper0_')) for n in names):
                    stats['contacts'][' | '.join(sorted(names))] += 1
            return osc
        c.run_controller = instrumented_run
        # Verify at every physics substep that the measured torque is what was applied.
        control = r.control
        def checked_control(action, policy_step=False):
            control(action, policy_step=policy_step)
            applied = sim.data.ctrl[r._ref_joint_actuator_indexes]
            if not np.allclose(applied, expected[0], rtol=1e-9, atol=1e-9):
                stats['clip_verified'] = False
                raise AssertionError('Torque instrumentation differs from actuator command')
        r.control = checked_control
        rows = []
        for k in range(int(duration * 20)):
            action = DUMMY.copy()
            if native:
                axis = 'xyz'.index(case[-1])
                if k < 5: action[axis] = .1
                elif k < 10: action[axis] = -.1
            env.step(action)
            sim.forward()
            p, R = pose(sim, site)
            gp, gR = c.goal_pos.copy(), c.goal_ori.copy()
            assert np.isfinite(sim.data.qpos).all() and np.isfinite(sim.data.qvel).all()
            if not native:
                assert np.allclose(gp, target_p) and np.allclose(gR, target_R)
            row = {'t_s': (k+1)/20., 'position_error_mm': np.linalg.norm(gp-p)*1000,
                   'rotation_error_deg': angle_deg(gR, R), 'start_displacement_mm': np.linalg.norm(p-start_p)*1000}
            for j, ax in enumerate('xyz'):
                row['eef_'+ax] = p[j]; row['goal_'+ax] = gp[j]; row['action_'+ax] = action[j]
            rows.append(row)
        p, R = pose(sim, site)
        tail = rows[-10:]
        position_error = float(np.mean([x['position_error_mm'] for x in tail]))
        rotation_error = float(np.mean([x['rotation_error_deg'] for x in tail]))
        result = {
            'robot': robot_name, 'extra_pd': pd_on, 'case': case, 'seed': seed,
            'duration_s': duration, 'gripper': type(r.gripper).__name__, 'controller': type(c).__name__,
            'reset_qpos_sha256': hashlib.sha256(qreset.tobytes()).hexdigest(),
            'initial_eef_pos': initial_p.tolist(), 'start_eef_pos': start_p.tolist(),
            'warmup_drift_mm': float(np.linalg.norm(start_p-initial_p)*1000),
            'start_eef_rotmat': start_R.tolist(), 'final_eef_pos': p.tolist(),
            'tail_position_error_mm': position_error, 'tail_rotation_error_deg': rotation_error,
            'max_position_error_mm': max(x['position_error_mm'] for x in rows),
            'final_offset_from_start_mm': ((p-start_p)*1000).tolist(),
            'final_rotation_from_start_deg': angle_deg(R, start_R),
            'screen_pass': None if native else position_error < 2 and rotation_error < 2,
            'torque_saturation_fraction_per_joint': (stats['sat']/stats['n']).tolist(),
            'peak_preclip_torque_limit_ratio': stats['peak_ratio'].tolist(),
            'joint_limit_util_min': stats['min_util'].tolist(), 'joint_limit_util_max': stats['max_util'].tolist(),
            'jacobian_sigma_min_mixed_units': stats['sigma_min'],
            'max_extra_pd_torque_nm': stats['peak_pd_nm'], 'physics_samples': stats['n'],
            'torque_instrumentation_verified': stats['clip_verified'],
            'robot_contact_pair_counts': dict(stats['contacts']),
            'control_freq': env.env.control_freq, 'physics_dt': env.env.model_timestep,
        }
        name = '{}_pd{}_{}'.format(robot_name, 'on' if pd_on else 'off', case)
        with (out / (name + '.csv')).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (out / (name + '.json')).write_text(json.dumps(result, indent=2))
        print('{}: pos={:.3f}mm ori={:.3f}deg sat={:.1%} pass={}'.format(name, position_error,
              rotation_error, max(result['torque_saturation_fraction_per_joint']), result['screen_pass']), flush=True)
        return result
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    out = ROOT / args.output; out.mkdir(parents=True, exist_ok=True)
    cases = ['hold'] + [kind+'_'+ax+'_'+sign for kind in ('pos', 'rot') for ax in 'xyz' for sign in ('plus','minus')]
    cases += ['native_'+ax for ax in 'xyz']
    robots = ['ECO65', 'Panda']
    if args.smoke: cases, robots = ['hold', 'pos_z_plus'], ['ECO65']
    manifest = {'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                'control_source_sha256': hashlib.sha256(CONTROL_SOURCE.encode()).hexdigest(),
                'seed': 7, 'cases': cases, 'robots': robots, 'threshold_note': '2 mm / 2 deg tail mean is a screening criterion, not a hardware specification or task success metric.'}
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    results = []
    for robot in robots:
        for mode in (True, False):
            for case in cases:
                results.append(run_case(robot, mode, case, 7, 3., out))
                (out/'summary.json').write_text(json.dumps(results, indent=2))
    for robot in robots:
        for case in cases:
            pair = [r for r in results if r['robot']==robot and r['case']==case]
            assert pair[0]['reset_qpos_sha256'] == pair[1]['reset_qpos_sha256'], 'PD pairs must start at identical simulator qpos'
    print('AUDIT_COMPLETE: {} cases; reset pairing and torque instrumentation verified'.format(len(results)), flush=True)


if __name__ == '__main__':
    main()
