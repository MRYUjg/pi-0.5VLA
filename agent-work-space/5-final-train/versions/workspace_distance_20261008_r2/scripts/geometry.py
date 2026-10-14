"""Process-local base translation and read-only kinematic diagnostics."""
import mujoco
import numpy as np


def apply_offset(offset):
    from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
    original = ManipulatorModel.set_base_xpos

    def translated(self, position):
        assert 'ECO65' in type(self).__name__, type(self).__name__
        return original(self, np.asarray(position) + [offset, 0., 0.])

    ManipulatorModel.set_base_xpos = translated


class Geometry:
    def __init__(self, env, grip_site, initialization):
        self.m = env.sim.model._model
        self.d = env.sim.data._data
        self.gid = grip_site
        robot = env.env.robots[0]
        self.joints = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
                       for name in robot.robot_model.joints]
        assert len(self.joints) == 6 and min(self.joints) >= 0
        self.qadr = self.m.jnt_qposadr[self.joints]
        self.dofs = self.m.jnt_dofadr[self.joints]
        self.base = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, robot.robot_model.root_body)
        goal = env.env.parsed_problem['goal_state'][0]
        self.target_name = goal[1]
        self.target = initialization.body_ids[self.target_name]
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))

    def read(self):
        m, d = self.m, self.d
        mujoco.mj_jacSite(m, d, self.jacp, self.jacr, self.gid)
        values = np.linalg.svd(self.jacp[:, self.dofs], compute_uv=False)
        # Joint origins 2/3/4 define shoulder/elbow/wrist in this 6-axis chain.
        shoulder, elbow, wrist = d.xanchor[np.asarray(self.joints)[1:4]]
        denominator = np.linalg.norm(elbow-shoulder) + np.linalg.norm(wrist-elbow)
        q = d.qpos[self.qadr]
        margins = np.minimum(q-m.jnt_range[self.joints, 0], m.jnt_range[self.joints, 1]-q)
        return {'base_world_m': d.xpos[self.base].tolist(),
                'target_world_m': d.xpos[self.target].tolist(),
                'tcp_world_m': d.site_xpos[self.gid].tolist(),
                'base_target_horizontal_m': float(np.linalg.norm((d.xpos[self.target]-d.xpos[self.base])[:2])),
                'tcp_target_distance_m': float(np.linalg.norm(d.xpos[self.target]-d.site_xpos[self.gid])),
                'arm_qpos_rad': q.tolist(), 'minimum_joint_margin_deg': float(np.rad2deg(margins.min())),
                'elbow_extension_ratio': float(np.linalg.norm(wrist-shoulder)/denominator),
                'jacobian_position_smin_m_per_rad': float(values[-1]),
                'jacobian_position_condition': float(values[0]/max(values[-1], 1e-12))}
