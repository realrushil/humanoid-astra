"""Small local right-arm IK planner using the actual MuJoCo collision model.

This is a position-only, fixed-configuration kinematic planner, not a balance
controller or a global motion planner. Imports stay light until instantiated.
"""
import math
from ..models import finite_number

ARM_SUFFIXES = ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow',
                'wrist_roll', 'wrist_pitch', 'wrist_yaw')
RIGHT_ARM = tuple(f'right_{name}_joint' for name in ARM_SUFFIXES)
# SONIC's 17-element IsaacLab upper-body wire order, pinned source mapping:
# MuJoCo body indices 12,13,14,15,22,16,23,17,24,18,25,19,26,20,27,21,28.
UPPER_BODY = ('waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint') + tuple(
    f'{side}_{name}_joint' for name in ARM_SUFFIXES for side in ('left', 'right'))


def upper_reference(sample, right_arm=None):
    """Build named waist/arm reference; optional right arm must have seven radians."""
    names, values = sample['body_joint_names'], sample['body_joint_positions']
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError('invalid named body joint state')
    joints = dict(zip(names, values))
    if right_arm is not None:
        if len(right_arm) != 7 or not all(finite_number(v) for v in right_arm):
            raise ValueError('seven finite right-arm joint values required')
        joints.update(zip(RIGHT_ARM, right_arm))
    if not all(name in joints and finite_number(joints[name]) for name in UPPER_BODY):
        raise ValueError('missing or invalid waist/arm joint state')
    return [float(joints[name]) for name in UPPER_BODY]


class ArmPlanner:
    def __init__(self, model_path):
        import mujoco
        import numpy as np
        self.mj, self.np = mujoco, np
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.wrist = self.model.body('right_wrist_yaw_link').id
        self.joints = [self.model.joint(name).id for name in RIGHT_ARM]
        self.q_indices = self.model.jnt_qposadr[self.joints]
        self.v_indices = self.model.jnt_dofadr[self.joints]
        if not all(self.model.jnt_limited[j] for j in self.joints):
            raise ValueError('right-arm joints must have explicit limits')
        self.lower = self.model.jnt_range[self.joints, 0] + .005
        self.upper = self.model.jnt_range[self.joints, 1] - .005

    def _load(self, sample):
        names, positions = sample['joint_names'], sample['joint_positions']
        from ..sim_state import robot_joint_ids
        expected = [self.model.joint(i).name for i in robot_joint_ids(self.model)]
        if (len(names) != len(positions) or len(set(names)) != len(names)
                or set(names) != set(expected) or not all(finite_number(v) for v in positions)):
            raise ValueError('measured joints must exactly match the planning model')
        root = list(sample['pelvis_position']) + list(sample['pelvis_quaternion_wxyz'])
        if len(root) != 7 or not all(finite_number(v) for v in root) or abs(math.hypot(*root[3:])-1.) > .01:
            raise ValueError('finite root pose and unit WXYZ quaternion required')
        self.data.qpos[:7] = root
        # Free object poses must come from this same measured snapshot. Keep
        # them in planning collisions; never silently use initial scene poses.
        for j in range(self.model.njnt):
            if self.model.jnt_type[j]!=self.mj.mjtJoint.mjJNT_FREE or self.model.jnt_qposadr[j]==0: continue
            name=self.model.body(int(self.model.jnt_bodyid[j])).name
            obj=sample.get('objects',{}).get(name)
            if obj is None: raise ValueError('missing measured free-object pose: '+name)
            pose=list(obj['position_world'])+list(obj['quaternion_wxyz'])
            if len(pose)!=7 or not all(finite_number(v) for v in pose) or abs(math.hypot(*pose[3:])-1.)>.01:
                raise ValueError('invalid measured free-object pose: '+name)
            adr=self.model.jnt_qposadr[j]
            self.data.qpos[adr:adr+7]=pose
        for name, value in zip(names, positions):
            self.data.joint(name).qpos[0] = value
        self.data.qvel[:] = 0.
        self.mj.mj_forward(self.model, self.data)

    def _collision(self):
        # Match the simulator's declared collision pairs/exclusions. This checks
        # sampled configurations only; omitted collision meshes are not inferred.
        for contact in self.data.contact:
            root=self.model.body('pelvis').id
            if not any(self.model.body_rootid[self.model.geom_bodyid[g]]==root for g in contact.geom):
                continue
            if contact.dist >= -.002:
                continue
            bodies = {self.model.body(int(self.model.geom_bodyid[g])).name for g in contact.geom}
            if any(bodies == {'world', f'{side}_ankle_roll_link'} for side in ('left', 'right')):
                continue
            return dict(bodies=sorted(bodies),
                        geom_ids=[int(g) for g in contact.geom],
                        geoms=[self.model.geom(int(g)).name or f'geom_{g}' for g in contact.geom],
                        distance_m=float(contact.dist))
        return None

    def _distance(self, q, geom_ids):
        """Signed geometry distance in metres; touches only planning data."""
        self.data.qpos[self.q_indices] = q
        self.mj.mj_forward(self.model, self.data)
        return self.mj.mj_geomDistance(self.model, self.data, *geom_ids, .05, None)

    def _clear_step(self, q, delta):
        """Try at most eight local corrections, checking the whole sampled step.

        Finite differences (1e-4 rad) estimate the colliding pair's distance
        gradient. Project the proposed joint displacement toward 3 mm separation,
        then reapply limits and all collision checks. This is a local heuristic,
        not a guarantee of finding a path or maintaining a 3 mm clearance globally.
        """
        np = self.np
        last_collision = None
        for retry in range(8):
            delta *= min(1., .025/max(float(np.max(np.abs(delta))), 1e-12))
            candidate = np.clip(q+delta, self.lower, self.upper)
            collision = None
            for alpha in (.2, .4, .6, .8, 1.):
                self.data.qpos[self.q_indices] = q+(candidate-q)*alpha
                self.mj.mj_forward(self.model, self.data)
                collision = self._collision()
                if collision:
                    break
            if collision is None:
                return candidate, last_collision, retry
            last_collision = collision
            blocked_q = self.data.qpos[self.q_indices].copy()
            if retry == 7:
                return None, collision, retry
            ids = collision['geom_ids']
            distance = self._distance(q, ids)
            gradient = np.zeros(len(q))
            for index in range(len(q)):
                plus, minus = q.copy(), q.copy()
                plus[index] += .0001
                minus[index] -= .0001
                gradient[index] = (self._distance(plus, ids)-self._distance(minus, ids))/.0002
            norm_squared = float(gradient @ gradient)
            if norm_squared < 1e-10:
                self.data.qpos[self.q_indices] = blocked_q
                self.mj.mj_forward(self.model, self.data)
                return None, collision, retry
            correction = max(0., .003-distance-float(gradient @ delta))/norm_squared
            delta += correction*gradient

    def plan(self, sample, target_world):
        """Return a seven-joint path in radians; target is wrist body origin, world m.

        Root, waist, left arm and fingers remain measured values in a separate
        MjData. Actual robot motion is never changed by this computation.
        """
        if len(target_world) != 3 or not all(finite_number(v) for v in target_world):
            raise ValueError('target must be three finite world coordinates')
        self._load(sample)
        np, mj = self.np, self.mj
        target = np.asarray(target_world, dtype=float)
        initial_wrist = self.data.xpos[self.wrist].copy()
        if np.linalg.norm(target-initial_wrist) > .25:
            return dict(status='rejected', reason='target_outside_local_envelope', joint_path=[])
        collision = self._collision()
        if collision:
            return dict(status='rejected', reason='initial_model_collision', joint_path=[], collision=collision)
        q = self.data.qpos[self.q_indices].copy()
        if np.any(q < self.lower) or np.any(q > self.upper):
            return dict(status='rejected', reason='initial_joint_limit', joint_path=[])
        path = [q.tolist()]
        corrections = 0
        last_collision = None
        jac = np.zeros((3, self.model.nv))
        for _ in range(160):
            error = target-self.data.xpos[self.wrist]
            distance = float(np.linalg.norm(error))
            if distance <= .008:
                return dict(status='planned', reason='local_ik', joint_path=path,
                            position_error=distance, target_world=list(target_world),
                            collision_check='discrete_declared_model_pairs',
                            collision_corrections=corrections,
                            max_joint_step_rad=.025)
            mj.mj_jacBody(self.model, self.data, jac, None, self.wrist)
            arm_jac = jac[:, self.v_indices]
            # Damped least squares, metres/radians. Bound every configuration
            # increment to 0.025 rad; this also spaces the collision samples.
            delta = arm_jac.T @ np.linalg.solve(arm_jac @ arm_jac.T + .0025*np.eye(3), error)
            next_q, collision, count = self._clear_step(q, delta)
            corrections += count
            if collision:
                last_collision = collision
            if next_q is None:
                return dict(status='rejected', reason='path_model_collision', joint_path=[],
                            collision=collision, collision_corrections=corrections,
                            rejected_right_arm=self.data.qpos[self.q_indices].tolist())
            if float(np.max(np.abs(next_q-q))) < 1e-7:
                break
            q = next_q
            path.append(q.tolist())
        result = dict(status='rejected', reason='path_model_collision' if last_collision else 'ik_not_converged',
                      joint_path=[], collision_corrections=corrections,
                      position_error=float(np.linalg.norm(target-self.data.xpos[self.wrist])))
        if last_collision:
            result['collision'] = last_collision
        return result
