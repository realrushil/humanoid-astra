"""Measured-clearance correction of raw neural wrist targets.

Only 14 arm position references change. The common world +Z bias preserves the
raw desired wrist orientations and separation; finger and WBC commands pass
through. Extra references, not total neural command speed, are slew-limited.
"""
import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation
from .hand_clearance import ClearanceBias


class AcquisitionHandClearance:
    def __init__(self, robot, names, geometry):
        self.robot, self.names = robot, names
        self.model = robot.pinocchio_wrapper.model
        dofs = robot.get_joint_group_indices('arms')
        self.arms = [n for n in robot.joint_names if robot.dof_index(n) in dofs]
        assert len(self.arms) == 14 and not robot.is_floating_base_model
        self.indices = [names.index(n) for n in self.arms]
        self.locked = [self.model.getJointId(n) for n in robot.joint_names if n not in self.arms]
        self.frames = [robot.supplemental_info.hand_frame_names[s] for s in ('left','right')]
        self.clearance = ClearanceBias(geometry)
        self.extra = np.zeros(14)
        self.last = None

    def command(self, observation, proposal):
        diagnostic = self.clearance.update(observation)
        wxyz = observation['root_quat']
        rotation = Rotation.from_quat([*wxyz[1:], wxyz[0]]).as_matrix()
        return self._project(observation['joint_pos'], proposal, diagnostic, rotation.T[:,2])

    def command_measured(self, joints, proposal, feedback):
        """Body+Dex3 q and observed table up in pelvis frame; no true root pose."""
        if feedback['status'] != 'available':
            raise ValueError('sensor_hand_clearance_unavailable')
        up = np.asarray(feedback['up_body'], dtype=float)
        joints = np.asarray(joints, dtype=float)
        if (joints.shape != (len(self.names),) or not np.isfinite(joints).all() or
                up.shape != (3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1) > .001):
            raise ValueError('Invalid measured hand correction inputs')
        diagnostic = self.clearance.update_gap(feedback['time_s'], feedback['margin_m'])
        return self._project(joints, proposal, diagnostic, up)

    def _project(self, joints, proposal, diagnostic, up):
        proposal = np.asarray(proposal, dtype=float)
        if proposal.shape != (50,) or not np.isfinite(proposal).all():
            raise ValueError('Expected finite 50-value native action')
        bias = diagnostic['bias_m']
        desired_extra = np.zeros(14)
        errors = []
        if bias > 1e-9:
            q = self.robot.q_default.copy()
            for name, measured in zip(self.names, joints):
                q[self.robot.dof_index(name)] = (proposal[self.names.index(name)]
                    if name in self.arms or 'hand' in name else measured)
            model = pin.buildReducedModel(self.model, self.locked, q)
            reduced_names = list(model.names)[1:]
            configuration = pink.Configuration(model, model.createData(),
                np.array([q[self.robot.dof_index(n)] for n in reduced_names]))
            shift = up * bias
            tasks = []
            targets = {}
            for frame in self.frames:
                target = configuration.get_transform_frame_to_world(frame).copy()
                target.translation += shift
                task = FrameTask(frame, position_cost=1., orientation_cost=.5, gain=1.)
                task.set_target(target)
                targets[frame] = target
                tasks.append(task)
            posture = PostureTask(cost=.001)
            posture.set_target_from_configuration(configuration)
            tasks.append(posture)
            # Static target projection, not eight physical steps. Actual reference
            # changes are bounded below and require separate physical validation.
            for _ in range(8):
                velocity = pink.solve_ik(configuration, tasks, dt=1., solver='osqp')
                configuration.integrate_inplace(velocity, 1.)
            for frame, target in targets.items():
                actual = configuration.get_transform_frame_to_world(frame)
                errors.append(float(np.linalg.norm(actual.translation-target.translation)))
                if np.linalg.norm(pin.log3(actual.rotation.T @ target.rotation)) > .02:
                    raise ValueError('Projected wrist orientation error')
            if max(errors) > .002:
                raise ValueError('Projected wrist translation error')
            solved = dict(zip(reduced_names, configuration.q))
            desired_extra = np.array([solved[n]-proposal[self.names.index(n)] for n in self.arms])
        clipped = np.clip(desired_extra, -.20, .20)
        self.extra += np.clip(clipped-self.extra, -.02, .02)
        output = proposal.copy()
        for index, name, delta in zip(self.indices, self.arms, self.extra):
            dof = self.robot.dof_index(name)
            output[index] = np.clip(proposal[index]+delta,
                self.model.lowerPositionLimit[dof], self.model.upperPositionLimit[dof])
        self.last = {**diagnostic, 'extra_arm_reference_rad': self.extra.tolist(),
            'maximum_ideal_projection_error_m': max(errors, default=0.),
            'projection_clipped': bool(np.any(abs(desired_extra) > .20)),
            'corrected': bool(np.any(output != proposal))}
        return output.tolist()
