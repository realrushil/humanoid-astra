"""Measured-state paired wrist control with loaded actuator-reference retention.

Targets are world-frame metres; joint commands are radians. Only 14 arm joints
are commanded here. HOMIE retains balance ownership; finger references remain unchanged.
"""
import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation

class PairedWristController:
    def __init__(self, robot_model, joint_names, measured, root_pos, root_xyzw, wrists_world, translation_world=None, *, inward_m=0., max_translation_m=.06):
        # Optional common world translation preserves relative wrist targets.
        # The common loaded-lift request moves at 1 cm/s.
        self.translation = None if translation_world is None else np.array(translation_world,dtype=float)
        if isinstance(inward_m,bool) or not np.isfinite(inward_m) or not 0<=inward_m<=.004:
            raise ValueError('inward wrist displacement must be 0–4 mm per hand')
        self.inward_m=inward_m
        # Ordinary lift/translation keeps 6 cm. Placement preparation explicitly
        # opts into its physically probed 8 cm envelope; no caller can exceed it.
        if isinstance(max_translation_m,bool) or not np.isfinite(max_translation_m) or not 0<max_translation_m<=.08:
            raise ValueError('common wrist translation limit must be positive and at most 8 cm')
        if self.translation is not None:
            if self.translation.shape!=(3,) or not np.isfinite(self.translation).all() or np.linalg.norm(self.translation)>max_translation_m+1e-6:
                raise ValueError(f'common wrist translation must be finite and at most {100*max_translation_m:g} cm')
        if inward_m and (self.translation is None or np.linalg.norm(self.translation)>0):
            raise ValueError('inward preparation requires zero common translation')
        self.robot = robot_model
        self.names = joint_names
        arm_dofs = robot_model.get_joint_group_indices('arms')
        self.arms = [n for n in robot_model.joint_names if robot_model.dof_index(n) in arm_dofs]
        self.ids = [joint_names.index(n) for n in self.arms]
        assert len(self.ids) == 14 and not robot_model.is_floating_base_model
        self.model = robot_model.pinocchio_wrapper.model
        self.locked = [self.model.getJointId(n) for n in robot_model.joint_names if n not in self.arms]
        self.frames = [robot_model.supplemental_info.hand_frame_names[s] for s in ['left','right']]
        self.targets = {}
        q = self.full_q(measured)
        data = self.model.createData()
        pin.framesForwardKinematics(self.model, data, q)
        root = pin.SE3(Rotation.from_quat(root_xyzw).as_matrix(), np.array(root_pos))
        self.fk_errors = {}
        for frame in self.frames:
            pose = root * data.oMf[self.model.getFrameId(frame)]
            observed = wrists_world[frame]
            error = float(np.linalg.norm(pose.translation - np.array(observed['pos'])))
            angle = float(np.linalg.norm(pin.log3(pose.rotation.T @ Rotation.from_quat(observed['xyzw']).as_matrix())))
            self.fk_errors[frame] = {'position_m':error,'rotation_rad':angle}
            if error > .01 or angle > .05:
                raise ValueError(f'Wrist FK disagrees with simulator: {self.fk_errors}')
            self.targets[frame] = pose.copy()
        # Move horizontally apart along the measured line between the wrists.
        # No box-specific shoulder angles or absolute scene coordinates.
        axis = self.targets[self.frames[0]].translation - self.targets[self.frames[1]].translation
        axis[2] = 0
        if np.linalg.norm(axis) < .1: raise ValueError('Wrist separation too small for outward direction')
        self.axis = axis / np.linalg.norm(axis)

    def full_q(self, measured):
        q = self.robot.q_default.copy()
        for i,name in enumerate(self.names): q[self.robot.dof_index(name)] = measured[i]
        return q

    def command(self, measured, root_pos, root_xyzw, previous, elapsed):
        # Build the reduced model at measured locked-joint values each step.
        # Native ReducedRobotModel builds at q0 even when fixed_values is passed.
        q = self.full_q(measured)
        model = pin.buildReducedModel(self.model, self.locked, q)
        names = list(model.names)[1:]
        reduced = np.array([q[self.robot.dof_index(n)] for n in names])
        config = pink.Configuration(model, model.createData(), reduced)
        root = pin.SE3(Rotation.from_quat(root_xyzw).as_matrix(), np.array(root_pos))
        tasks = []
        for sign,frame in zip([1,-1],self.frames):
            target = self.targets[frame].copy()
            if self.inward_m:
                # Close along the horizontal measured wrist axis at 2 mm/s.
                # Retain orientation and loaded finger targets; no force target.
                target.translation -= sign*self.axis*min(self.inward_m,.002*elapsed)
            elif self.translation is None:
                target.translation += sign*self.axis*min(.06,.03*elapsed)
            else:
                distance=np.linalg.norm(self.translation)
                target.translation += self.translation*min(1.,.01*elapsed/max(distance,1e-9))
            # PINK gain=1 requests deadbeat correction at 50 Hz. The position
            # servo has dynamics; compare a 0.1 gain (nominal 5/s error feedback).
            task = FrameTask(frame, position_cost=1., orientation_cost=.5, gain=0.1)
            task.set_target(root.inverse()*target)
            tasks.append(task)
        posture = PostureTask(cost=.01)
        posture.set_target_from_configuration(config)
        tasks.append(posture)
        velocity = pink.solve_ik(config,tasks,dt=.02,solver='osqp')
        self.last_velocity = velocity.tolist()
        output = np.array(previous).copy()
        for j,name in enumerate(names):
            i = self.names.index(name)
            lo,hi = model.lowerPositionLimit[j],model.upperPositionLimit[j]
            # Keep the loaded servo target; replacing it with measured angles
            # unloads the contact even when the requested wrist velocity is zero.
            delta=max(-.5,min(.5,float(velocity[j])))*.02
            output[i] = max(lo,min(hi,previous[i]+delta))
        if not np.isfinite(output).all(): raise ValueError('Nonfinite IK command')
        return output
