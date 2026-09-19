"""Native joint-action adapter with sensor-only HOMIE feedback.

Retains the upstream action format, policy weights, pass-through upper targets
and sole actuator writer. Only process_actions is replaced because the upstream
method hardcodes prepare_observations to read simulator root poses/velocities.
With visual_grasp_checks, the outer pickup/hold toolkit also uses sensor inputs.
"""
from pathlib import Path
from copy import deepcopy
import numpy as np
import pinocchio as pin
import torch

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_joint_action import G1DecoupledWBCJointAction
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.run_policy import (
    convert_sim_joint_to_wbc_joint, postprocess_actions,
)

from .sensor_attitude import GravityEstimate, initial_up_from_depth, homie_observation
from .sensor_kinematics import camera_in_body


class SensorJointAction(G1DecoupledWBCJointAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.num_envs != 1 or cfg.wbc_version != 'homie_v2':
            raise ValueError('sensor action supports one audited HOMIE environment')
        self.sensor_source = None
        self.gravity = None
        self.last_sensor_input = None
        self.arm_gravity = None  # Optional calibrated robot-only servo compensation.
        self.sensor_model = pin.buildModelFromUrdf(str(
            Path(__file__).parent/'assets/arena_g1_rev1_0_kinematics.urdf'))
        self.body_names = [self.robot_model.joint_names[i]
                           for i in self.robot_model.get_joint_group_indices('body')]

    def up_at_measurement(self,packet,camera):
        """Read-only attitude prediction at the new camera/encoder time.

        Capture precedes the next action call. Advance a copy so perception
        cannot advance/reset the policy's persistent IMU integrator.
        """
        if self.gravity is None:
            if camera['step']!=packet['step']:
                raise ValueError('attitude bootstrap requires synchronized depth and encoders')
            joints=dict(zip(packet['joint_names'],packet['q_rad'],strict=True))
            transform=camera_in_body(self.sensor_model,joints,camera['calibration'])
            return initial_up_from_depth(camera['depth_m'],camera['calibration']['intrinsic'],transform)
        estimate=deepcopy(self.gravity)
        if packet['time_s']!=estimate.time_s:
            estimate.update(packet['time_s'],packet['gyro_rad_s'],packet['specific_force_m_s2'])
        return estimate.up.copy()

    def process_actions(self, actions):
        if self.sensor_source is None:
            raise ValueError('sensor action requires an explicit measurement source')
        measured = self.sensor_source()
        packet, camera = measured['proprioception'], measured['rgbd']
        if self.gravity is None:
            if camera['step'] != packet['step']:
                raise ValueError('attitude bootstrap requires synchronized depth and encoders')
            joints = dict(zip(packet['joint_names'],packet['q_rad'],strict=True))
            transform = camera_in_body(self.sensor_model,joints,camera['calibration'])
            up = initial_up_from_depth(camera['depth_m'],camera['calibration']['intrinsic'],transform)
            self.gravity = GravityEstimate(up,packet['time_s'],packet['gyro_rad_s'])
        else:
            self.gravity.update(packet['time_s'],packet['gyro_rad_s'],packet['specific_force_m_s2'])
        observation = homie_observation(packet,self.robot_model.joint_names,self.body_names,self.gravity.up)

        self._raw_actions[:] = actions[:,:self.action_dim]
        command = actions.clone()
        self.set_wbc_goal(self.get_navigation_cmd_from_actions(command),
                          self.get_base_height_cmd_from_actions(command),
                          self.get_torso_orientation_rpy_cmd_from_actions(command))
        self.wbc_policy.set_goal(self._wbc_goal)
        targets = convert_sim_joint_to_wbc_joint(command[:,:self._num_joints],
                    self._asset.data.joint_names,self.wbc_g1_joints_order)
        upper = targets[:,self.robot_model.get_joint_group_indices('upper_body')]
        self.wbc_policy.set_observation(observation)
        result = self.wbc_policy.get_action(upper)
        self._processed_actions = postprocess_actions(result,self._asset.data,
                                                       self.wbc_g1_joints_order,self.device)
        self.last_sensor_input = dict(step=packet['step'],time_s=packet['time_s'],
            up_body=self.gravity.up.tolist(),accel_correction_used=self.gravity.accel_used,
            observation={key:value.tolist() for key,value in observation.items()})
        if self.arm_gravity is not None:
            motor_targets=self.arm_gravity.command(packet,measured['hand_positions'],
                self.gravity.up.tolist(),self._processed_actions[0].cpu().numpy())
            self._processed_actions[0]=torch.as_tensor(motor_targets,device=self.device,dtype=self._processed_actions.dtype)
            self.last_sensor_input['arm_gravity']=self.arm_gravity.last
            # Record actual float32 motor targets, separately from the outer
            # 50-value command and the compensation's double-precision solve.
            self.last_sensor_input['motor_targets_rad']=self._processed_actions[0].cpu().numpy().tolist()
