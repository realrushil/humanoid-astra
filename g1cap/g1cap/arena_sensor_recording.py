"""Native measurement recorder for onboard RGB-D and body proprioception.

By default this is diagnostic; sensor_balance also supplies its read-only packets
to HOMIE's observation adapter. Visual pickup/hold uses the same packets. Actuator
effort is an ideal motor-torque proxy, not calibrated external-contact feedback.
"""
import hashlib
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from .arena_sensors import proprioception_packet
from .hand_sensors import DEX3_JOINTS,hand_position_packet


def _values(value):
    return value.detach().cpu().numpy()


class SensorRecorder:
    def __init__(self, output, native, robot_model, *, sensor_balance=False,hand_positions=False,arm_gravity=False,
                 rgbd_period_steps=5):
        if type(rgbd_period_steps) is not int or rgbd_period_steps<=0:
            raise ValueError('rgbd_period_steps must be a positive integer')
        camera = native.scene['robot_head_cam']
        native_period_s = float(camera.cfg.update_period)
        # A zero native period means an update on every simulation step.
        capture_period_s=rgbd_period_steps*.02
        if (not math.isfinite(native_period_s) or native_period_s<0 or
                native_period_s>capture_period_s+1e-9 or
                (native_period_s>0 and abs(capture_period_s/native_period_s-
                                           round(capture_period_s/native_period_s))>1e-8)):
            raise ValueError('native camera update period does not support requested RGB-D capture period')
        self.rgbd_period_steps=rgbd_period_steps
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        self.native = native
        self.latest_packet = self.latest_rgbd = None
        self.latest_hands = None
        self.hand_stream = None
        robot = native.scene['robot']
        body = robot_model.get_joint_group_indices('body')
        self.names = [robot_model.joint_names[i] for i in body]
        if len(self.names) != 29:
            raise ValueError('expected the 29 body joints, excluding finger sensors')
        self.ids = [robot.joint_names.index(n) for n in self.names]
        # Static configuration, read once. This is not an estimated object load
        # or a contact measurement. The pinned native IdealPD API clips effort
        # against this per-joint tensor before publishing applied_effort.
        self.effort_limits={n:float(limit) for _,actuator in robot.actuators.items()
            for n,limit in zip(actuator.joint_names,_values(actuator.effort_limit)[0],strict=True)
            if n in self.names}
        self.body_stiffness={n:float(k) for _,actuator in robot.actuators.items()
            for n,k in zip(actuator.joint_names,_values(actuator.stiffness)[0],strict=True)
            if n in self.names}
        self.hand_ids = [robot.joint_names.index(n) for n in DEX3_JOINTS] if hand_positions else None
        self.intrinsic = _values(camera.data.intrinsic_matrices)[0].tolist()
        offset = camera.cfg.offset
        metadata = dict(version=1, track='diagnostic_ideal_rgbd_proprioception',
            controller_observations=('sensor pickup/hold and HOMIE' if hand_positions else
                'mixed: HOMIE sensor feedback; outer manipulation privileged' if sensor_balance else
                'unchanged_privileged_baseline'),
            body_effort_limits_nm=self.effort_limits,
            body_stiffness_nm_rad=self.body_stiffness,arm_gravity_compensation=arm_gravity,
            joint_names=self.names, joint_frame='URDF joint axes; SDK serial/parallel mapping not calibrated',
            torque_source='robot.actuators.applied_effort after clipping; synthetic motor-output proxy',
            actuator_classes={n:type(a).__name__ for n,a in robot.actuators.items()},
            imu_source='Isaac Imu sensor on pelvis, identity mounting',
            imu_frame='pelvis: x forward, y left, z up; accelerometer measures specific force',
            noise_model='ideal; zero added noise/bias/latency, not a hardware calibration',
            proprioception_hz=50, rgbd_hz=50/rgbd_period_steps,
            rgbd_period_steps=rgbd_period_steps,native_camera_update_period_s=native_period_s,
            rgbd_latency_assumption='ideal zero added capture and delivery latency; not hardware calibrated',
            camera=dict(parent_frame='head_link',width=camera.cfg.width,height=camera.cfg.height,
                intrinsic=self.intrinsic,offset_position_m=list(offset.pos),
                offset_quaternion_xyzw=list(offset.rot),offset_convention=offset.convention,
                depth_kind='distance_to_image_plane',depth_units='metres',
                optical_axes='x right, y down, z forward',invalid_depth='NaN',
                frame_storage='RGB PNG plus depth float32 NPY; no instance/semantic labels'),
            prohibited_outputs=['world_pose','box_mass','box_pose','contact_force','support_flag'])
        self.camera_calibration = metadata['camera']
        if hand_positions:
            metadata['hands']=dict(model='unitree_dex3_1',joint_names=list(DEX3_JOINTS),
                source='ideal joint-position encoder proxy; no commanded positions',
                frame='URDF joint axes, radians',rate_hz=50,
                assumption='user-approved simulation configuration; hardware calibration unconfirmed',
                excluded=['pressure','tactile','finger_torque','wrist_force_torque'])
            self.hand_stream=(self.output/'hands.jsonl').open('w')
        (self.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
        self.stream = (self.output/'proprioception.jsonl').open('w')
        self.frames = (self.output/'rgbd.jsonl').open('w')

    def capture(self, step, time_s):
        robot = self.native.scene['robot']
        imu = self.native.scene['body_imu'].data
        packet = proprioception_packet(step=step,time_s=time_s,joint_names=self.names,
            q=_values(robot.data.joint_pos)[0,self.ids].tolist(),
            dq=_values(robot.data.joint_vel)[0,self.ids].tolist(),
            tau_est=_values(robot.actuators.applied_effort)[0,self.ids].tolist(),
            gyro=_values(imu.ang_vel_b)[0].tolist(),accel=_values(imu.lin_acc_b)[0].tolist())
        self.latest_packet = packet
        self.stream.write(json.dumps(packet,allow_nan=False)+'\n');self.stream.flush()
        if self.hand_ids is not None:
            hand_q=_values(robot.data.joint_pos)[0,self.hand_ids].tolist()
            self.latest_hands=hand_position_packet(step=step,time_s=time_s,left=hand_q[:7],right=hand_q[7:])
            self.hand_stream.write(json.dumps(self.latest_hands,allow_nan=False)+'\n');self.hand_stream.flush()
        if step % self.rgbd_period_steps:
            return
        camera = self.native.scene['robot_head_cam']
        rgb = _values(camera.data.output['rgb'])[0,:,:,:3].astype(np.uint8)
        depth = _values(camera.data.output['distance_to_image_plane'])[0].squeeze(-1).astype(np.float32)
        if depth.shape != rgb.shape[:2]:
            raise ValueError('RGB/depth shapes differ')
        # Unknown pixels remain unknown; do not infer free space or nearby contact.
        depth[~np.isfinite(depth) | (depth<=0)] = np.nan
        self.latest_rgbd = dict(step=step,time_s=time_s,rgb=rgb,depth_m=depth,
                               calibration=self.camera_calibration)
        color_path=self.output/f'{step:06d}-rgb.png'
        depth_path=self.output/f'{step:06d}-depth.npy'
        imageio.imwrite(color_path,rgb)
        np.save(depth_path,depth,allow_pickle=False)
        frame=dict(step=step,time_s=time_s,width=rgb.shape[1],height=rgb.shape[0],
            rgb_file=color_path.name,depth_file=depth_path.name,
            rgb_sha256=hashlib.sha256(color_path.read_bytes()).hexdigest(),
            depth_sha256=hashlib.sha256(depth_path.read_bytes()).hexdigest(),
            valid_depth_fraction=float(np.isfinite(depth).mean()))
        self.frames.write(json.dumps(frame,allow_nan=False)+'\n');self.frames.flush()

    def measurements(self):
        """Owner-thread read-only view; holds RGB-D between configured captures."""
        if self.latest_packet is None or self.latest_rgbd is None:
            raise ValueError('no captured sensor measurements')
        result=dict(proprioception=self.latest_packet,rgbd=self.latest_rgbd)
        if self.latest_hands is not None:result['hand_positions']=self.latest_hands
        return result

    def close(self):
        self.stream.close()
        self.frames.close()
        if self.hand_stream is not None:self.hand_stream.close()
