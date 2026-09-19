"""Floor-relative height/tilt and gyro heading for horizontally co-moving arms.

This control frame omits XY translation by definition. It is not a global or
stance position estimate and must not be published as one. Floor offset is the
pelvis height above a locally level floor, in metres. Heading is gyro-derived.
"""
import math
import numpy as np

from .scene_motion import _rigid, _vector


class FloorCarryFrame:
    def __init__(self):
        self.up = self.height = self.time = None
        self.segment = 0
        self.camera_transform = self.camera_gyro = None

    def update(self, time_s, gyro_rotation, floor):
        if not np.isfinite(time_s) or (self.time is not None and time_s <= self.time):
            raise ValueError('floor frame time must be finite and increase')
        if self.time is not None and time_s - self.time > .150001:
            self.up = None
        self.time = time_s
        self.camera_transform = self.camera_gyro = None
        unavailable = dict(status='unavailable', time_s=time_s, segment=self.segment,
            body_in_control_frame=None, frame='horizontally_co_moving_height_heading',
            horizontal_position_observed=False)
        if floor['status'] != 'observed_candidate':
            self.up = None
            return unavailable
        normal = _vector(floor['normal_body'])
        height = float(floor['offset_m'])
        if not np.isclose(np.linalg.norm(normal), 1., atol=1e-5) or not np.isfinite(height) or height <= 0:
            raise ValueError('invalid floor measurement')
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(gyro_rotation, float)
        _rigid(transform)
        if self.up is None:
            self.up = transform[:3, :3] @ normal
            self.height = height
            self.segment += 1
        forward_world = transform[:3, 0].copy()
        forward_world -= self.up * (self.up @ forward_world)
        forward_body = np.array([1., 0., 0.])
        forward_body -= normal * (normal @ forward_body)
        if min(np.linalg.norm(forward_world), np.linalg.norm(forward_body)) < .5:
            raise ValueError('invalid floor heading')
        forward_world /= np.linalg.norm(forward_world)
        forward_body /= np.linalg.norm(forward_body)
        world_basis = np.column_stack([forward_world, np.cross(self.up, forward_world), self.up])
        body_basis = np.column_stack([forward_body, np.cross(normal, forward_body), normal])
        transform[:3, :3] = world_basis @ body_basis.T
        transform[:3, 3] = self.up * (height - self.height)
        self.camera_transform=transform.copy()
        self.camera_gyro=np.asarray(gyro_rotation,float).copy()
        return dict(status='observed_floor_heading_frame', time_s=time_s, segment=self.segment,
            body_in_control_frame=transform.tolist(), frame='horizontally_co_moving_height_heading',
            horizontal_position_observed=False, height_change_m=height-self.height)

    def at_imu(self,time_s,gyro_rotation):
        """Internal 50 Hz control pose; camera height is held, never refreshed.

        Reuse the existing IMU integrator relative to its camera-time rotation.
        This omits horizontal translation and holds depth-derived height for at
        most 150 ms. Neither assumption is hardware calibration or odometry.
        The caller must pair this rotation with the current encoder timestamp.
        """
        if (type(time_s) not in (int,float) or not math.isfinite(time_s)
                or self.camera_transform is None or not 0<=time_s-self.time<=.150001):
            raise ValueError('floor_control_frame_unavailable')
        gyro=np.eye(4);gyro[:3,:3]=np.asarray(gyro_rotation,float);_rigid(gyro)
        transform=self.camera_transform.copy()
        if time_s!=self.time:
            transform[:3,:3]=transform[:3,:3]@self.camera_gyro.T@gyro[:3,:3]
        return dict(status='gyro_propagated_floor_frame',time_s=time_s,segment=self.segment,
            body_in_control_frame=transform.tolist(),observed_at_s=self.time,
            height_age_s=time_s-self.time,
            assumption='gyro orientation; camera height held; no horizontal localization')
