"""Gravity direction from calibrated depth and IMU; no heading/position claim.

The bootstrap assumes an initially upright robot (within 30 degrees) observing
a level plane. It does not use a scene normal. Gyro integration is corrected
slowly by near-1g specific force; this simple gate cannot reject all sustained
accelerations. Bias/noise are not calibrated. Long carrying trials must qualify
drift before this estimator is credited as reliable load-transfer feedback.
"""
import math
import numpy as np

from .rgbd_geometry import depth_points, fit_planes
from .arena_sensors import proprioception_packet


def _vector(value):
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError('expected three finite measurements')
    return v


def initial_up_from_depth(depth, intrinsic, camera_in_body):
    """Choose a dominant observed level-plane normal; no object masks required."""
    points = depth_points(depth, intrinsic)[::4,::4].reshape(-1,3)
    rotation = np.asarray(camera_in_body)[:3,:3]
    candidates = []
    for plane in fit_planes(points, max_planes=3):
        normal = rotation @ plane['normal']
        if normal[2] < 0:
            normal = -normal
        if normal[2] >= math.cos(math.radians(30)):
            candidates.append(normal)
    if not candidates:
        raise ValueError('no observed level-plane bootstrap')
    if any(np.dot(candidates[0], n) < math.cos(math.radians(5)) for n in candidates[1:]):
        raise ValueError('ambiguous observed level planes')
    return candidates[0]


class GravityEstimate:
    """Maintain body-frame up at monotonic IMU times; 5 s correction constant."""
    def __init__(self, up, time_s, gyro):
        self.up = _vector(up)
        if np.linalg.norm(self.up) < 1e-8 or not math.isfinite(time_s):
            raise ValueError('invalid gravity initialization')
        self.up = self.up / np.linalg.norm(self.up)
        self.time_s = float(time_s)
        self.gyro = _vector(gyro)
        self.accel_used = False

    def update(self, time_s, gyro, specific_force):
        dt = time_s - self.time_s
        if not math.isfinite(dt) or not 0 < dt <= .1:
            raise ValueError('IMU time must advance by at most 0.1 seconds')
        omega, accel = _vector(gyro), _vector(specific_force)
        # A world-fixed vector rotates opposite the body angular velocity.
        turn = -.5 * (self.gyro + omega) * dt
        angle = np.linalg.norm(turn)
        if angle > 1e-12:
            axis = turn / angle
            self.up = (self.up*math.cos(angle) + np.cross(axis,self.up)*math.sin(angle)
                       + axis*np.dot(axis,self.up)*(1-math.cos(angle)))
        magnitude = np.linalg.norm(accel)
        self.accel_used = bool(abs(magnitude-9.81) <= .981)
        if self.accel_used:
            blend = 1 - math.exp(-dt/5.)
            self.up = (1-blend)*self.up + blend*accel/magnitude
        self.up /= np.linalg.norm(self.up)
        self.time_s, self.gyro = float(time_s), omega
        return self.up.copy()


def up_quaternion(up):
    """Encode body up as WXYZ body-to-level rotation; absolute yaw is unobserved."""
    up = _vector(up)
    if np.linalg.norm(up) < 1e-8:
        raise ValueError('zero up vector')
    up = up/np.linalg.norm(up)
    if up[2] < -1 + 1e-10:
        return np.array([0.,1.,0.,0.])
    q = np.array([1+up[2],up[1],-up[0],0.])
    return q/np.linalg.norm(q)


def homie_observation(packet, model_joint_names, body_joint_names, up):
    """Only HOMIE's four consumed fields, assembled from permitted measurements.

    Finger slots, root translation and linear velocity are unused format padding,
    explicitly zero, not estimated measurements. This adapter is specific to the
    audited HOMIE vector, not a general native-policy observation implementation.
    """
    clean = proprioception_packet(step=packet['step'],time_s=packet['time_s'],
        joint_names=packet['joint_names'],q=packet['q_rad'],dq=packet['dq_rad_s'],
        tau_est=packet['tau_est_nm'],gyro=packet['gyro_rad_s'],
        accel=packet['specific_force_m_s2'])
    if set(packet) != set(clean) or packet['version'] != 1:
        raise ValueError('unexpected sensor packet fields or version')
    if (set(clean['joint_names']) != set(body_joint_names) or
            len(set(model_joint_names)) != len(model_joint_names)):
        raise ValueError('sensor/body joint names differ')
    q, dq = np.zeros((1,len(model_joint_names))), np.zeros((1,len(model_joint_names)))
    for name, position, velocity in zip(clean['joint_names'],clean['q_rad'],clean['dq_rad_s']):
        index = model_joint_names.index(name)
        q[0,index], dq[0,index] = position, velocity
    pose, velocity = np.zeros((1,7)), np.zeros((1,6))
    pose[0,3:] = up_quaternion(up)
    velocity[0,3:] = clean['gyro_rad_s']
    return dict(q=q,dq=dq,floating_base_pose=pose,floating_base_vel=velocity)
