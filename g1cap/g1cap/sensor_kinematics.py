"""Known robot kinematics from encoder feedback; no simulator world pose.

Uses the fixed-base G1 model as a coordinate convention, not a physical base
constraint. Results map points into the current pelvis frame (x forward, y left,
z up; metres). Missing joints on a requested frame's ancestry are errors.
Unrelated branches do not require invented finger or other sensor readings.
"""
import numpy as np
import pinocchio as pin


def frame_in_body(model, frame, joint_positions, *, body='pelvis'):
    """Return T_body_frame from named joint angles, without clipping feedback."""
    if any(j.nq != 1 for j in model.joints[1:]):
        raise ValueError('expected fixed-base model with one-coordinate G1 joints')
    for name in [frame, body]:
        if not model.existFrame(name):
            raise ValueError(f'unknown robot frame: {name}')
    required = set()
    for name in [frame, body]:
        joint_id = model.frames[model.getFrameId(name)].parentJoint
        while joint_id:
            required.add(model.names[joint_id])
            joint_id = model.parents[joint_id]
    missing = required - joint_positions.keys()
    if missing:
        raise ValueError(f'missing ancestor joint measurements: {sorted(missing)}')
    q = pin.neutral(model)
    for name, value in joint_positions.items():
        if not model.existJointName(name) or name == 'universe':
            raise ValueError(f'unknown measured joint: {name}')
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f'invalid measurement for {name}')
        q[model.joints[model.getJointId(name)].idx_q] = value
    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    body_pose = data.oMf[model.getFrameId(body)]
    frame_pose = data.oMf[model.getFrameId(frame)]
    return (body_pose.inverse() * frame_pose).homogeneous.copy()


def camera_in_body(model, joint_positions, calibration):
    """Return T_pelvis_camera for calibrated ROS optical (+X right,+Y down,+Z forward).

    The installed camera config stores the mounting quaternion as XYZW. Only
    small serialization norm error is normalized; other conventions are rejected
    so camera axes cannot change silently. Calibration is known robot geometry,
    not an observed/privileged camera pose in the world.
    """
    if calibration['offset_convention'] != 'ros':
        raise ValueError('expected a calibrated ROS optical camera frame')
    position = np.asarray(calibration['offset_position_m'], dtype=float)
    xyzw = np.asarray(calibration['offset_quaternion_xyzw'], dtype=float)
    if (position.shape != (3,) or xyzw.shape != (4,) or
            not np.isfinite(position).all() or not np.isfinite(xyzw).all() or
            abs(np.linalg.norm(xyzw)-1.) > .001):
        raise ValueError('invalid calibrated camera mounting transform')
    mounting = np.eye(4)
    mounting[:3,:3] = pin.Quaternion(xyzw/np.linalg.norm(xyzw)).matrix()
    mounting[:3,3] = position
    return frame_in_body(model, calibration['parent_frame'], joint_positions) @ mounting


def box_in_robot_frames(model, estimate, packet, calibration):
    """Express an accepted box pose relative to pelvis and both wrist frames.

    Encoder and camera times must match. Results belong to the robot frames at
    that observation time, not the current moving pelvis after a delayed frame.
    Positions are metres; this geometry never establishes contact or support.
    """
    from .arena_sensors import proprioception_packet
    clean=proprioception_packet(step=packet['step'],time_s=packet['time_s'],
        joint_names=packet['joint_names'],q=packet['q_rad'],dq=packet['dq_rad_s'],
        tau_est=packet['tau_est_nm'],gyro=packet['gyro_rad_s'],
        accel=packet['specific_force_m_s2'])
    if set(packet)!=set(clean) or packet['version']!=1:
        raise ValueError('unexpected sensor packet fields or version')
    unavailable=dict(status='unavailable',box_center_pelvis_m=None,
                     box_axes_pelvis=None,box_center_wrist_m=None)
    if estimate['status']!='accepted':
        return dict(unavailable,reason='unavailable_visual_pose')
    time_s=estimate['observed_at_s']
    if not np.isfinite(time_s) or abs(clean['time_s']-time_s)>1e-6:
        return dict(unavailable,reason='unsynchronized_encoders')
    joints=dict(zip(clean['joint_names'],clean['q_rad'],strict=True))
    camera=camera_in_body(model,joints,calibration)
    center=np.asarray(estimate['center_camera_m'],float)
    axes=np.asarray(estimate['axes_camera'],float)
    if (center.shape!=(3,) or axes.shape!=(3,3) or not np.isfinite(center).all()
            or not np.isfinite(axes).all() or not np.allclose(axes.T@axes,np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(axes),1.,atol=1e-5)):
        raise ValueError('invalid accepted visual pose')
    position=camera[:3,:3]@center+camera[:3,3]
    wrists={side:frame_in_body(model,f'{side}_wrist_yaw_link',joints) for side in ['left','right']}
    relative={side:(pose[:3,:3].T@(position-pose[:3,3])).tolist() for side,pose in wrists.items()}
    return dict(status='accepted',observed_at_s=time_s,track_epoch=estimate['track_epoch'],
                frame='pelvis_and_wrist_frames_at_observation_time',
                box_center_pelvis_m=position.tolist(),box_axes_pelvis=(camera[:3,:3]@axes).tolist(),
                box_center_wrist_m=relative,
                wrist_positions_pelvis_m={side:pose[:3,3].tolist() for side,pose in wrists.items()})
