"""Explicit measurement quantities for the RGB-D/proprioception track.

No world pose, object property, contact oracle or inferred grasp flag belongs in
this packet. Synthetic measurement provenance is recorded by the native adapter.
IMU vectors use pelvis-aligned sensor axes; camera points use optical +Z forward,
+X right, +Y down. Estimated torque is joint-axis N m, not external contact force.
"""
import math


def _finite(values, count, name):
    if not isinstance(values, (list, tuple)) or len(values) != count:
        raise ValueError(f'{name} must contain {count} measurements')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError(f'{name} contains invalid measurements')
    return list(values)


def proprioception_packet(*, step, time_s, joint_names, q, dq, tau_est, gyro, accel):
    """Copy a synchronized measurement sample; no implicit defaults for missing data."""
    if type(step) is not int or step < 0:
        raise ValueError('step must be a nonnegative integer')
    _finite([time_s], 1, 'time_s')
    if time_s < 0:
        raise ValueError('time_s must be nonnegative')
    if (not isinstance(joint_names, (list, tuple)) or not joint_names or
            any(not isinstance(n, str) or not n for n in joint_names) or
            len(set(joint_names)) != len(joint_names)):
        raise ValueError('joint_names must be nonempty and unique')
    count = len(joint_names)
    return dict(version=1, step=step, time_s=time_s, joint_names=list(joint_names),
                q_rad=_finite(q, count, 'q'), dq_rad_s=_finite(dq, count, 'dq'),
                tau_est_nm=_finite(tau_est, count, 'tau_est'),
                gyro_rad_s=_finite(gyro, 3, 'gyro'),
                specific_force_m_s2=_finite(accel, 3, 'accel'))


def optical_point(u, v, depth_m, intrinsic):
    """Unproject optical-axis depth into camera metres; invalid depth stays unknown."""
    if len(intrinsic) != 3:
        raise ValueError('camera intrinsic matrix must be 3 by 3')
    for row in intrinsic:
        _finite(row, 3, 'intrinsic row')
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    if (fx <= 0 or fy <= 0 or intrinsic[0][1] != 0 or intrinsic[1][0] != 0 or
            list(intrinsic[2]) != [0, 0, 1]):
        raise ValueError('expected rectified pinhole camera intrinsics')
    _finite([u, v], 2, 'pixel')
    if isinstance(depth_m, bool) or not isinstance(depth_m, (int, float)):
        raise ValueError('depth must be a number')
    if not math.isfinite(depth_m) or depth_m <= 0:
        return None
    return [(u-intrinsic[0][2])*depth_m/fx, (v-intrinsic[1][2])*depth_m/fy, depth_m]
