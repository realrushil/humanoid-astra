"""A bounded remembered source-plane observation for local transport.

The plane is measured once in an RGB-D frame and may be queried only inside
the same uninterrupted :class:`StanceMotion` segment. Distances are metres in
the segment frame, positive on the free side. The uncertainty arguments are
explicit sensitivity assumptions; they are not hardware calibration and this
class does not provide navigation permission or a complete collision map.
"""

import numpy as np

from .scene_motion import _rigid


class RememberedSourcePlane:
    """Remember one measured source-front half-space with latched loss."""

    def __init__(self, time_s, motion, normal_body, offset_body, anchor_body,
                 *, plane_error_m, plane_angle_rad, max_age_s):
        if motion.get('status') != 'tracked_local_segment' or abs(motion['time_s'] - time_s) > 1e-8:
            raise ValueError('initial motion unavailable')
        transform = _rigid(motion['body_in_segment'])
        normal = np.asarray(normal_body, float)
        anchor = np.asarray(anchor_body, float)
        values = np.r_[time_s, normal, offset_body, anchor,
                       plane_error_m, plane_angle_rad, max_age_s]
        if (normal.shape != (3,) or anchor.shape != (3,) or
                not np.isfinite(values).all() or abs(np.linalg.norm(normal) - 1) > 1e-6 or
                min(time_s, plane_error_m, plane_angle_rad) < 0 or
                max_age_s <= 0 or plane_angle_rad > np.pi or
                abs(normal @ anchor + offset_body) > 1e-5):
            raise ValueError('invalid measured plane')
        self.invalid = False
        self.time = float(time_s)
        self.segment = motion['segment']
        self.max_age = float(max_age_s)
        self.normal = transform[:3, :3] @ normal
        self.anchor = transform[:3, :3] @ anchor + transform[:3, 3]
        self.error = float(plane_error_m)
        self.angle = float(plane_angle_rad)

    def query(self, now_s, motion, points_body, *, translation_error_m,
              rotation_error_rad):
        """Return conditional distances, or latch unavailable on continuity loss."""
        values = np.r_[now_s, translation_error_m, rotation_error_rad]
        if (not np.isfinite(values).all() or min(translation_error_m, rotation_error_rad) < 0 or
                rotation_error_rad > np.pi):
            raise ValueError('invalid uncertainty')
        if (self.invalid or not 0 <= now_s - self.time <= self.max_age or
                motion.get('status') != 'tracked_local_segment' or
                motion.get('segment') != self.segment or
                abs(motion['time_s'] - now_s) > 1e-8):
            self.invalid = True
            raise ValueError('source memory continuity unavailable')
        transform = _rigid(motion['body_in_segment'])
        points = np.asarray(points_body, float)
        if (points.ndim != 2 or points.shape[1] != 3 or not len(points) or
                not np.isfinite(points).all()):
            raise ValueError('invalid points')
        mapped = points @ transform[:3, :3].T + transform[:3, 3]
        nominal = (mapped - self.anchor) @ self.normal
        pose_error = (translation_error_m +
                      2 * np.sin(rotation_error_rad / 2) * np.linalg.norm(points, axis=1))
        plane_error = (self.error +
                       2 * np.sin(self.angle / 2) * np.linalg.norm(mapped - self.anchor, axis=1))
        uncertainty = pose_error + plane_error
        return dict(status='remembered_conditional_plane', observed_at_s=self.time,
                    transform_at_s=float(now_s), age_s=float(now_s - self.time),
                    segment=self.segment, nominal_m=nominal.tolist(),
                    lower_m=(nominal - uncertainty).tolist(),
                    uncertainty_m=uncertainty.tolist(),
                    fresh_obstacle_observation=False)
