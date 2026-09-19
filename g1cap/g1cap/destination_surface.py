"""Strict RGB-D observation of one finite destination-surface candidate.

The caller supplies a semantic/color candidate mask from the onboard image. This
module validates only the measured RGB-D geometry: one connected candidate,
finite depth, a single visible plane and a finite two-dimensional extent. It
does not infer a world pose, support/contact, destination route, or release
success. Optical coordinates use metres with ``x`` right, ``y`` down and ``z``
forward.
"""

import math

import numpy as np
from scipy.ndimage import label

from .rgbd_geometry import depth_points, fit_planes


def green_destination_mask(rgb):
    """Return a conservative semantic mask for the declared green destination.

    This is a color cue only; it does not encode a world pose or table bounds.
    The thresholds are deliberately strict and operate on uint8 RGB pixels.
    """
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('RGB must be uint8 HxWx3')
    red, green, blue = [image[..., i].astype(np.int16) for i in range(3)]
    return (green >= 45) & (green >= red + 12) & (green >= blue + 8)


def destination_direction_segment(candidate, camera_to_body, body_to_segment):
    """Transform a candidate ray into a normalized local-segment XY direction.

    ``camera_to_body`` and ``body_to_segment`` are calibrated 3x3 rotations;
    they are static/encoder-derived transforms, never simulator object poses.
    A near-vertical ray or malformed matrix is rejected rather than projected
    into navigation.  The returned direction has no range or world position.
    """
    if candidate.get('status') != 'observed_destination_candidate':
        raise ValueError('destination_observation_unavailable')
    ray = np.asarray(candidate.get('centroid_camera_m'), float)
    first = np.asarray(camera_to_body, float)
    second = np.asarray(body_to_segment, float)
    if ray.shape != (3,) or first.shape != (3, 3) or second.shape != (3, 3):
        raise ValueError('destination_direction_invalid')
    if not np.isfinite(ray).all() or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError('destination_direction_invalid')
    norm = np.linalg.norm(ray)
    if norm <= 1e-6:
        raise ValueError('destination_direction_invalid')
    direction = second @ (first @ (ray / norm))
    planar = direction[:2]
    planar_norm = np.linalg.norm(planar)
    if planar_norm < .5:
        raise ValueError('destination_direction_near_vertical')
    planar /= planar_norm
    return planar.tolist()


def observe_destination_surface(time_s, rgb, candidate_mask, depth_m, intrinsic):
    """Return one conservative finite surface candidate or ``unavailable``."""
    if isinstance(time_s, bool) or not isinstance(time_s, (int, float)) or not math.isfinite(time_s) or time_s < 0:
        raise ValueError('invalid observation time')
    image = np.asarray(rgb)
    mask = np.asarray(candidate_mask)
    depth = np.asarray(depth_m, float)
    k = np.asarray(intrinsic, float)
    if image.ndim != 3 or image.shape[2] != 3 or mask.shape != image.shape[:2] or mask.dtype != np.bool_:
        raise ValueError('RGB and candidate mask shape/type mismatch')
    if depth.shape != mask.shape or k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError('invalid depth or camera calibration')
    if not np.isfinite(image).all():
        raise ValueError('RGB contains nonfinite values')
    if not mask.any():
        return dict(status='unavailable', reason='destination_candidate_missing')
    components, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count != 1:
        return dict(status='unavailable', reason='destination_candidate_ambiguity', components=int(count))
    valid = mask & np.isfinite(depth) & (depth > 0)
    valid_fraction = float(valid.sum() / mask.sum())
    if valid.sum() < 200 or valid_fraction < .8:
        return dict(status='unavailable', reason='destination_depth_incomplete',
                    valid_points=int(valid.sum()), valid_fraction=valid_fraction)
    points = depth_points(depth, k)[valid]
    planes = fit_planes(points, threshold_m=.003, max_planes=1, min_points=200)
    if len(planes) != 1 or planes[0]['rms_m'] > .003:
        return dict(status='unavailable', reason='destination_plane_unavailable')
    plane = planes[0]
    normal = np.asarray(plane['normal'], float)
    # Build deterministic tangent axes from the camera optical basis.
    ref = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
    tangent = np.cross(normal, ref)
    tangent /= np.linalg.norm(tangent)
    bitangent = np.cross(normal, tangent)
    coordinates = points @ np.stack([tangent, bitangent], axis=1)
    spans = np.ptp(coordinates, axis=0)
    if np.any(spans < .15):
        return dict(status='unavailable', reason='destination_extent_insufficient',
                    observed_spans_m=spans.tolist())
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    centroid = points.mean(axis=0)
    return dict(status='observed_destination_candidate', observed_at_s=float(time_s),
                centroid_camera_m=centroid.tolist(),
                range_camera_m=float(np.linalg.norm(centroid)),
                normal_camera=normal.tolist(), offset_m=float(plane['offset_m']),
                tangent_camera=tangent.tolist(), bitangent_camera=bitangent.tolist(),
                extent_min_m=minimum.tolist(), extent_max_m=maximum.tolist(),
                observed_spans_m=spans.tolist(), finite_support=True,
                valid_points=int(valid.sum()), valid_fraction=valid_fraction,
                plane_rms_m=float(plane['rms_m']),
                association='single_connected_candidate_mask',
                frame='camera_optical_at_observation_time')


class DestinationSurfaceStream:
    """Keep one candidate only within a declared camera/view epoch."""

    def __init__(self, *, max_gap_s=.150001):
        if not math.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError('invalid destination stream gap')
        self.max_gap_s = float(max_gap_s)
        self.epoch = None
        self.last_time = None
        self.invalid = False

    def initialize(self, sample, *, view_epoch):
        if sample.get('status') != 'observed_destination_candidate':
            raise ValueError('destination_candidate_unavailable')
        if isinstance(view_epoch, bool) or not isinstance(view_epoch, int):
            raise ValueError('invalid destination view epoch')
        self.epoch = view_epoch
        self.last_time = float(sample['observed_at_s'])
        self.invalid = False
        return dict(sample, stream_status='initialized', view_epoch=view_epoch)

    def submit(self, sample, *, view_epoch):
        if self.invalid or self.epoch is None:
            raise ValueError('destination_stream_uninitialized')
        if view_epoch != self.epoch:
            self.invalid = True
            raise ValueError('destination_view_epoch_changed')
        if sample.get('status') != 'observed_destination_candidate':
            self.invalid = True
            raise ValueError('destination_candidate_unavailable')
        time_s = sample.get('observed_at_s')
        if (not isinstance(time_s, (int, float)) or isinstance(time_s, bool) or
                not math.isfinite(time_s) or not 0 < time_s - self.last_time <= self.max_gap_s):
            self.invalid = True
            raise ValueError('destination_observation_gap')
        self.last_time = float(time_s)
        return dict(sample, stream_status='tracked', view_epoch=view_epoch)
