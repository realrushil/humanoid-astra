"""Finite-pixel evidence for an explicit proposed tabletop footprint.

This internal, unwired check does not choose a target or authorize motion.
It consumes one synchronized head RGB-D sample and its observed plane only.
"""
import math
import numpy as np
from scipy.ndimage import binary_erosion
from .destination_surface import green_destination_mask
from .rgbd_geometry import depth_points


def observe_footprint(*, candidate, rgb, depth_m, intrinsic, corners_camera_m,
                      image_time_s, now_s, erosion_pixels):
    """Check four ordered convex footprint corners on the observed plane.

    Corners are metres in the capture-time camera optical frame (right, down,
    forward). The caller must enlarge the footprint for declared uncertainty;
    this function neither estimates uncertainty nor assumes a box orientation.
    Every closed pixel cell intersecting the projected polygon must have green,
    positive finite depth within 3 mm of the plane after square-mask erosion.
    Unknown cells are never filled. 150 ms freshness, 3 mm residual and supplied
    pixel erosion are development assumptions, not hardware calibration.

    A positive result is finite-resolution observation only: no subpixel
    continuity, load support, collision clearance, reachability or release proof.
    The caller is responsible for passing the image identified by image_time_s.
    """
    def unknown(reason):
        return dict(status='unknown', reason=reason, placement_authorized=False)

    if not isinstance(candidate, dict) or candidate.get('status') != 'observed_destination_candidate':
        return unknown('destination_unavailable')
    stamp = candidate.get('observed_at_s')
    clocks = (stamp, image_time_s, now_s)
    if any(isinstance(t, bool) or not isinstance(t, (int, float)) or
           not math.isfinite(t) or t < 0 for t in clocks):
        return unknown('invalid_time')
    if stamp != image_time_s or not 0 <= now_s-stamp <= .15+1e-9:
        return unknown('unsynchronized_or_stale_image')
    if isinstance(erosion_pixels, bool) or not isinstance(erosion_pixels, int) or not 0 <= erosion_pixels <= 8:
        return unknown('invalid_erosion')
    try:
        depth = np.asarray(depth_m, float)
        k = np.asarray(intrinsic, float)
        corners = np.asarray(corners_camera_m, float)
        normal = np.asarray(candidate['normal_camera'], float)
        offset = float(candidate['offset_m'])
        mask = green_destination_mask(rgb)
        cloud = depth_points(depth, k)  # Validates rectified pinhole calibration.
    except (ValueError, TypeError, KeyError, IndexError):
        return unknown('invalid_geometry_input')
    if depth.shape != mask.shape or normal.shape != (3,) or corners.shape != (4, 3):
        return unknown('invalid_geometry_shape')
    if not np.isfinite(corners).all() or not np.isfinite(normal).all() or not math.isfinite(offset):
        return unknown('nonfinite_geometry')
    if abs(np.linalg.norm(normal)-1) > 1e-6:
        return unknown('invalid_plane_normal')
    if np.any(corners[:, 2] <= 0) or np.max(abs(corners@normal+offset)) > 1e-6:
        return unknown('footprint_not_on_visible_plane')
    edges = np.roll(corners, -1, axis=0)-corners
    turns = np.cross(edges, np.roll(edges, -1, axis=0))@normal
    if not (np.all(turns > 1e-10) or np.all(turns < -1e-10)):
        return unknown('footprint_not_strictly_convex')
    projected = corners@k.T
    polygon = projected[:, :2]/projected[:, 2, None]
    # Reject clipping before allocating the bounded image-cell grid.
    lower = polygon.min(axis=0)-.5
    upper = polygon.max(axis=0)+.5
    height, width = depth.shape
    if np.any(lower <= -1) or np.any(upper >= [width, height]):
        return unknown('footprint_outside_image')
    lo = np.ceil(lower).astype(int)
    hi = np.floor(upper).astype(int)
    x, y = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
    centers = np.c_[x.ravel(), y.ravel()]
    edges = np.roll(polygon, -1, axis=0)-polygon
    axes = np.r_[np.eye(2), np.c_[-edges[:, 1], edges[:, 0]]]
    touched = np.ones(len(centers), bool)
    for axis in axes:
        values = polygon@axis
        middle = centers@axis
        radius = .5*np.abs(axis).sum()
        touched &= (middle+radius >= values.min()-1e-10) & (middle-radius <= values.max()+1e-10)
    cells = centers[touched]
    mask &= np.isfinite(depth) & (depth > 0) & (abs(cloud@normal+offset) <= .003)
    if erosion_pixels:
        mask = binary_erosion(mask, structure=np.ones((3, 3)),
                              iterations=erosion_pixels, border_value=0)
    known = mask[cells[:, 1], cells[:, 0]]
    passed = bool(len(cells) and known.all())
    return dict(status='observed_footprint' if passed else 'unknown',
                reason=None if passed else 'footprint_contains_unknown_pixels',
                observed_at_s=stamp, age_s=now_s-stamp,
                frame='camera_optical_at_observation_time',
                tested_pixels=len(cells), unknown_pixels=int((~known).sum()),
                erosion_pixels=erosion_pixels, plane_residual_limit_m=.003,
                placement_authorized=False)
