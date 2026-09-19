"""Fresh source-table association under an explicit level-table assumption.

The table is parallel to the observed local floor. It supplies a height
reference and observed obstacle, not support underneath a carried box. The
co-moving floor/heading frame cannot identify finite or coplanar table bounds.
"""
import numpy as np

from .level_source_plane import box_corners, fit_level_plane
from .rgbd_geometry import fit_planes
from .scene_motion import _rigid


class SourcePlane:
    def __init__(self):
        self.plane = self.segment = self.last_time = None
        self.initialized = False
        self.geometry = None  # Fresh obstacle fit; independent of box-height precision.

    def observe(self, time_s, frame, camera_transform, box, initial, *, points, floor_points):
        """Associate current camera-time clouds; never reuse an old height.

        `initial` must be the current strict under-box support observation.
        Even that initial association must satisfy the level-plane model using
        fresh source/floor points. Later observations use their own prior plane.
        """
        self.geometry = None
        unavailable = dict(status='unavailable', reason='table_track_unavailable')
        if frame['status'] == 'unavailable':
            self.plane = None
            return unavailable
        if abs(frame['time_s'] - time_s) > 1e-8:
            raise ValueError('source plane requires synchronized control frame')
        camera = _rigid(frame['body_in_control_frame']) @ _rigid(camera_transform)
        if self.last_time is not None and not 0 < time_s - self.last_time <= .150001:
            self.plane = None
        if self.segment != frame['segment']:
            self.plane = None
        self.last_time, self.segment = time_s, frame['segment']
        initializing = self.plane is None
        if initializing:
            # Only an explicit new SourcePlane object can associate after an
            # epoch loss. Seeing another table does not preserve identity.
            if self.initialized or initial is None or initial['status'] != 'observed_candidate':
                return unavailable
            if box['status'] != 'accepted':
                return unavailable
            expected_normal = np.asarray(initial['normal_camera'])
            expected_offset = initial['offset_m']
        else:
            normal, offset = self.plane
            expected_normal = camera[:3, :3].T @ normal
            expected_offset = offset + normal @ camera[:3, 3]
        floor_points = np.asarray(floor_points, float)
        if (floor_points.ndim != 2 or floor_points.shape[1] != 3 or len(floor_points) < 200
                or not np.isfinite(floor_points).all()):
            self.plane = None
            return dict(unavailable, reason='floor_geometry_unavailable')
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1] != 3:
            return dict(unavailable, reason='source_geometry_unavailable')
        near = points[np.isfinite(points).all(axis=1) &
                      (abs(points @ expected_normal + expected_offset) < .03)]
        candidates = []
        for plane in fit_planes(near, max_planes=2, min_points=200):
            normal, offset = plane['normal'].copy(), plane['offset_m']
            if normal @ expected_normal < 0:
                normal, offset = -normal, -offset
            if normal @ expected_normal < np.cos(np.radians(5)) or abs(offset - expected_offset) > .03:
                continue
            queries = box_corners(box) if box['status'] == 'accepted' else plane['points'][::20]
            quality = fit_level_plane(floor_points, plane['points'], queries)
            if (quality['status'] != 'measured' or quality['source_residual_p90_m'] > .003
                    or quality['floor_normal_perturbation_deg'] > 5
                    or (initializing and quality['height_perturbation_m'] > .01)):
                continue
            normal = np.asarray(quality['normal'])
            offset = quality['offset_m']
            if normal @ expected_normal < 0:
                normal, offset = -normal, -offset
            if normal @ expected_normal < np.cos(np.radians(5)) or abs(offset - expected_offset) > .03:
                continue
            axis = np.eye(3)[np.argmin(abs(normal))]
            x = np.cross(normal, axis)
            x /= np.linalg.norm(x)
            spans = np.ptp(plane['points'] @ np.stack([x, np.cross(normal, x)], axis=1), axis=0)
            gap = None
            if box['status'] == 'accepted':
                gap = float(np.min(box_corners(box) @ normal + offset))
            candidates.append(dict(status='observed_candidate', normal_camera=normal.tolist(),
                offset_m=float(offset), gap_m=gap, observed_spans_m=spans.tolist(),
                rms_m=float(np.sqrt(np.mean((plane['points'] @ normal + offset) ** 2))),
                height_reference_only=True, prediction_quality=quality,
                identity='initial_under_box_plane' if initializing else 'fresh_floor_parallel_source_plane',
                match_angle_deg=float(np.degrees(np.arccos(np.clip(normal @ expected_normal, -1, 1)))),
                match_offset_m=float(abs(offset - expected_offset))))
        if len(candidates) != 1:
            return dict(unavailable, reason='table_plane_match_ambiguity', candidates=len(candidates))
        result = candidates[0]
        normal = camera[:3, :3] @ result['normal_camera']
        self.plane = normal, result['offset_m'] - normal @ camera[:3, 3]
        self.initialized = True
        # A bounded tabletop fit can remain useful for a particular arm path
        # even when its box-height query is too imprecise. Keep the uncertainty
        # with that evidence; do not label the failed height query available.
        self.geometry = dict(result,observed_at_s=time_s,height_reference_only=False)
        if result['prediction_quality']['height_perturbation_m'] > .01:
            return dict(unavailable,reason='source_height_precision_insufficient')
        return result
