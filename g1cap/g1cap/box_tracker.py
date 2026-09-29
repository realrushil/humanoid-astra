"""Lean RGB-D tracking of the one brown box, for the v2 control track.

Reuses the existing cuboid initialisation (upright box on a visible support)
and the point-to-plane/silhouette registration, with the same acceptance gates
as the legacy frontend (BoxEstimateStream). It leaves out everything else the
legacy frontend did per frame (full-resolution plane fitting, robot-pixel
exclusion over the whole cloud, source-plane association, scene stability),
which cost 0.4-1.3 s per frame.

Colour association (brown = red > 1.2 green > 1.1 blue) is the same explicit
fixture assumption as before; it is not object recognition.
"""
import numpy as np

from .rgbd_box_geometry import initialize_box
from .rgbd_box_tracking import box_image_points, refine_box_pose
from .visual_box_state import BoxEstimateStream


def box_mask(rgb):
    c = np.asarray(rgb, float)
    return (c[..., 0] > 1.2 * c[..., 1]) & (c[..., 1] > 1.1 * c[..., 2])


class BoxTracker:
    def __init__(self):
        self.track = BoxEstimateStream()
        self.last_fit = None

    def update(self, camera):
        """Process one RGB-D frame; returns the accepted estimate or an unavailable record."""
        t, k = camera["time_s"], camera["calibration"]["intrinsic"]
        rgb, depth = np.asarray(camera["rgb"])[..., :3], camera["depth_m"]
        mask = box_mask(rgb)
        seed = self.track.prediction_seed(t) if self.track.epoch else None
        if seed is None:
            c = rgb.astype(float)
            neutral = (c.max(axis=2) - c.min(axis=2) < 25) & ~mask
            initial = initialize_box(mask, neutral, depth, k)
            self.last_fit = initial
            if initial["status"] == "cuboid_candidate":
                self.track.initialize(time_s=t, now_s=t, center=initial["center_camera_m"],
                                      rotation=initial["axes_camera"], size=initial["dimensions_m"])
        else:
            points, edges = box_image_points(mask, depth, k)
            center, rotation, fit = refine_box_pose(points, edges, seed["center_camera_m"],
                                                    seed["axes_camera"], seed["dimensions_m"])
            fit["total_face_points"] = len(points)
            self.last_fit = fit
            self.track.submit(time_s=t, now_s=t, center=center, rotation=rotation,
                              size=seed["dimensions_m"], quality=fit)
        return self.track.observe(t)
