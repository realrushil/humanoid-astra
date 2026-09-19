"""Visible metric geometry only; no object dimensions, identity or world pose.

Points and plane offsets use metres in the input camera's optical frame. Plane
fits describe observed surfaces, never a complete collision volume. The default
3 mm inlier tolerance is an experiment assumption, not sensor calibration.
"""
import numpy as np

from .arena_sensors import optical_point


def depth_points(depth_m, intrinsic):
    """Return H x W x 3 optical points; preserve invalid depth as three NaNs."""
    k = np.asarray(intrinsic, dtype=float)
    optical_point(0., 0., 1., k.tolist())  # Shared rectified-camera validation.
    depth = np.asarray(depth_m, dtype=float)
    if depth.ndim != 2:
        raise ValueError('depth must be a two-dimensional image')
    z = np.where(np.isfinite(depth) & (depth > 0), depth, np.nan)
    v, u = np.indices(z.shape)
    return np.stack(((u-k[0,2])*z/k[0,0], (v-k[1,2])*z/k[1,1], z), axis=-1)


def fit_planes(points, *, threshold_m=.003, max_planes=4, min_points=100, seed=0):
    """Bounded RANSAC followed by least-squares fits to visible point samples.

    Each result contains unit normal n, offset d (n.dot(p)+d=0), observed
    inlier points and RMS residual. Normals face the camera origin when possible;
    their sign is not gravity or semantic 'up'. Fit residual is not total sensor
    uncertainty. Nonfinite samples are ignored; insufficient/collinear data
    returns no plane. No hidden dimensions are supplied for missing surfaces.
    """
    remaining = np.asarray(points, dtype=float)
    if remaining.ndim != 2 or remaining.shape[1] != 3:
        raise ValueError('points must have shape N x 3')
    if (not np.isfinite(threshold_m) or threshold_m <= 0 or
            type(max_planes) is not int or max_planes < 1 or
            type(min_points) is not int or min_points < 3):
        raise ValueError('invalid plane fitting limits')
    remaining = remaining[np.isfinite(remaining).all(axis=1)]
    rng = np.random.default_rng(seed)
    planes = []
    for _ in range(max_planes):
        if len(remaining) < min_points:
            break
        sample = remaining[rng.choice(len(remaining), min(3000, len(remaining)), replace=False)]
        best_count, best = 0, None
        for _ in range(400):
            a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
            normal = np.cross(b-a, c-a)
            magnitude = np.linalg.norm(normal)
            if magnitude < 1e-10:
                continue
            normal /= magnitude
            offset = -normal @ a
            count = np.count_nonzero(np.abs(sample @ normal + offset) <= threshold_m)
            if count > best_count:
                best_count, best = count, (normal, offset)
        if best is None:
            break
        inliers = np.abs(remaining @ best[0] + best[1]) <= threshold_m
        if np.count_nonzero(inliers) < min_points:
            break
        selected = remaining[inliers]
        center = selected.mean(axis=0)
        _, singular, vectors = np.linalg.svd(selected-center, full_matrices=False)
        if singular[1] < 1e-8:  # A line does not identify a plane normal.
            break
        normal = vectors[-1]
        offset = -normal @ center
        if offset < 0:
            normal, offset = -normal, -offset
        # Recompute the final membership after refinement, not the seed fit.
        inliers = np.abs(remaining @ normal + offset) <= threshold_m
        selected = remaining[inliers]
        if len(selected) < min_points:
            break
        rms = float(np.sqrt(np.mean((selected @ normal + offset)**2)))
        planes.append(dict(normal=normal, offset_m=float(offset), points=selected, rms_m=rms))
        remaining = remaining[~inliers]
    return planes
