"""Fit a table parallel to fresh floor depth, in camera optical metres.

The sensitivity model allows correlated +/-3 mm displacement along the fitted
normal. Point membership, tangential coordinates, box pose and calibration are
held fixed. This is a simulation assumption, not total hardware uncertainty.
"""
from itertools import product
import numpy as np


def _core(points):
    points = np.asarray(points, float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 200
            or not np.isfinite(points).all()):
        return None
    distance = np.linalg.norm(points - np.median(points, axis=0), axis=1)
    return points[distance <= np.quantile(distance, .9)]


def box_corners(box):
    """Eight corners of the accepted camera-frame cuboid; no scene dimensions."""
    signs = np.asarray(list(product([-1, 1], repeat=3)))
    return (np.asarray(box['center_camera_m']) +
            (signs * np.asarray(box['dimensions_m']) / 2) @ np.asarray(box['axes_camera']).T)


def _hull(points):
    """Monotone-chain hull of measured 2D points, used only for sensitivity."""
    points = np.unique(points, axis=0)
    def half(sequence):
        result = []
        for point in sequence:
            while len(result) >= 2:
                a, b = result[-1] - result[-2], point - result[-1]
                if a[0]*b[1] - a[1]*b[0] > 0:
                    break
                result.pop()
            result.append(point)
        return result
    if len(points) <= 2:
        return points
    return np.asarray(half(points)[:-1] + half(points[::-1])[:-1])


def fit_level_plane(floor_points, source_points, queries, *, error_m=.003):
    """Fresh floor normal and source offset, with bounded fit sensitivity.

    Trimming the outer 10% prevents a few distant points from determining the
    footprint used for the sensitivity calculation. It does not identify a
    finite table or distinguish different coplanar surfaces.
    """
    floor, source = _core(floor_points), _core(source_points)
    queries = np.asarray(queries, float)
    if (floor is None or len(floor) < 200 or source is None or queries.ndim != 2
            or queries.shape[1] != 3 or len(queries) == 0 or not np.isfinite(queries).all()):
        return dict(status='unavailable', reason='insufficient_plane_geometry')
    if not np.isfinite(error_m) or error_m <= 0:
        raise ValueError('normal point-error assumption must be positive metres')
    center = floor.mean(axis=0)
    _, singular, axes = np.linalg.svd(floor - center, full_matrices=False)
    if singular[1] < 1e-8:
        return dict(status='unavailable', reason='floor_degenerate')
    normal = axes[2]
    # Equal in-plane singular values leave PCA axes arbitrary. Use a camera
    # axis to define reproducible tangent and coefficient coordinates.
    reference = np.array([1., 0., 0.]) if abs(normal[0]) < .9 else np.array([0., 1., 0.])
    tangent = np.cross(normal, reference)
    tangent /= np.linalg.norm(tangent)
    axes = np.stack([tangent, np.cross(normal, tangent), normal])
    inverse = np.linalg.pinv(np.c_[(floor - center) @ axes[:2].T, np.ones(len(floor))])
    coefficient_bound = error_m * np.abs(inverse[:2]).sum(axis=1)
    source_uv = (source - center) @ axes[:2].T
    query_uv = (queries - center) @ axes[:2].T
    vertices = _hull(source_uv)
    deltas = (query_uv[:, None, :] - vertices[None, :, :]).reshape(-1, 2)
    # Both slope coefficients depend on the SAME point-error vector. Its L1
    # influence preserves that correlation. Convexity lets the measured hull's
    # vertices bound every retained source point, without an axis-dependent
    # enclosing rectangle. Small batches bound working memory. No sqrt(N) gain.
    amplification = max(float(np.max(np.abs(deltas[i:i+16] @ inverse[:2]).sum(axis=1)))
                        for i in range(0, len(deltas), 16))
    height_bound = error_m * (1 + float(amplification))
    # The influence above is OLS. SVD/TLS instead uses (A-lambda I)^-1 b,
    # where A is tangential covariance and lambda its perturbed smallest
    # eigenvalue. Rayleigh bounds lambda by (normal RMS + error)^2. Bound the
    # extra slope explicitly; a nearly degenerate floor cannot be certified.
    minimum_tangent_variance = singular[1]**2 / len(floor)
    maximum_normal_variance = (singular[2] / np.sqrt(len(floor)) + error_m)**2
    if maximum_normal_variance >= minimum_tangent_variance:
        return dict(status='unavailable', reason='floor_ill_conditioned')
    ols_slope_bound = float(np.linalg.norm(coefficient_bound))
    tls_correction = (maximum_normal_variance /
                      (minimum_tangent_variance - maximum_normal_variance)) * ols_slope_bound
    height_bound += float(np.max(np.linalg.norm(deltas, axis=1))) * tls_correction
    slope_bound = ols_slope_bound + tls_correction
    offset = -float(np.median(source @ normal))
    # A tilted normal also changes normalization of signed plane distance.
    normalization_change = 1 - 1 / np.sqrt(1 + slope_bound**2)
    height_bound += float(np.max(abs(queries @ normal + offset))) * normalization_change
    residual = abs(source @ normal + offset)
    return dict(status='measured', normal=normal.tolist(), offset_m=offset,
        floor_core_points=len(floor), source_points=len(source),
        source_residual_p90_m=float(np.quantile(residual, .9)),
        floor_normal_perturbation_deg=float(np.degrees(np.arctan(slope_bound))),
        tls_slope_correction_bound=float(tls_correction),
        height_perturbation_m=height_bound, assumption_normal_point_error_m=error_m)


def floor_distances(points, queries, error_m=.003):
    """Signed distances to a fresh SVD floor fit, with conditional TLS bounds.

    Fixed point memberships/tangents and normal-point error are assumed. Pose,
    segmentation, calibration and changed floor elevation are not bounded here.
    Queries and points are in the same camera frame, in metres.
    """
    if not np.isfinite(error_m) or error_m<=0:
        raise ValueError('positive finite normal-point error required')
    floor = _core(points)
    queries = np.asarray(queries, float)
    if floor is None or len(floor)<200 or queries.ndim!=2 or queries.shape[1]!=3 or len(queries)==0 or not np.isfinite(queries).all():
        return dict(status='unavailable', reason='floor_geometry_unavailable')
    center=floor.mean(axis=0)
    _,singular,axes=np.linalg.svd(floor-center,full_matrices=False)
    amin=singular[1]**2/len(floor)
    cmax=(singular[2]/np.sqrt(len(floor))+error_m)**2
    if cmax>=amin:return dict(status='unavailable',reason='floor_ill_conditioned')
    normal=axes[2]
    reference=np.array([1.,0.,0.]) if abs(normal[0])<.9 else np.array([0.,1.,0.])
    tangent=np.cross(normal,reference);tangent/=np.linalg.norm(tangent)
    tangents=np.stack([tangent,np.cross(normal,tangent)])
    inverse=np.linalg.pinv(np.c_[(floor-center)@tangents.T,np.ones(len(floor))])
    coefficients=error_m*np.abs(inverse[:2]).sum(axis=1)
    C=float(np.linalg.norm(coefficients));delta=cmax/(amin-cmax)*C
    uv=(queries-center)@tangents.T;z=(queries-center)@normal
    bounds=error_m*np.abs(np.c_[uv,np.ones(len(uv))]@inverse).sum(axis=1)
    bounds+=np.linalg.norm(uv,axis=1)*delta
    bounds+=abs(z)*(1-1/np.sqrt(1+(C+delta)**2))
    return dict(status='measured',normal=normal.tolist(),center=center.tolist(),
        distances_m=z.tolist(),bounds_m=bounds.tolist(),normal_bound_deg=float(np.degrees(np.arctan(C+delta))))

