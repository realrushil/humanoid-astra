"""Observed cuboid and candidate support geometry, in camera metres.

Masks identify one cuboid and potential support surfaces. Geometry thresholds
are simulation assumptions; outputs do not establish contact or full table
collision bounds. The support normal is checked against IMU-derived up rather
than box orientation, because a box can tilt independently of its support.
"""
import numpy as np
from .rgbd_geometry import depth_points,fit_planes


def _mask(value,depth):
    result=np.asarray(value)
    if result.dtype!=bool or result.ndim!=2 or result.shape!=np.asarray(depth).shape:
        raise ValueError('mask must be boolean and match the depth image')
    return result


def initialize_box(object_mask, support_mask, depth, intrinsic):
    """Initial complete cuboid hypothesis from an upright box on a planar support.

    Masks are supplied by the perception frontend. Missing extents stay unknown;
    visible-boundary coverage is an uncalibrated hypothesis test, not certainty.
    """
    mask=_mask(object_mask,depth)
    neutral=_mask(support_mask,depth)
    cloud = depth_points(depth, intrinsic)
    finite = np.isfinite(cloud).all(axis=2)
    obj = cloud[mask & finite]
    result = dict(status='unknown', reason='insufficient_surfaces',
                  center_camera_m=None, dimensions_m=None, axes_camera=None,
                  image_edge_clipping=bool(mask[0].any() or mask[-1].any() or
                                           mask[:, 0].any() or mask[:, -1].any()))
    planes = fit_planes(obj, max_planes=3)
    if len(planes) < 2:
        return result
    backgrounds = fit_planes(cloud[neutral & finite], max_planes=4)
    candidates = []
    for support in backgrounds:
        n, d = support['normal'].copy(), support['offset_m']
        if np.median(obj@n+d) < 0:
            n, d = -n, -d
        if abs(np.quantile(obj@n+d, .01)) > .015:
            continue
        for top in planes:
            if abs(top['normal']@n) < .99 or np.mean(top['points']@n+d) < .03:
                continue
            sides = [p for p in planes if abs(p['normal']@n) < .1]
            if sides:
                # A rotated cuboid often exposes two side faces. They are
                # alternative axis conventions, not different support surfaces.
                # Use the better-observed face without assuming a world yaw.
                front = max(sides, key=lambda p: len(p['points']))
                candidates.append((n, d, top, front))
    if len(candidates) != 1:
        result['reason'] = 'support_or_face_ambiguity'
        return result
    n, d, top, front = candidates[0]
    # Use the actual top normal for box axes; the support is used only to
    # estimate bottom height, under the explicit resting-box assumption.
    up = top['normal'].copy()
    if up@n < 0:
        up = -up
    toward = front['normal'] - (front['normal']@up)*up
    toward /= np.linalg.norm(toward)
    across = np.cross(toward, up)  # Right-handed: across x toward = up.
    axes = np.stack([across, toward, up], axis=1)
    top_pts, front_pts = top['points']@axes, front['points']@axes
    # The visible front plane provides the near footprint boundary; its
    # intersection with the top is more informative than top point-span alone.
    front_offset = np.median(front_pts[:, 1])
    far_offset = np.quantile(top_pts[:, 1], .001)
    side_points = np.concatenate([top_pts[:, 0], front_pts[:, 0]])
    left, right = np.quantile(side_points, [.001, .999])
    upper = np.median(top_pts[:, 2])
    height = np.mean(top['points']@n+d)
    lower = upper-height
    dims = np.array([right-left, front_offset-far_offset, height])
    center = axes@np.array([(right+left)/2, (front_offset+far_offset)/2, (upper+lower)/2])
    # Plane points alone cannot prove all silhouette edges were observed.
    # Check support for all four predicted top-face edges, including span
    # along each edge. This is a coverage heuristic, not occlusion certainty.
    bounds = [(left, right), (far_offset, front_offset)]
    edge_coverage = []
    for axis in [0, 1]:
        other = 1-axis
        for bound in bounds[axis]:
            near = top_pts[np.abs(top_pts[:, axis]-bound) < .008]
            coverage = np.ptp(near[:, other])/dims[other] if len(near) else 0.
            edge_coverage.append(float(coverage))
    result.update(visible_candidate_dimensions_m=dims.tolist(),
                  top_edge_coverage=edge_coverage)
    if result['image_edge_clipping']:
        result['reason'] = 'image_clipped'
    elif min(edge_coverage) < .75:
        result['reason'] = 'incomplete_top_edges'
    else:
        result.update(status='cuboid_candidate', reason='resting_box_and_edge_coverage_assumptions',
                      center_camera_m=center.tolist(), dimensions_m=dims.tolist(),
                      axes_camera=axes.tolist(),
                      uncertainty_note='No calibrated error bound; full-box hypothesis under stated assumptions')
    return result


def observed_support_gap(support_mask,depth,k,box,up_camera,*,planes=None):
    if box['status']!='accepted':return dict(status='unavailable',reason='box_pose_unavailable')
    neutral=_mask(support_mask,depth)
    if planes is None:
        points=depth_points(depth,k)
        planes=fit_planes(points[neutral],max_planes=4,min_points=200)
    axes=np.array(box['axes_camera']);up=np.asarray(up_camera,float)
    if up.shape!=(3,) or not np.isfinite(up).all() or np.linalg.norm(up)<1e-8:
        raise ValueError('a finite estimated gravity-up direction is required')
    up=up/np.linalg.norm(up)
    center=np.array(box['center_camera_m']);half=np.array(box['dimensions_m'])/2
    candidates=[]
    for plane in planes:
        n,d=plane['normal'].copy(),plane['offset_m']
        if n@up<0:n,d=-n,-d
        if n@up<np.cos(np.radians(20)):continue
        # Reject small hand/robot planar patches. This is an observed extent
        # threshold, not an assumption about the table's full collision bounds.
        x=axes[:,0]-(axes[:,0]@n)*n;x/=np.linalg.norm(x);y=np.cross(n,x)
        spans=np.ptp(plane['points']@np.stack([x,y],axis=1),axis=0)
        if min(spans)<.25:continue
        clearance=float(center@n+d-np.abs(axes.T@n)@half)
        if not -.02<=clearance<=.25:continue
        # Require observed plane samples around the projected box center.
        # Rectangular support hypothesis only; no load-bearing contact claim.
        local=(plane['points']-center)@np.stack([x,y],axis=1)
        if not np.all((local.min(axis=0)<-.02)&(local.max(axis=0)>.02)):continue
        candidates.append(dict(gap_m=clearance,normal_camera=n.tolist(),offset_m=float(d),
            observed_spans_m=spans.tolist(),rms_m=plane['rms_m']))
    if len(candidates)!=1:return dict(status='unavailable',reason='support_plane_ambiguity',candidates=len(candidates))
    return dict(status='observed_candidate',**candidates[0])
