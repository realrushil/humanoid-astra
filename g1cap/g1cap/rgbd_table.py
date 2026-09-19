"""Visible tabletop front from depth, a measured support plane and robot exclusion.

The strict detector accepts one facing segment at least 15 cm long. The
approach estimator may request shorter candidates for temporal association.
Thresholds (3 mm plane band, 4 mm line residual, span) are development
assumptions on ideal RGB-D, not measured hardware uncertainty or full extent.
"""
import numpy as np
from .rgbd_geometry import depth_points


def segments(points,normal,offset,min_span=.15):
    # Plane axes need no world heading; normal faces above the support.
    basis=np.eye(3)[np.argmin(np.abs(normal))]
    x=np.cross(normal,basis);x/=np.linalg.norm(x);y=np.cross(normal,x)
    axes=np.stack([x,y],axis=1);origin=-offset*normal
    xy=(points-origin)@axes;remaining=np.arange(len(xy));rng=np.random.default_rng(0);lines=[]
    for _ in range(6):
        if len(remaining)<30:break
        sample=xy[remaining];best=np.zeros(len(sample),bool)
        for _ in range(250):
            a,b=sample[rng.choice(len(sample),2,replace=False)];delta=b-a
            if np.linalg.norm(delta)<.05:continue
            n=np.array([-delta[1],delta[0]])/np.linalg.norm(delta)
            inside=np.abs((sample-a)@n)<.004
            if inside.sum()>best.sum():best=inside
        if best.sum()<30:break
        ids=remaining[best];pts=xy[ids];center=pts.mean(axis=0)
        _,_,vectors=np.linalg.svd(pts-center,full_matrices=False);direction=vectors[0]
        n=np.array([-direction[1],direction[0]])
        # Orient away from projected camera origin (the plane frame origin).
        if center@n<0:n=-n
        distance=float(center@n);along=(pts-center)@direction;limits=np.quantile(along,[.01,.99])
        span=float(limits[1]-limits[0]);remaining=remaining[~best]
        if span<min_span:continue
        normal3=axes@n;center3=origin+axes@center
        lines.append(dict(normal_camera=normal3.tolist(),offset_m=-float(normal3@center3),
            camera_to_line_m=distance,observed_span_m=span,points=len(ids),
            rms_m=float(np.sqrt(np.mean(((pts-center)@n)**2))),
            endpoints_camera_m=[(origin+axes@(center+v*direction)).tolist() for v in limits],
            boundary_point_indices=ids.tolist()))
    return lines


def front_candidates(depth,intrinsic,support,neutral,exclude_robot,min_span=.15):
    """Fresh facing segments; caller must resolve ambiguity before using one."""
    unavailable=dict(status='unavailable',reason='support_unavailable')
    if support is None or support['status']!='observed_candidate':return unavailable
    cloud=depth_points(depth,intrinsic)
    n=np.asarray(support['normal_camera'],float);d=float(support['offset_m'])
    if n.shape!=(3,) or not np.isfinite(n).all() or not np.isfinite(d) or abs(np.linalg.norm(n)-1)>.001:
        raise ValueError('invalid measured support plane')
    signed=cloud@n+d
    planar=np.isfinite(cloud).all(axis=2)&(abs(signed)<.003)&np.asarray(neutral,bool)
    planar[planar]=~exclude_robot(cloud[planar])
    if planar.sum()<30:return dict(unavailable,reason='insufficient_visible_support')
    boundary=np.zeros(depth.shape,bool)
    for dv,du in ((3,0),(-3,0),(0,3),(0,-3)):
        height=np.roll(signed,(dv,du),(0,1));neighbor=np.roll(depth,(dv,du),(0,1))
        # Select the last visible surface pixel, not all three rows of the
        # confirmation band. At coarse resolution those rows are centimetres
        # apart and would create several fictitious parallel front edges.
        edge=~np.roll(planar,(dv//3,du//3),(0,1))
        candidates=planar&edge&np.isfinite(height)&(height<-.005)&(neighbor>=depth-.005)
        candidates[:3]=False;candidates[-3:]=False;candidates[:,:3]=False;candidates[:,-3:]=False
        boundary|=candidates
    points=cloud[boundary]
    lines=segments(points,n,d,min_span) if len(points)>=30 else []
    centroid=np.median(cloud[planar],axis=0)
    facing=[line for line in lines if centroid@line['normal_camera']+line['offset_m']>.03]
    return dict(status='observed_candidates',lines=facing,planar_points=int(planar.sum()))


def front_boundary(depth,intrinsic,support,neutral,exclude_robot):
    """Strict independent-frame detector, including the 15 cm span threshold."""
    result=front_candidates(depth,intrinsic,support,neutral,exclude_robot)
    if result['status']=='unavailable':return result
    facing=result['lines']
    if len(facing)!=1:
        return dict(status='unavailable',reason='ambiguous_front' if facing else 'front_not_visible',candidate_count=len(facing))
    return dict(status='observed_candidate',line=facing[0],planar_points=result['planar_points'])
