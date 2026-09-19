"""Local RGB-D cuboid registration; no simulator state or fixed object dimensions.

A bounded robust point-to-plane fit uses projected silhouettes for tangential
pose constraints. Boundary depth selects valid rays; it is not fitted to a 3D
box edge. Silhouette residuals use modeled optical depth to convert normalized
image-normal error into approximate metres. Parameters are ideal-simulation assumptions: 3 mm robust loss,
4 cm correspondence radius, 12 iterations, and singular-value ratio 0.01 for
local rank. These diagnose local geometry, not calibrated pose uncertainty.
Object segmentation, initialization, time/identity checks and final acceptance
belong to the caller. Weakly constrained or lost fits must not imply success.
"""
import itertools
import numpy as np
from .rgbd_geometry import depth_points


def _rotation_increment(v):
    angle=np.linalg.norm(v)
    if angle<1e-12:return np.eye(3)
    x,y,z=v/angle;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    return np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*(k@k)


def box_image_points(object_mask, depth_m, intrinsic):
    """Return bounded interior/silhouette samples from an associated object mask.

    Mask identity is the caller's responsibility; no color or scene labels live
    here. Unknown depths and image borders do not imply a measured silhouette.
    The 2 cm depth-jump threshold is an uncalibrated experiment assumption.
    """
    mask=np.asarray(object_mask)
    depth=np.asarray(depth_m,float);k=intrinsic
    if mask.dtype!=bool or mask.ndim!=2 or mask.shape!=depth.shape:
        raise ValueError('object mask must be boolean and match the depth image')
    cloud=depth_points(depth,k);valid=np.isfinite(cloud).all(axis=2)
    interior=mask&valid
    # Interior samples avoid mixed RGB/depth boundary pixels. Keep four-neighbor
    # silhouette candidates only when the neighboring depth is farther away.
    # Nearer non-object surfaces are occluders, not the box's own silhouette.
    boundary=np.zeros(mask.shape,bool)
    for axis in [0,1]:
        for shift in [-2,2]:
            neighbor=np.roll(mask,shift,axis=axis)
            other_depth=np.roll(depth,shift,axis=axis)
            good=np.roll(valid,shift,axis=axis)
            hit=mask&valid&~neighbor&good&(other_depth-depth>.02)
            if axis==0:
                hit[:2]=False;hit[-2:]=False
            else:
                hit[:,:2]=False;hit[:,-2:]=False
            boundary|=hit
            interior &= neighbor
    points=cloud[interior];edges=cloud[boundary]
    # Deterministic bounded samples keep registration cheap and reproducible.
    if len(points)>1200:points=points[np.linspace(0,len(points)-1,1200,dtype=int)]
    if len(edges)>300:edges=edges[np.linspace(0,len(edges)-1,300,dtype=int)]
    return points,edges


def _constraints(points,edges,center,rotation,size,*,use_edges=True):
    half=np.asarray(size)/2
    p=(points-center)@rotation
    # Nearest finite box face chooses correspondence; residual is the plane
    # distance only, so an interior face does not fake in-plane observability.
    candidates=[]
    for axis in range(3):
        for sign in [-1,1]:
            q=np.clip(p,-half,half);q[:,axis]=sign*half[axis]
            candidates.append(np.linalg.norm(p-q,axis=1))
    distance=np.stack(candidates,axis=1);choice=distance.argmin(axis=1)
    axis=choice//2;sign=np.where(choice%2,1.,-1.)
    n=np.eye(3)[axis]*sign[:,None]
    residual=np.sum(p*n,axis=1)-half[axis]
    keep=distance.min(axis=1)<.04
    A=np.concatenate([n,np.cross(p,n)],axis=1)[keep];b=residual[keep]
    face_rows=len(b)
    edge_count=0
    if use_edges and len(edges):
        # A contour pixel constrains a camera ray, not a 3D edge point. Near a
        # grazing face, even one pixel of inward sampling changes depth by cm.
        # Depth still selects valid object/background boundaries in extraction.
        edges=edges[edges[:,2]>1e-8]
        rays=edges[:,:2]/edges[:,2,None]
        corners=np.array(list(itertools.product([-1,1],repeat=3)))*half
        optical=corners@rotation.T+center
        choices=[];geometry=[]
        camera=-center@rotation
        for i,j in itertools.combinations(range(8),2):
            diff=corners[j]-corners[i]
            if np.count_nonzero(diff)!=1 or min(optical[i,2],optical[j,2])<=1e-8:continue
            axis=int(np.argmax(abs(diff)));other=[k for k in range(3) if k!=axis]
            facing=[np.sign(corners[i,k])*(camera[k]-corners[i,k])>0 for k in other]
            if facing[0]==facing[1]:continue
            u0=optical[i,:2]/optical[i,2];u1=optical[j,:2]/optical[j,2]
            direction=u1-u0;length=np.linalg.norm(direction)
            if length<1e-8:continue
            fraction=np.clip((rays-u0)@direction/(length*length),0,1)
            projected=u0+fraction[:,None]*direction
            # Perspective-correct interpolation recovers the corresponding 3D
            # point on the modeled edge from its projected segment fraction.
            along=fraction*optical[i,2]/((1-fraction)*optical[j,2]+fraction*optical[i,2])
            point=corners[i]+along[:,None]*diff
            z=(point@rotation.T+center)[:,2]
            choices.append(np.linalg.norm(rays-projected,axis=1)*z)
            normal=np.array([-direction[1],direction[0]])/length
            geometry.append((point,z,projected,normal))
        if choices:
            distances=np.stack(choices,axis=1);closest=distances.argmin(axis=1)
            for index,(point,z,projected,normal) in enumerate(geometry):
                pick=(closest==index)&(distances[:,index]<.04)
                if not np.any(pick):continue
                # One image-normal row per pixel leaves the unobserved tangent
                # free. Multiplying normalized-image error by modeled optical
                # depth expresses the local residual in approximate metres.
                selected=point[pick];uv=projected[pick]
                n=np.column_stack([np.full(len(uv),normal[0]),np.full(len(uv),normal[1]),-uv@normal])@rotation
                A=np.concatenate([A,np.concatenate([n,np.cross(selected,n)],axis=1)])
                b=np.concatenate([b,z[pick]*((rays[pick]-uv)@normal)])
                edge_count+=int(pick.sum())
    return A,b,dict(face_rows=face_rows,edge_points=edge_count)


def refine_box_pose(points, edges, center, rotation, size, *, use_edges=True):
    """Refine an existing cuboid hypothesis against current optical-frame points.

    Input center/size use metres; rotation maps right-handed box axes into the
    camera frame. Size must come from the caller's prior observation, not a hidden
    nominal fallback. This local fit does not establish object identity, contact,
    global recovery or statistical certainty. Return (center, rotation, quality).
    Partial rank retains unmeasured prior components; callers must honor status.
    """
    points=np.asarray(points,float);edges=np.asarray(edges,float)
    center=np.array(center,float).copy();rotation=np.array(rotation,float).copy()
    size=np.asarray(size,float)
    if (points.ndim!=2 or points.shape[1]!=3 or edges.ndim!=2 or edges.shape[1]!=3
            or center.shape!=(3,) or rotation.shape!=(3,3) or size.shape!=(3,)
            or not all(np.isfinite(x).all() for x in [points,edges,center,rotation,size])
            or np.any(size<=0)):
        raise ValueError('invalid cuboid pose, dimensions or measured points')
    if (not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(rotation),1.,atol=1e-5)):
        raise ValueError('box rotation must be orthonormal and right-handed')
    if len(points)<100:return center,rotation,dict(status='lost',reason='insufficient_depth')
    # Scaled rotation columns compare rotation-induced surface displacement
    # against translation; eigenvalues are geometry diagnostics, not covariance.
    radius=max(np.linalg.norm(size)/2,.01)
    for _ in range(12):
        A,b,counts=_constraints(points,edges,center,rotation,size,use_edges=use_edges)
        if len(b)<100:return center,rotation,dict(status='lost',reason='insufficient_correspondences')
        weight=np.sqrt(np.minimum(1.,.003/np.maximum(abs(b),1e-9)))
        scaled=A.copy();scaled[:,3:]/=radius
        delta=np.linalg.lstsq(scaled*weight[:,None],b*weight,rcond=.003)[0]
        delta[3:]/=radius
        # Trust bounds apply per iteration. Whole-update acceptance is separate.
        delta[:3]*=min(1.,.015/max(np.linalg.norm(delta[:3]),1e-12))
        delta[3:]*=min(1.,.12/max(np.linalg.norm(delta[3:]),1e-12))
        center+=rotation@delta[:3];rotation=rotation@_rotation_increment(delta[3:])
        if np.linalg.norm(delta)<1e-6:break
    A,b,counts=_constraints(points,edges,center,rotation,size,use_edges=use_edges)
    scaled=A.copy();scaled[:,3:]/=radius
    singular=np.linalg.svd(scaled,compute_uv=False)
    ratios=singular/singular[0] if len(singular) and singular[0]>0 else np.zeros(6)
    rank=int(np.count_nonzero(ratios>.01))
    return center,rotation,dict(status='full_rank_candidate' if rank==6 else 'partially_constrained',
        rms_m=float(np.sqrt(np.mean(b*b))),constraint_rank=rank,
        relative_singular_values=ratios.tolist(),**counts)

