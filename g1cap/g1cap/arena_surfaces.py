"""Measured collision bounds and selected-support observations.

World metres, WXYZ box/root and XYZW link rotations. Static support boxes are
world AABBs. These sampled bounds do not certify a continuous swept volume.
"""
import itertools,math
from g1cap.toolkit.arena_approach import rotate_xyzw,GUARDED_BODIES


def transformed_bounds(shape,pose):
    points=[]
    for point in itertools.product(*zip(shape['min'],shape['max'])):
        r=rotate_xyzw(pose['xyzw'],point)
        points.append([r[i]+pose['pos'][i] for i in range(3)])
    return {'min':[min(p[i] for p in points) for i in range(3)],
            'max':[max(p[i] for p in points) for i in range(3)]}


def box_bounds(row):
    # Old recordings omit dimensions and belong to the original20cm fixture.
    size=row.get('box_size_m',[.2,.2,.2])
    if (not isinstance(size,(list,tuple)) or len(size)!=3 or
        any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0 for v in size)):
        raise ValueError('box_size_m must contain three positive finite full dimensions')
    w,x,y,z=row['box_quat']
    return transformed_bounds(dict(min=[-v/2 for v in size],max=[v/2 for v in size]),
                              dict(pos=row['box_pos'],xyzw=[x,y,z,w]))


def contained(box,surface,margin=0.):
    return all(surface['min'][i]+margin<=box['min'][i] and box['max'][i]<=surface['max'][i]-margin for i in range(2))


def separation(a,b):
    return math.sqrt(sum(max(0.,a['min'][i]-b['max'][i],b['min'][i]-a['max'][i])**2 for i in range(3)))


def lower_body_clearance(geometry,row,surfaces):
    distances=[]
    for name in GUARDED_BODIES:
        pose=row['body_poses'][name]
        for shape in geometry['robot'][name]:
            bound=transformed_bounds(shape,pose)
            distances.extend(separation(bound,s) for s in surfaces)
    if not distances:raise ValueError('missing lower-body geometry')
    return min(distances)



def support_observation(box, surface, upward_N, robot_peak_N, lower_clearance_m, *, mass_kg=.1):
    """Configured object weight; support force is simulator normal force.

    Support requires at least half its weight and height within1cm. This is a
    current observation, not task success; final release needs a stable window.
    """
    if isinstance(mass_kg,bool) or not isinstance(mass_kg,(int,float)) or not math.isfinite(mass_kg) or mass_kg<=0:
        raise ValueError('positive finite object mass required')
    clearance=box['min'][2]-surface['max'][2]
    return dict(clearance_m=clearance,box_upward_N=upward_N,
                robot_contact_peak_N=robot_peak_N,
                supported=upward_N>=.5*mass_kg*9.81 and abs(clearance)<=.01,
                contained=contained(box,surface),lower_body_clearance_m=lower_clearance_m,
                bounds=surface)


def envelope_radius(geometry, row):
    """Current whole-robot/box planar radius about the measured pelvis, metres."""
    maximum=0.
    for name,shapes in geometry['robot'].items():
        pose=row['body_poses'][name]
        for shape in shapes:
            for point in itertools.product(*zip(shape['min'],shape['max'])):
                rotated=rotate_xyzw(pose['xyzw'],point)
                world_xy=[rotated[i]+pose['pos'][i] for i in range(2)]
                maximum=max(maximum,math.dist(world_xy,row['root_pos'][:2]))
    bounds=box_bounds(row)
    for point in itertools.product(*zip(bounds['min'][:2],bounds['max'][:2])):
        maximum=max(maximum,math.dist(point,row['root_pos'][:2]))
    return maximum


def turn_clearance(geometry,row,surfaces):
    """Conservative circle-to-support-footprint gap, not a swept-volume proof."""
    radius=envelope_radius(geometry,row)
    root=row['root_pos']
    distance=min(math.hypot(*(max(0.,s['min'][i]-root[i],root[i]-s['max'][i]) for i in range(2)))
                 for s in surfaces)
    return distance-radius


def read_support_parts(stage):
    """Read static CollisionAPI bounds once; dynamic robot poses come from physics."""
    from pxr import Usd,UsdGeom,UsdPhysics
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],
                           useExtentsHint=False,ignoreVisibility=True)
    result={}
    for name in ('source','destination'):
        parts=[]
        for part in [name+'_top',*[f'{name}_leg_{i}' for i in range(4)]]:
            prim=stage.GetPrimAtPath('/World/envs/env_0/'+part+'/geometry/mesh')
            if not prim.HasAPI(UsdPhysics.CollisionAPI):raise ValueError('missing support collision '+part)
            bounds=cache.ComputeWorldBound(prim).ComputeAlignedBox()
            if bounds.IsEmpty():raise ValueError('empty support bounds '+part)
            parts.append(dict(name=part,min=list(bounds.GetMin()),max=list(bounds.GetMax())))
        result[name]=parts
    return result
