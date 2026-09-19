"""Read CollisionAPI bounds in each rigid body's local frame; metres.

USD local bounds are static asset data. During physics, transform them using
measured link poses rather than stale USD world transforms (Fabric may own pose).
"""
def source_hull_vertices(prim,body,bounds,transforms):
    """Mesh source vertices in the rigid link frame; preserve recorded bounds."""
    from pxr import UsdGeom
    import numpy as np
    from scipy.spatial import ConvexHull
    points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),dtype=float)
    transform=np.asarray(transforms.ComputeRelativeTransform(prim,body)[0])
    # USD matrices use row vectors, including translation in the last row.
    points=points@transform[:3,:3]+transform[3,:3]
    if np.any(points<np.array(bounds['min'])-2e-6) or np.any(points>np.array(bounds['max'])+2e-6):
        raise ValueError('source_hull_leaves_recorded_bounds')
    return points[ConvexHull(points).vertices].tolist()


def collision_geometry(stage,robot):
    from pxr import Usd,UsdGeom,UsdPhysics
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],useExtentsHint=False,ignoreVisibility=True)
    transforms=UsdGeom.XformCache(Usd.TimeCode.Default())
    result={'robot':{},'source_parts':{},'units':'metres','native_pose_quaternion':'XYZW','empty_collision_containers':[]}
    for name in robot.body_names:
        body=stage.GetPrimAtPath('/World/envs/env_0/Robot/'+name)
        if not body.IsValid():raise ValueError('Missing body prim '+name)
        shapes=[]
        for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            if enabled is False:continue
            box=cache.ComputeRelativeBound(prim,body).ComputeAlignedBox()
            if box.IsEmpty():
                result['empty_collision_containers'].append({'prim':str(prim.GetPath()),'type':prim.GetTypeName(),
                    'children':[(str(c.GetPath()),c.GetTypeName()) for c in prim.GetChildren()]})
                continue
            shape={'prim':str(prim.GetPath()),'type':prim.GetTypeName(),
                'min':list(box.GetMin()),'max':list(box.GetMax())}
            if any(w in name for w in ('shoulder','elbow','wrist','hand')):
                shape['placement_representation']='conservative_oriented_bounds'
                if prim.GetTypeName()=='Mesh' and UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()=='convexHull':
                    shape['convex_vertices']=source_hull_vertices(prim,body,shape,transforms)
                    shape['placement_representation']='source_convex_hull'
            shapes.append(shape)
        result['robot'][name]=shapes
    for name in ['source_top',*[f'source_leg_{i}' for i in range(4)]]:
        prim=stage.GetPrimAtPath('/World/envs/env_0/'+name+'/geometry/mesh')
        assert prim.HasAPI(UsdPhysics.CollisionAPI)
        box=cache.ComputeWorldBound(prim).ComputeAlignedBox()
        result['source_parts'][name]={'prim':str(prim.GetPath()),'min':list(box.GetMin()),'max':list(box.GetMax())}
    result['initial_body_poses']={name:{'pos':robot.data.body_pos_w[0,i].cpu().tolist(),
        'xyzw':robot.data.body_quat_w[0,i].cpu().tolist()} for i,name in enumerate(robot.body_names)}
    return result
