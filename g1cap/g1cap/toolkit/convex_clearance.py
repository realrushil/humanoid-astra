"""Native arm/support distance using static source hulls and measured link poses.

World metres; link quaternions are XYZW. Coal shapes are cached for one world.
Primitive/non-hull colliders use conservative oriented local bounds. Source
hulls are not a byte-identical PhysX cooking export or a dynamics prediction.
"""


class ConvexClearance:
    def __init__(self, geometry, supports):
        import coal
        import numpy as np
        self.coal=coal;self.np=np;self.arms=[];self.parts=[]
        for name,shapes in geometry['robot'].items():
            if not any(w in name for w in ('shoulder','elbow','wrist','hand')):
                continue
            for shape in shapes:
                low,high=np.array(shape['min']),np.array(shape['max'])
                if 'convex_vertices' in shape:
                    vertices=np.asarray(shape['convex_vertices'],dtype=float)
                    if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4 or not np.isfinite(vertices).all():
                        raise ValueError('invalid_source_hull_vertices')
                    if np.any(vertices<low-2e-6) or np.any(vertices>high+2e-6):
                        raise ValueError('source_hull_leaves_recorded_bounds')
                    points=coal.StdVec_Vec3s()
                    for point in vertices:points.append(point)
                    collider=coal.ConvexBase.convexHull(points,True,'Qt');offset=np.zeros(3)
                else:
                    collider=coal.Box(high-low);offset=(high+low)/2
                self.arms.append((name,collider,offset))
        if not self.arms:raise ValueError('missing_arm_collision_geometry')
        for parts in supports.values():
            for part in parts:
                low,high=np.array(part['min']),np.array(part['max'])
                self.parts.append((coal.Box(high-low),coal.Transform3s(np.eye(3),(low+high)/2)))
        self.request=coal.DistanceRequest()

    def for_pose(self, obs, drop):
        """Return a distance evaluator for common world-XY shifts at this pose.

        Apply the same hypothetical downward shift as the conservative planner.
        The real controller rechecks fresh measured poses after retraction.
        """
        from scipy.spatial.transform import Rotation
        np=self.np;coal=self.coal;arms=[]
        for name,shape,offset in self.arms:
            pose=obs['body_poses'][name]
            rotation=Rotation.from_quat(pose['xyzw']).as_matrix()
            center=np.array(pose['pos'])+rotation@offset-np.array([0.,0.,drop])
            arms.append((shape,rotation,center))
        def distance(delta):
            return min(coal.distance(shape,coal.Transform3s(rotation,center+delta),part,tf,
                                     self.request,coal.DistanceResult())
                       for shape,rotation,center in arms for part,tf in self.parts)
        return distance
