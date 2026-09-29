"""Offline candidate: source clearance over a requested rigid-body yaw sweep.

Uses only measured geometry and the existing SourceSides uncertainty assumptions.
This is an admission estimate, not a dynamics simulation or navigation permission.
The runtime must still screen each command and verify retained, settled load.
"""
import math
import numpy as np
from g1cap._source_turn_probe import TurnPlane

class DepartureSweep(TurnPlane):
    def __init__(self,source,rotation,up,yaw_rad):
        if (isinstance(yaw_rad,bool) or not isinstance(yaw_rad,(int,float))
                or not math.isfinite(yaw_rad) or not 0<abs(yaw_rad)<=math.pi/2):
            raise ValueError('departure_turn_outside_envelope')
        super().__init__(source,rotation,up)
        self.yaw_rad=float(yaw_rad)
        # Query-body coordinates, radians about measured floor up through pelvis.
        # Samples <=0.05 rad apart: existing +/-0.05 rad chord allowance covers
        # every intermediate heading and the endpoint residual yaw allowance.
        count=math.ceil(abs(yaw_rad)/.05)
        x,y,z=self.up
        cross=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
        self.rotations=[np.eye(3)+math.sin(a)*cross+(1-math.cos(a))*(cross@cross)
                        for a in np.linspace(0,yaw_rad,count+1)]
    def bounds(self,points,now,rotation):
        points=np.asarray(points,float)
        # Each whole convex part must clear at every heading. The source's
        # top/front alternative is resolved per part, never per vertex.
        from ._prepared_source_sides import PreparedSourceSides
        if isinstance(self.source,PreparedSourceSides):
            if not np.allclose(rotation,self.rotation,atol=1e-10):
                raise ValueError('turn_query_frame_changed')
            # Preserve every sampled heading and its own whole-part side choice.
            rotated=points@np.swapaxes(np.asarray(self.rotations),1,2)
            radius=np.linalg.norm(rotated-(rotated@self.up)[...,None]*self.up,axis=-1).max(axis=-1)
            travel=radius*self.chord
            values=self.source.bounds_many(rotated,now,rotation,
                front_closing_m=.10+self.front_tangent*travel,
                top_closing_m=.02+self.top_tangent*travel,
                point_displacement_m=.12+travel)
            return np.minimum.reduce(values)
        return np.minimum.reduce([super(DepartureSweep,self).bounds(points@R.T,now,rotation)
                                  for R in self.rotations])
