"""Fixed-command source-plane evaluation; no cache across control steps.

Temporary optimization candidate. Snapshot measured geometry once, then retain
the original whole-convex-part choice and uncertainty for every queried part.
Time/frame changes require a new instance. This supplies no motion permission.
"""
import numpy as np
from .source_stop import SourceSides
from .loaded_stop import rotation as validated_rotation

class PreparedSourceSides(SourceSides):
    def __init__(self,source,now,rotation):
        super().__init__(source.top,source.memory,source.motion,source.gyro,source.time,pose_errors=source.pose_errors)
        self.now=float(now);self.query_rotation=validated_rotation(rotation).copy()
        self.top_data=None;self.front_data=None;self.front_error=None
        if self.top is not None:
            self.top.bounds([[0.,0.,0.]],now,rotation)
            relative=self.query_rotation.T@self.top.rotation
            self.top_data=(relative@self.top.normal,relative@self.top.anchor,
                           self.top.offset,self.top.error,self.top.normal_difference)
        try:
            if (self.motion.get('status')!='tracked_local_segment'
                    or not 0<=now-self.motion.get('time_s',float('nan'))<=.150001):
                self.memory.invalid=True
                raise ValueError('stop_front_motion_unavailable')
            pose=np.asarray(self.motion['body_in_segment']).copy()
            pose[:3,:3]=pose[:3,:3]@self.gyro.T@self.query_rotation
            frame=dict(self.motion,time_s=now,body_in_segment=pose.tolist())
            # Original query performs the complete continuity/rigid validation.
            query=self.memory.query(now,frame,[[0.,0.,0.]],translation_error_m=self.pose_errors['translation_error_m'],
                              rotation_error_rad=self.pose_errors['rotation_error_rad'])
            self.pose_translation_error=query['pose_translation_error_m']
            self.pose_rotation_error=query['pose_rotation_error_rad']
            self.front_normal_difference=2*np.sin(self.pose_rotation_error/2)+2*np.sin(self.memory.angle/2)
            self.lipschitz=1+max(self.top.normal_difference if self.top else 0,self.front_normal_difference)
            self.front_data=(pose[:3,:3].copy(),pose[:3,3].copy(),
                self.memory.normal.copy(),self.memory.anchor.copy(),
                self.memory.error,2*np.sin(self.memory.angle/2))
        except ValueError as error:
            self.front_error=str(error)
            if self.top_data is None:raise

    def bounds(self,points,now,rotation,*,front_closing_m=0.,top_closing_m=0.,point_displacement_m=0.):
        if now!=self.now or not np.array_equal(rotation,self.query_rotation):
            raise ValueError('prepared source command time or frame changed')
        allowances=np.asarray([front_closing_m,top_closing_m,point_displacement_m],float)
        points=np.asarray(points,float)
        if not np.isfinite(allowances).all() or min(allowances)<0:
            raise ValueError('invalid directional motion allowances')
        if points.ndim!=2 or points.shape[1]!=3 or not len(points) or not np.isfinite(points).all():
            raise ValueError('invalid query points')
        top=None
        if self.top_data is not None:
            normal,anchor,offset,error,difference=self.top_data
            top=(points@normal+offset-error-difference*np.linalg.norm(points-anchor,axis=1)
                 -top_closing_m-difference*point_displacement_m)
        if self.front_data is None or self.memory.invalid:
            if top is None:raise ValueError(self.front_error or 'source memory continuity unavailable')
            return top
        R,translation,normal,anchor,error,difference=self.front_data
        mapped=points@R.T+translation
        nominal=(mapped-anchor)@normal
        pose_error=self.pose_translation_error+2*np.sin(self.pose_rotation_error/2)*np.linalg.norm(points,axis=1)
        plane_error=error+difference*np.linalg.norm(mapped-anchor,axis=1)
        front=(nominal-(pose_error+plane_error)-front_closing_m
               -self.front_normal_difference*point_displacement_m)
        return front if top is None or min(front)>min(top) else top


    def bounds_many(self,points,now,rotation,*,front_closing_m=0.,top_closing_m=0.,point_displacement_m=0.):
        """Independent convex parts shaped (parts, vertices, xyz), all one command.

        Each part chooses one whole top or front bound. No choice crosses parts,
        headings or vertices. Per-part motion allowances are metres, as in bounds.
        Scalar bounds remains the independent reference and ordinary stop path.
        """
        if now!=self.now or not np.array_equal(rotation,self.query_rotation):
            raise ValueError('prepared source command time or frame changed')
        points=np.asarray(points,float)
        if (points.ndim!=3 or points.shape[-1]!=3 or not points.shape[0]
                or not points.shape[1] or not np.isfinite(points).all()):
            raise ValueError('invalid query points')
        allowances=[np.broadcast_to(np.asarray(value,float),(len(points),))[:,None]
                    for value in (front_closing_m,top_closing_m,point_displacement_m)]
        if any(not np.isfinite(value).all() or np.any(value<0) for value in allowances):
            raise ValueError('invalid directional motion allowances')
        front_closing,top_closing,displacement=allowances
        top=None
        if self.top_data is not None:
            normal,anchor,offset,error,difference=self.top_data
            top=(points@normal+offset-error-difference*np.linalg.norm(points-anchor,axis=-1)
                 -top_closing-difference*displacement)
        if self.front_data is None or self.memory.invalid:
            if top is None:raise ValueError(self.front_error or 'source memory continuity unavailable')
            return top
        R,translation,normal,anchor,error,difference=self.front_data
        mapped=points@R.T+translation
        nominal=(mapped-anchor)@normal
        pose_error=self.pose_translation_error+2*np.sin(self.pose_rotation_error/2)*np.linalg.norm(points,axis=-1)
        plane_error=error+difference*np.linalg.norm(mapped-anchor,axis=-1)
        front=(nominal-(pose_error+plane_error)-front_closing
               -self.front_normal_difference*displacement)
        if top is None:return front
        return np.where(front.min(axis=-1,keepdims=True)>top.min(axis=-1,keepdims=True),front,top)

def prepare(source,now,rotation):
    return PreparedSourceSides(source,now,rotation)
