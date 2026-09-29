"""Sensor-screened stopping only: fresh top OR bounded original-front memory.

Assumes a static source boundary, continuous stance segment and declared pose
error over <=150ms. No height reconstruction, navigation or workspace map.
"""
import numpy as np
from .loaded_stop import LoadedStopClearance
from .remembered_source_plane import RememberedSourcePlane

class SourceSides:
    def __init__(self,top,memory,motion,gyro,time_s,*,pose_errors=None):
        self.top,self.memory,self.motion=top,memory,motion
        self.gyro=np.asarray(gyro)
        # Vertical prediction must cover the age of every retained top bound.
        self.time=min(time_s,top.time) if top is not None else time_s
        angle=np.deg2rad(2.)
        translation=.10
        if memory.uncertainty is not None:
            # The retained top may be older than the current motion ledger.
            # Its age belongs to vertical prediction, not this pose-error query.
            try:
                errors=memory.pose_uncertainty(motion['time_s'])
                angle=errors['rotation_error_rad'];translation=errors['translation_error_m']
            except ValueError:pass  # invalid memory may still allow a fresh top
        if pose_errors is not None:
            values=[pose_errors['translation_error_m'],pose_errors['rotation_error_rad']]
            if not np.isfinite(values).all() or values[0]<.10 or not np.deg2rad(2)<=values[1]<=np.pi:
                raise ValueError('invalid source pose uncertainty')
            translation=max(translation,values[0]);angle=max(angle,values[1])
        self.pose_errors=dict(translation_error_m=translation,rotation_error_rad=angle)
        self.front_normal_difference=2*np.sin(angle/2)+2*np.sin(memory.angle/2)
        self.lipschitz=1+max(top.normal_difference if top else 0,self.front_normal_difference)
    def bounds(self,points,now,rotation,*,front_closing_m=0.,top_closing_m=0.,point_displacement_m=0.):
        """Whole-part bound with caller-supplied motion envelopes, in metres.

        Closing bounds the positive displacement toward each nominal plane;
        total point displacement also bounds changes in normal/pose uncertainty.
        Both use the fixed query frame and must cover every point throughout the
        proposed motion and stopping.
        These arguments do not estimate a gait envelope or authorize navigation.
        The existing stopping caller uses zero here and its separate arm path
        and vertical predictor. All points must describe one convex part.
        """
        allowances=np.array([front_closing_m,top_closing_m,point_displacement_m],float)
        if allowances.shape!=(3,) or not np.isfinite(allowances).all() or min(allowances)<0:
            raise ValueError('invalid directional motion allowances')
        top=(self.top.bounds(points,now,rotation)-top_closing_m
             -self.top.normal_difference*point_displacement_m) if self.top else None
        try:
            if (self.motion.get('status')!='tracked_local_segment'
                    or not 0<=now-self.motion.get('time_s',float('nan'))<=.150001):
                self.memory.invalid=True
                raise ValueError('stop_front_motion_unavailable')
            frame=dict(self.motion);pose=np.asarray(frame['body_in_segment']).copy()
            pose[:3,:3]=pose[:3,:3]@self.gyro.T@np.asarray(rotation)
            frame.update(time_s=now,body_in_segment=pose.tolist())
            front=self.memory.query(now,frame,points,translation_error_m=self.pose_errors['translation_error_m'],rotation_error_rad=self.pose_errors['rotation_error_rad'])
        except ValueError:
            if top is None:raise
            return top
        values=(np.asarray(front['lower_m'])-front_closing_m
                -(2*np.sin(front['pose_rotation_error_rad']/2)+2*np.sin(self.memory.angle/2))*point_displacement_m)
        # A whole convex part chooses a side, never individual vertices.
        return values if top is None or min(values)>min(top) else top

class SourceStopClearance(LoadedStopClearance):
    def __init__(self,*args):
        super().__init__(*args)
        self.source_owner=None;self.front_memory=None;self.front_lost=False
        self.uncertainty=None
        self.motion=None;self.front_gyro=None
        self.cached_top=None;self.cached_top_segment=None
    def observe_front(self,front,motion,gyro,source_owner,source_available):
        self.motion=motion;self.front_gyro=np.asarray(gyro).copy()
        # Reset by actual sensor-tracker lifetime, never by a newly visible edge.
        if source_owner is not self.source_owner:
            self.source_owner=source_owner;self.front_memory=None;self.front_lost=False
            self.cached_top=None;self.cached_top_segment=None
        if self.front_memory is not None and source_owner is not None and source_owner.plane is None:
            self.front_memory.invalid=True
        if self.front_memory is not None and (motion.get('status')!='tracked_local_segment'
                or motion.get('segment')!=self.front_memory.segment):
            self.front_memory.invalid=True
        if self.front_memory is not None and (front.get('status')!='observed_candidate'
                or front.get('association')!='track'):
            self.front_lost=True
        if source_owner is None or not source_available:return
        if front.get('status')!='observed_candidate' or self.front_lost:return
        if (motion.get('status')!='tracked_local_segment'
                or not np.isfinite(front.get('observed_at_s',float('nan')))
                or abs(front['observed_at_s']-motion.get('time_s',float('inf')))>1e-8):
            self.front_lost=True;return
        if self.front_memory is not None and self.front_memory.invalid:return
        if self.front_memory is not None and front['observed_at_s']<=self.front_memory.time:return
        n=-np.asarray(front['normal_body']);d=-front['offset_body_m']
        self.front_memory=RememberedSourcePlane(motion['time_s'],motion,n,d,-d*n,
            plane_error_m=.003,plane_angle_rad=np.deg2rad(.5),max_age_s=10.,uncertainty=getattr(self,'uncertainty',None))
    def measure(self,packet,seed,rotation,up_body,source):
        if isinstance(self.plane,SourceSides):self.plane=self.plane.top
        super().measure(packet,seed,rotation,up_body,source)
        # Bridge only short missing detections, with the original camera/gyro
        # measurement and existing 150 ms limit. Never bridge lost identity,
        # floor geometry or stance continuity, or renew a missing observation.
        camera_t=seed['time_s'];motion=self.motion or {}
        valid=(self.source_owner is not None and self.source_owner.plane is not None
               and motion.get('status')=='tracked_local_segment'
               and abs(motion.get('time_s',float('inf'))-camera_t)<1e-8
               and bool(self.heights) and abs(self.heights[-1][0]-camera_t)<1e-8
               and seed['box']['status']=='accepted')
        cached=self.cached_top
        if (not valid or (cached is not None and
                (motion.get('segment')!=self.cached_top_segment
                 or not 0<=packet['time_s']-cached.time<=.150001))):
            self.cached_top=None;self.cached_top_segment=None
        if valid and self.plane is not None:
            self.cached_top=self.plane;self.cached_top_segment=motion['segment']
        elif valid and source is None:
            self.plane=self.cached_top
        elif source is not None:
            self.cached_top=None;self.cached_top_segment=None
        if self.front_memory is not None and not self.front_memory.invalid and self.motion is not None:
            self.plane=SourceSides(self.plane,self.front_memory,self.motion,self.front_gyro,seed['time_s'])
    def screen(self,*args):
        result=super().screen(*args)
        factor=self.plane.lipschitz if isinstance(self.plane,SourceSides) else 1+self.plane.normal_difference
        if factor>1:
            extra=result['path']['interpolation_allowance_m']*(factor-1)
            result['lower_m']-=extra
            result['status']='clear' if result['lower_m']>=.03 else 'insufficient_margin'
            result['additional_interpolation_allowance_m']=extra
        remembered=isinstance(self.plane,SourceSides) and not self.front_memory.invalid
        result['source_reference']=('fresh_top_or_remembered_front' if self.plane.top is not None
            else 'remembered_front') if remembered else 'fresh_top'
        if remembered:
            result.update(source_observed_at_s=self.front_memory.time,
                source_age_s=args[0]['time_s']-self.front_memory.time,
                assumed_translation_error_m=self.front_memory.pose_uncertainty(args[0]['time_s'])['translation_error_m'],
                assumed_rotation_error_rad=self.front_memory.pose_uncertainty(args[0]['time_s'])['rotation_error_rad'])
        return result
