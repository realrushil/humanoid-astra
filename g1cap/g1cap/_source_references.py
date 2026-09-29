"""Bounded time-sampled source history; original timestamps and source lifetime."""
import numpy as np
from ._source_front_track import SourceFrontTrack
from g1cap.remembered_source_plane import RememberedSourcePlane

class SourceFrontReferences:
    def __init__(self,owner,*,uncertainty=None):
        self.uncertainty=uncertainty
        self.owner=owner;self.track=SourceFrontTrack(owner,uncertainty=uncertainty)
        self.latest=self.long=None;self.failure=None;self.query_time=None
        self.history={}
    def invalidate(self,reason):
        self.failure=reason
        for memory in (self.latest,self.long,*self.history.values()):
            if memory is not None:memory.invalid=True
    def observe(self,t,candidates,owner,motion,source_valid=True):
        if self.query_time is not None and t<self.query_time:self.invalidate('source_observation_time_reversed')
        if not source_valid:self.invalidate('source_tracker_invalid')
        if self.failure:return dict(status='unavailable',reason=self.failure,time_s=t)
        if self.uncertainty is not None:
            if self.uncertainty.time is None and motion.get('status')!='tracked_local_segment':
                return dict(status='unavailable',reason='scene_uncertainty_uninitialized',time_s=t)
            try:self.uncertainty.observe(t,motion)
            except ValueError as error:
                self.invalidate(str(error))
                return dict(status='unavailable',reason=self.failure,time_s=t)
        result=self.track.observe(t,candidates,owner,motion)
        if self.track.failure:self.invalidate(self.track.failure)
        if result['status']=='observed_source_front':
            c=result['candidate'];b=result['uncertainty'];n=-np.asarray(c['normal_body']);a=np.asarray(c['anchor_body'])
            self.latest=RememberedSourcePlane(t,motion,n,-n@a,a,
                plane_error_m=b['anchor_error_m'],plane_angle_rad=b['normal_angle_rad'],max_age_s=10.,uncertainty=self.uncertainty)
            if b['span_m']>=.15:self.long=self.latest
            # First accepted observation in each one-second bucket. This is a
            # storage bound, not additional observed span or sensor accuracy.
            # Keep original measurement objects/times; never renew their age.
            self.history={k:m for k,m in self.history.items()
                          if not m.invalid and (m.uncertainty is not None or 0<=t-m.time<=m.max_age)}
            self.history.setdefault(int(t),self.latest)
            if self.uncertainty is not None:
                self.history=dict(sorted(self.history.items())[-10:])
        return result
    def available(self,t,owner,motion):
        if self.failure:return []
        if (not np.isfinite(t) or t<0 or self.query_time is not None and t<self.query_time):
            self.invalidate('source_query_time_invalid');return []
        self.query_time=t
        if owner is not self.owner:self.invalidate('source_owner_changed');return []
        if self.latest is None:return []
        if (motion.get('status')!='tracked_local_segment' or motion.get('segment')!=self.latest.segment
                or not 0<=t-motion.get('time_s',float('inf'))<=.150001):
            self.invalidate('source_query_motion_lost');return []
        result=[]
        for memory in (self.latest,self.long,*self.history.values()):
            if memory is None or any(memory is m for m in result):continue
            if self.uncertainty is None:
                if not 0<=t-memory.time<=memory.max_age:memory.invalid=True
            else:
                try:memory.pose_uncertainty(t)
                except ValueError:memory.invalid=True
            if not memory.invalid:result.append(memory)
        self.history={k:m for k,m in self.history.items() if not m.invalid}
        return result
