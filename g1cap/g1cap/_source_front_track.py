"""Conditional source-front association; no motor commands or memory refresh."""
import numpy as np
from g1cap.scene_motion import _rigid
from ._front_uncertainty import uncertainty

class SourceFrontTrack:
    def __init__(self,owner,*,uncertainty=None):
        if owner is None:raise ValueError('source perception owner required')
        self.uncertainty=uncertainty;self.original_capture=None
        self.owner=owner;self.original=None;self.last=None;self.failure=None;self.seen=None
    def observe(self,t,candidates,owner,motion):
        def unavailable(reason,latch=False):
            if latch:self.failure=reason
            return dict(status='unavailable',reason=reason,time_s=t)
        if self.failure:return unavailable(self.failure)
        if owner is not self.owner:return unavailable('source_owner_changed',True)
        if (not np.isfinite(t) or t<0 or self.seen is not None and t<=self.seen):
            return unavailable('front_timestamp_invalid',True)
        self.seen=t
        if (motion.get('status')!='tracked_local_segment' or
                abs(motion.get('time_s',float('inf'))-t)>1e-8):
            return unavailable('front_motion_unavailable',self.original is not None)
        pose=_rigid(motion['body_in_segment'])
        reacquiring=False
        if self.original is not None:
            if motion['segment']!=self.original['segment']:return unavailable('front_segment_changed',True)
            age=t-self.last['time_s']
            if self.uncertainty is None:
                if age>10.:return unavailable('front_memory_expired',True)
            else:
                # Keep the original association anchor; a fresh fragment cannot
                # erase uncertainty in the identity of the original boundary.
                try:self.uncertainty.bounds(self.original_capture,t)
                except ValueError as error:return unavailable(str(error),True)
            reacquiring=age>.150001
        measured=[]
        for candidate in candidates:
            try:bound=uncertainty(candidate['line'])
            except ValueError:continue
            normal=np.asarray(candidate['normal_body']);anchor=np.asarray(candidate['anchor_body'])
            if (normal.shape!=(3,) or anchor.shape!=(3,) or not np.isfinite(np.r_[normal,anchor]).all()
                    or abs(np.linalg.norm(normal)-1)>1e-6):raise ValueError('invalid measured front')
            r=dict(time_s=t,segment=motion['segment'],normal=pose[:3,:3]@normal,
                anchor=pose[:3,:3]@anchor+pose[:3,3],bound=bound,candidate=candidate)
            if self.original is None:
                if bound['span_m']>=.15:measured.append(r)
            elif (self.compatible(r,self.original,.10,np.deg2rad(2.))
                    and self.compatible(r,self.last,.10 if reacquiring else .03,np.deg2rad(2. if reacquiring else 5.))):measured.append(r)
        if len(measured)!=1:return unavailable('ambiguous_source_front' if measured else 'source_front_not_matched')
        if reacquiring and measured[0]['bound']['span_m']<.15:
            return unavailable('reacquisition_requires_long_fragment')
        self.last=measured[0]
        if self.original is None:
            self.original=self.last
            if self.uncertainty is not None:self.original_capture=self.uncertainty.snapshot()
        return dict(status='observed_source_front',time_s=t,association='initialized' if self.original is self.last else ('reacquired_original_boundary' if reacquiring else 'original_boundary_match'),
            candidate=self.last['candidate'],uncertainty=self.last['bound'],
            source_identity='conditional_geometric_association',segment=motion['segment'])
    @staticmethod
    def compatible(a,b,translation,rotation):
        """Symmetric cone/anchor consistency in the same stance-segment frame.

        Original-reference 10cm/2deg and temporal 3cm/5deg are explicit
        localization/association sensitivities, not calibrated identity proof.
        Keeping the original reference prevents unbounded pairwise ratcheting.
        """
        aa=a['bound']['normal_angle_rad'];ba=b['bound']['normal_angle_rad']
        angle=aa+ba+rotation
        if angle>=np.pi/2 or a['normal']@b['normal']<np.cos(angle):return False
        delta=a['anchor']-b['anchor'];radius=np.linalg.norm(delta)
        error=a['bound']['anchor_error_m']+b['bound']['anchor_error_m']+translation
        return (abs(delta@a['normal'])<=error+2*np.sin(aa/2)*radius
                and abs(delta@b['normal'])<=error+2*np.sin(ba/2)*radius)
