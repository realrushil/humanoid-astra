"""Acceptance and age semantics for one locally associated RGB-D cuboid track.

This is not object recognition. Continuity checks cannot disambiguate identical
objects or prove contact. Limits below are explicit simulation assumptions.
Only accepted snapshots are observations; prediction_seed is a short-lived fit
initialization and must never be exposed as a current measured pose.
"""
from copy import deepcopy
import math
import numpy as np


def _time(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError('time must be finite nonnegative seconds')
    return float(value)


def _pose(center,rotation,size):
    c,r,s=(np.asarray(x,float).copy() for x in [center,rotation,size])
    if (c.shape!=(3,) or r.shape!=(3,3) or s.shape!=(3,)
            or not all(np.isfinite(x).all() for x in [c,r,s]) or np.any(s<=0)
            or not np.allclose(r.T@r,np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(r),1.,atol=1e-5)):
        raise ValueError('invalid right-handed cuboid pose or dimensions')
    return c,r,s


class BoxEstimateStream:
    """Gate fitted candidates without extrapolating missing measurements.

    150 ms age, 250 ms continuity, 4 mm mixed fit RMS, 75% face correspondence coverage,
    1.5 m/s translation and 4 rad/s rotation are uncalibrated acceptance limits.
    RMS combines metric face distance with modeled-depth-scaled image-normal
    silhouette error; it is not a calibrated 3D pose uncertainty.
    Motion is relative to the camera, so camera motion also consumes the bound.
    Initialization is explicit: the caller must supply an observed complete-box
    hypothesis. Reinitialization starts a new local epoch, not a proven identity.
    """
    max_age_s=.15
    max_gap_s=.25
    max_rms_m=.004
    min_coverage=.75
    max_speed_m_s=1.5
    max_rotation_rad_s=4.

    def __init__(self):
        self._accepted=None
        self._last_received=-1.
        self._reason='not_initialized'
        self.epoch=0

    def _unavailable(self,reason):
        self._reason=reason
        return dict(status='unavailable',reason=reason,track_epoch=self.epoch,
                    observed_at_s=None,center_camera_m=None,axes_camera=None,dimensions_m=None)

    def initialize(self,*,time_s,now_s,center,rotation,size):
        time_s,now_s=_time(time_s),_time(now_s)
        c,r,s=_pose(center,rotation,size)
        if time_s>now_s+1e-9:return self._unavailable('future_frame')
        if now_s-time_s>self.max_age_s:return self._unavailable('stale')
        if time_s<=self._last_received:return self._unavailable('nonincreasing_frame_time')
        self.epoch+=1
        self._last_received=time_s
        self._store(time_s,c,r,s,dict(status='observed_initial_hypothesis'))
        return self.observe(now_s)

    def _store(self,time_s,center,rotation,size,quality):
        self._accepted=dict(status='accepted',reason='accepted',track_epoch=self.epoch,
            observed_at_s=time_s,center_camera_m=center.tolist(),axes_camera=rotation.tolist(),
            dimensions_m=size.tolist(),quality=deepcopy(quality))
        self._reason=None

    def prediction_seed(self,time_s):
        """Previous accepted pose for local fitting only; not a new observation."""
        time_s=_time(time_s)
        if self._accepted is None:return None
        age=time_s-self._accepted['observed_at_s']
        if age<0:return None
        if age>self.max_gap_s+1e-9:
            self._accepted=None
            self._reason='reinitialization_required'
            return None
        return deepcopy(self._accepted)

    def submit(self,*,time_s,now_s,center,rotation,size,quality):
        time_s,now_s=_time(time_s),_time(now_s)
        c,r,s=_pose(center,rotation,size)
        if time_s>now_s+1e-9:return self._unavailable('future_frame')
        if time_s<=self._last_received:return self._unavailable('nonincreasing_frame_time')
        self._last_received=time_s
        if now_s-time_s>self.max_age_s:return self._unavailable('stale')
        old=self.prediction_seed(time_s)
        if old is None:return self._unavailable('reinitialization_required')
        if quality.get('status')!='full_rank_candidate' or quality.get('constraint_rank')!=6:
            return self._unavailable('weak_geometry')
        rms=quality.get('rms_m',float('nan'))
        if not isinstance(rms,(int,float)) or not math.isfinite(rms) or rms<0 or rms>self.max_rms_m:
            return self._unavailable('large_fit_residual')
        count,total=quality.get('face_rows',0),quality.get('total_face_points',0)
        if (type(count) is not int or type(total) is not int or total<100
                or count>total or count/total<self.min_coverage):
            return self._unavailable('low_correspondence_coverage')
        if not np.allclose(s,old['dimensions_m'],rtol=0,atol=1e-9):
            return self._unavailable('dimensions_changed')
        dt=time_s-old['observed_at_s']
        angle=math.acos(float(np.clip((np.trace(np.array(old['axes_camera']).T@r)-1)/2,-1,1)))
        if (np.linalg.norm(c-old['center_camera_m'])>self.max_speed_m_s*dt
                or angle>self.max_rotation_rad_s*dt):
            return self._unavailable('implausible_motion')
        self._store(time_s,c,r,s,quality)
        return self.observe(now_s)

    def observe(self,now_s):
        """Return a timestamped accepted estimate or no pose; never relabel age."""
        now_s=_time(now_s)
        if self._reason is not None:return self._unavailable(self._reason)
        age=now_s-self._accepted['observed_at_s']
        if age<0:return self._unavailable('future_frame')
        if age>self.max_age_s+1e-9:return self._unavailable('stale')
        result=deepcopy(self._accepted)
        result['age_s']=age
        result['frame']='camera_optical_at_observation_time'
        return result
