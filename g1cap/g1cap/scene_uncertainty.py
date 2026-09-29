"""Explicit, uncalibrated drift budget for a continuous measured stance frame.

Only actual sensor poses advance this ledger. Hypothetical clearance queries
read it. Rates are required configuration, not fitted hardware noise estimates.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from .scene_motion import _rigid

def validate_forecast(elapsed_s,travel_m,rotation_rad):
    values=(travel_m,rotation_rad,elapsed_s)
    if (any(isinstance(v,bool) for v in values) or not np.isfinite(values).all()
            or min(values)<0 or elapsed_s<=0):
        raise ValueError('invalid scene uncertainty forecast')
    return dict(elapsed_s=float(elapsed_s),travel_m=float(travel_m),rotation_rad=float(rotation_rad))

@dataclass(frozen=True)
class SceneCapture:
    owner: object = field(repr=False)
    time_s: float
    distance_m: float
    rotation_rad: float

class SceneUncertainty:
    def __init__(self, *, translation_growth, rotation_growth,
                 max_translation_m, max_rotation_rad):
        # Coefficients multiply [travel metres, travel radians, elapsed seconds].
        # Result units are metres for translation and radians for orientation.
        for rates in (translation_growth,rotation_growth):
            if (len(rates)!=3 or any(isinstance(v,bool) for v in rates)
                    or not np.isfinite(rates).all() or min(rates)<0):
                raise ValueError('invalid scene uncertainty growth')
        if (not np.isfinite([max_translation_m,max_rotation_rad]).all()
                or not .10<max_translation_m or not np.deg2rad(2)<max_rotation_rad<=np.pi):
            raise ValueError('invalid scene uncertainty ceilings')
        self.translation_growth=np.array(translation_growth,float)
        self.rotation_growth=np.array(rotation_growth,float)
        self.max_translation_m=float(max_translation_m)
        self.max_rotation_rad=float(max_rotation_rad)
        self.time=None;self.pose=None;self.segment=None;self.failure=None
        self.distance=0.;self.angle=0.;self.owner=object()

    def observe(self,time_s,motion):
        if self.failure:raise ValueError(self.failure)
        try:
            if (isinstance(time_s,bool) or not math.isfinite(time_s) or time_s<0
                    or motion.get('status')!='tracked_local_segment'
                    or motion.get('time_s')!=time_s
                    or not isinstance(motion.get('segment'),int)
                    or isinstance(motion.get('segment'),bool)):
                raise ValueError('scene uncertainty observation invalid')
            pose=_rigid(motion['body_in_segment'])
            if self.time is not None:
                dt=time_s-self.time
                if motion['segment']!=self.segment or not 0<=dt<=.150001:
                    raise ValueError('scene uncertainty continuity lost')
                if dt==0:
                    if not np.array_equal(pose,self.pose):
                        raise ValueError('scene uncertainty repeated pose changed')
                    return
                self.distance+=float(np.linalg.norm(pose[:3,3]-self.pose[:3,3]))
                cosine=(np.trace(self.pose[:3,:3].T@pose[:3,:3])-1)/2
                self.angle+=float(np.arccos(np.clip(cosine,-1.,1.)))
            self.time=float(time_s);self.pose=pose.copy();self.segment=motion['segment']
        except (ValueError,TypeError,KeyError) as error:
            self.failure=str(error)
            raise ValueError(self.failure) from error

    def snapshot(self):
        if self.failure or self.time is None:raise ValueError(self.failure or 'scene uncertainty uninitialized')
        return SceneCapture(self.owner,self.time,self.distance,self.angle)

    def bounds(self,capture,now):
        if self.failure:raise ValueError(self.failure)
        if (not isinstance(capture,SceneCapture) or capture.owner is not self.owner
                or self.time is None or isinstance(now,bool) or not math.isfinite(now)
                or not 0<=now-self.time<=.150001 or capture.time_s>self.time):
            raise ValueError('scene uncertainty query invalid')
        changes=np.array([self.distance-capture.distance_m,self.angle-capture.rotation_rad,now-capture.time_s])
        if not np.isfinite(changes).all() or min(changes)<0:raise ValueError('scene uncertainty capture invalid')
        translation=.10+float(self.translation_growth@changes)
        angle=float(np.deg2rad(2)+self.rotation_growth@changes)
        if translation>self.max_translation_m or angle>self.max_rotation_rad:
            raise ValueError('scene uncertainty budget exhausted')
        return dict(translation_error_m=translation,rotation_error_rad=angle,
                    distance_m=float(changes[0]),rotation_rad=float(changes[1]),elapsed_s=float(changes[2]))

    def forecast(self,capture,now,*,elapsed_s,travel_m,rotation_rad):
        """Error after a declared future path budget; never a new observation.

        Callers must separately screen physical motion/stopping geometry.
        Rejection here cannot invalidate a present-state stopping certificate.
        """
        validate_forecast(elapsed_s,travel_m,rotation_rad)
        values=(travel_m,rotation_rad,elapsed_s)
        current=self.bounds(capture,now)
        translation=current['translation_error_m']+float(self.translation_growth@values)
        angle=current['rotation_error_rad']+float(self.rotation_growth@values)
        if translation>self.max_translation_m or angle>self.max_rotation_rad:
            raise ValueError('scene uncertainty future budget exhausted')
        return dict(current,translation_error_m=translation,rotation_error_rad=angle,
                    forecast_elapsed_s=float(elapsed_s),forecast_travel_m=float(travel_m),
                    forecast_rotation_rad=float(rotation_rad))
