"""Bounded stop/refine logic driven by observed retreat, not commanded travel.

Metres/seconds. Preview horizon 0.3 s and 5 mm measurement reserve are explicit
simulation assumptions. Final tolerance is the existing -2/+3 cm envelope.
"""
from collections import deque
import math

class RetreatStop:
    def __init__(self,distance,now):
        if isinstance(distance,bool) or not math.isfinite(distance) or not .2<=distance<=.5:raise ValueError('distance_outside_envelope')
        self.distance=distance;self.started=self.previous=now;self.phase_started=now
        self.phase='drive';self.speed=0.;self.refinements=0;self.history=deque(maxlen=3)
    def update(self,now,observed_at,progress,ready):
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (now,observed_at,progress)):
            raise ValueError('invalid_retreat_measurement')
        if type(ready) is not bool or now<self.previous or not -1e-8<=now-observed_at<=.150001:raise ValueError('stale_retreat_measurement')
        if self.history and observed_at<self.history[-1][0]:raise ValueError('retreat_measurement_reordered')
        if not self.history or observed_at>self.history[-1][0]:self.history.append((observed_at,progress))
        dt=now-self.previous;self.previous=now
        if now-self.started>12:raise ValueError('carry_retreat_timeout')
        if progress<-.03:raise ValueError('carry_wrong_direction')
        if progress>self.distance+.03:raise ValueError('carry_overshoot')
        velocity=0.
        if len(self.history)>1:
            velocity=max(0.,(self.history[-1][1]-self.history[0][1])/(self.history[-1][0]-self.history[0][0]))
        remaining=self.distance-progress;reserve=.3*velocity+.005;outcome=None
        if self.phase=='drive' and remaining<=reserve:
            self.phase='settle';self.phase_started=now;self.speed=0.
        elif self.phase=='settle':
            if now-self.phase_started>=1. and ready:
                if remaining<=.02:outcome='completed'
                else:
                    if self.refinements>=2:raise ValueError('carry_refinement_limit')
                    self.refinements+=1;self.phase='drive';self.phase_started=now
            elif now-self.phase_started>=5:raise ValueError('carry_settle_timeout')
        if self.phase=='drive':self.speed=min(.12,self.speed+.3*min(dt,.02))
        else:self.speed=0.
        return dict(phase=self.phase,speed_m_s=self.speed,measured_speed_m_s=velocity,
                reserve_m=reserve,remaining_m=remaining,refinements=self.refinements,outcome=outcome)
