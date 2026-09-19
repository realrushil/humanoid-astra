"""Observed-clearance objective; the shared paired wrist owner applies its offset.

Returns equal vertical displacement for BOTH existing wrist targets (metres).
No finger squeeze, navigation, force estimate, or actuator writer is added.
A 5 mm visual reserve, 1 cm/s and 4 cm per-grasp budget are uncalibrated
development bounds. Object mass and simulator poses/contacts are not inputs.
"""
import math
from collections import deque

class ClearanceLift:
    def __init__(self,now_s,clearance_m=.08,used_m=0.):
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in (now_s,clearance_m,used_m)):
            raise ValueError('invalid_parameter')
        if not .05<=clearance_m<=.10:raise ValueError('height_outside_envelope')
        if not 0<=used_m<=.04:raise ValueError('clearance_lift_budget_exhausted')
        self.goal=clearance_m+.005
        self.started=now_s;self.previous=None;self.offset=used_m;self.budget_at=None
        self.outcome=self.reason=None
        self.clearance_samples=deque()

    def validate(self,now_s,grasp):
        def finite(x):return type(x) in (int,float) and math.isfinite(x)
        t=grasp.get('time_s')
        if not finite(now_s) or now_s<self.started:raise ValueError('lift_camera_discontinuity')
        if not finite(t) or not 0<=now_s-t<=.150001:raise ValueError('lift_camera_stale')
        if self.previous is not None and not 0<=t-self.previous<=.150001:
            raise ValueError('lift_camera_discontinuity')
        if (grasp.get('status')!='available' or not grasp.get('opposing_near_wrists')
                or not grasp.get('attitude_ok') or not finite(grasp.get('gap_m'))
                or grasp['gap_m']<.02):raise ValueError('lift_visual_retention_lost')
        if grasp.get('settled_retention') is not True:raise ValueError('lift_grasp_unsettled')

    def update(self,now_s,grasp):
        if self.outcome:return self.result()
        self.validate(now_s,grasp)
        now=now_s;t=grasp['time_s']
        if t!=self.previous:self.clearance_samples.append((t,grasp['gap_m']))
        # Keep the interval crossing one second across camera rates; do not
        # replace elapsed physical coverage with a fixed number of frames.
        while len(self.clearance_samples)>2 and self.clearance_samples[1][0]<=t-1.+1e-9:
            self.clearance_samples.popleft()
        # Use the same uncalibrated 5 mm sensor reserve as the existing pickup
        # check. A single target crossing does not qualify the settled height.
        sustained=(len(self.clearance_samples)>=2
                   and t-self.clearance_samples[0][0]>=.999
                   and min(g for _,g in self.clearance_samples)>=self.goal)
        if sustained and grasp.get('ready'):
            self.outcome='completed';self.reason='observed_clearance_and_hold'
        elif now-self.started>=8.-1e-8:
            self.outcome='failed';self.reason='clearance_lift_timeout'
        elif self.budget_at is not None and now-self.budget_at>=1.-1e-8:
            self.outcome='failed';self.reason='clearance_lift_budget_exhausted'
        elif t!=self.previous:
            dt=0. if self.previous is None else t-self.previous
            # Monotonic bounded target: no integral windup, downward correction,
            # or motion extrapolated from repeated cached camera predicates.
            increment=min(.01*dt,max(0.,self.goal-grasp['gap_m']),.04-self.offset)
            self.offset+=increment
            if self.offset>=.04-1e-9 and self.budget_at is None:self.budget_at=now
        self.previous=t
        return self.result()

    def result(self):
        return dict(offset_m=self.offset,outcome=self.outcome,reason=self.reason)
