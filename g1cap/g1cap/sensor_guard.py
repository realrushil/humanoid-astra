"""Sensor stops, not a contact-force estimator or hardware safety controller.

Pelvis up is estimated from RGB-D/IMU; local clearances from RGB-D plus encoders.
Motor torque is actuator output, including gravity/inertia, not external force.
15 degrees, negative observed clearance and 95% effort for 0.5 seconds are
uncalibrated simulation thresholds. Occluded geometry and substep contacts can
be missed. The independent scorer retains its stricter ground-truth checks.
"""
import math

from .arena_sensors import _finite
from .hand_sensors import measured_joint_positions


class SensorGuard:
    def __init__(self,names,effort_limits):
        self.names=list(names)
        self.limits=dict(effort_limits)  # Static body-motor configuration, N m.
        if len(self.names)!=43 or len(set(self.names))!=43 or len(self.limits)!=29:
            raise ValueError('43 joint names and 29 body effort limits required')
        if not set(self.limits)<=set(self.names) or any(
                isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0
                for v in self.limits.values()):raise ValueError('invalid static effort limits')
        self.saturated_since={n:None for n in self.limits}
        self.previous_step=self.previous_time=None
        self.result={'time_s':None,'fault':'sensor_guard_uninitialized'}
        self.latched=None

    def update(self,body,hands,up_body,front,hand):
        """Exactly once per 50 Hz encoder packet, before controller decisions."""
        result=dict(time_s=body.get('time_s'),fault=self.latched)
        try:
            measured_joint_positions(body,hands,self.names)
            if set(body['joint_names'])!=set(self.limits):raise ValueError('body mapping changed')
            up=_finite(up_body,3,'estimated up')
            if abs(sum(x*x for x in up)-1.)>1e-5:raise ValueError('up must be normalized')
            now=body['time_s']
            if self.previous_time is not None and (
                    body['step']!=self.previous_step+1 or abs(now-self.previous_time-.02)>1e-6):
                result['fault']=result['fault'] or 'sensor_sample_gap'
            self.previous_step,self.previous_time=body['step'],now
            tilt=math.acos(max(-1.,min(1.,up[2])))
            fractions={n:abs(t)/self.limits[n] for n,t in zip(body['joint_names'],body['tau_est_nm'],strict=True)}
            for n,fraction in fractions.items():
                if fraction<.95:self.saturated_since[n]=None
                elif self.saturated_since[n] is None:self.saturated_since[n]=now
            duration=max((now-t for t in self.saturated_since.values() if t is not None),default=0.)
            front_gap=front['clearance_m'] if front['status']=='available' else None
            hand_gap=hand['margin_m'] if hand['status']=='available' else None
            for value in (front_gap,hand_gap):
                if value is not None:_finite([value],1,'observed clearance')
            result.update(tilt_rad=tilt,max_effort_fraction=max(fractions.values()),
                          saturation_duration_s=duration,front_clearance_m=front_gap,hand_gap_m=hand_gap)
            for condition,reason in ((tilt>math.radians(15),'sensor_body_tilt_limit'),
                    (duration>=.5-1e-9,'sensor_effort_saturation'),
                    (front_gap is not None and front_gap<0.,'sensor_front_clearance_limit'),
                    (hand_gap is not None and hand_gap<0.,'sensor_hand_clearance_limit')):
                if condition:result['fault']=result['fault'] or reason
        except (KeyError,TypeError,ValueError):
            result['fault']=result['fault'] or 'invalid_sensor_measurement'
        self.latched=result['fault']
        self.result=result
        return dict(result)

    def fault(self,now_s):
        if self.result['time_s'] is None:return 'sensor_guard_uninitialized'
        if abs(now_s-self.result['time_s'])>1e-8:return 'sensor_guard_stale'
        return self.result['fault']
