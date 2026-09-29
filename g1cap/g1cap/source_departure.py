"""Bounded source departure state machine; experimental, not wired to motors.

Inputs must be current sensor-derived evidence. drive_clear admits the existing live checks or the separate
complete current-command travel certificate; a failed full-turn sweep cannot authorize
translation by itself. Distances are metres, yaw radians, clocks seconds.
"""
import math

def finite(x):
    return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)

class SourceDeparture:
    def __init__(self,now,yaw_rad,sample,*,planning_margin_m=.03,max_duration_s=12.):
        if not finite(now) or now<0 or not finite(yaw_rad) or not 0<abs(yaw_rad)<=math.pi/2:
            raise ValueError('invalid_departure_request')
        if not finite(planning_margin_m) or not .03<=planning_margin_m<=.20:
            raise ValueError('invalid_departure_planning_margin')
        if not finite(max_duration_s) or not 0<max_duration_s<=30:
            raise ValueError('invalid_departure_duration')
        self.max_duration=float(max_duration_s)
        self.planning_margin=float(planning_margin_m)
        self.started=self.previous=now;self.yaw=yaw_rad
        self.identity=(sample.get('track_epoch'),sample.get('segment'))
        if None in self.identity or sample.get('retained') is not True:
            raise ValueError('departure_initial_retention_unavailable')
        self.phase='drive';self.stopped_at=None;self.ready_since=None
        self.last_capture=None;self.speed=0.;self.outcome=None;self.reason=None
    def _finish(self,reason):
        self.outcome='failed';self.reason=reason;self.phase='stopped';self.speed=0.
    def update(self,now,sample):
        if self.outcome:return self.result()
        capture=sample.get('time_s')
        if (not finite(now) or now<self.previous or not finite(capture)
                or not 0<=now-capture<=.150001
                or (self.last_capture is not None and (capture<self.last_capture or capture-self.last_capture>.150001))):
            self._finish('departure_observation_clock_invalid');return self.result()
        if (sample.get('retained') is not True
                or (sample.get('track_epoch'),sample.get('segment'))!=self.identity):
            self._finish('departure_retention_discontinuity');return self.result()
        dt=now-self.previous;self.previous=now
        if now-self.started>self.max_duration:
            self._finish('departure_timeout');return self.result()
        sweep=sample.get('sweep',{})
        valid=(finite(sweep.get('time_s')) and abs(sweep['time_s']-capture)<1e-8
               and finite(sweep.get('yaw_rad')) and abs(sweep['yaw_rad']-self.yaw)<1e-8
               and finite(sweep.get('lower_m'))
               and sweep.get('status') in ('clear','insufficient_margin'))
        clear=valid and sweep['status']=='clear' and sweep['lower_m']>=self.planning_margin
        normal=sample.get('front_normal_xy',[])
        direction=(len(normal)==2 and all(finite(v) for v in normal)
                   and abs(math.hypot(*normal)-1.)<=.001)
        drive=(valid and sample.get('direction_available') is True
               and sample.get('drive_clear') is True and direction)
        # One departure, then settle. No automatic restart after visibility loss.
        if self.phase=='drive' and (clear or not drive):
            self.phase='settle';self.stopped_at=now;self.speed=0.
        if self.phase=='drive':
            self.speed=min(.08,self.speed+.3*min(dt,.02))
            self.navigation=[-self.speed*normal[0],-self.speed*normal[1],0.]
        else:
            self.navigation=[0.,0.,0.]
            ready=clear and sample.get('settled') is True
            if not ready:self.ready_since=None
            elif capture!=self.last_capture:
                if self.ready_since is None:self.ready_since=capture
                if capture-self.ready_since>=1.-1e-8:
                    self.outcome='completed';self.reason='departure_turn_clearance_and_hold_verified'
            if self.outcome is None and now-self.stopped_at>=5.:
                self._finish('departure_did_not_settle_with_clearance')
        self.last_capture=capture
        return self.result()
    def result(self):
        navigation=getattr(self,'navigation',[0.,0.,0.]) if self.outcome is None else [0.,0.,0.]
        return dict(phase=self.phase,navigation=navigation,outcome=self.outcome,reason=self.reason,
                    requested_yaw_rad=self.yaw,evidence_start_s=self.ready_since,planning_margin_m=self.planning_margin)
