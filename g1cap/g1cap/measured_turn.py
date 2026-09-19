"""Bounded sensor yaw objective from measured floor/IMU heading only.

Radian angles use right-hand rotation about observed control-frame up. No XY
localization, obstacle map or contact measurement is implied. The caller keeps
all geometry/grasp/effort guards and the sole whole-body actuator owner active.
"""
import math
import numpy as np

class HeadingReference:
    """Relative planar heading about measured floor up; no XY localization."""
    def __init__(self,body_rotation,up):
        self.up=np.asarray(up,float).copy()
        if self.up.shape!=(3,) or not np.isfinite(self.up).all() or np.linalg.norm(self.up)<.5:
            raise ValueError('invalid_turn_up')
        self.up/=np.linalg.norm(self.up)
        self.initial=self.heading(body_rotation)

    def heading(self,rotation):
        rotation=np.asarray(rotation,float)
        if rotation.shape!=(3,3) or not np.isfinite(rotation).all():
            raise ValueError('invalid_turn_rotation')
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6) or np.linalg.det(rotation)<0:
            raise ValueError('invalid_turn_rotation')
        direction=rotation[:,0]-self.up*(self.up@rotation[:,0])
        if np.linalg.norm(direction)<.5:raise ValueError('turn_heading_unavailable')
        return direction/np.linalg.norm(direction)

    def angle(self,rotation):
        forward=self.heading(rotation)
        return math.atan2(self.up@np.cross(self.initial,forward),self.initial@forward)

    def rotation(self,angle):
        x,y,z=self.up;cross=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        return np.eye(3)+math.sin(angle)*cross+(1-math.cos(angle))*(cross@cross)


class MeasuredHeadingTurn:
    def __init__(self,time_s,angle_rad,body_rotation,up,minimum_yaw_rate=.04,
                 require_sustained_readiness=False):
        if not math.isfinite(angle_rad) or not 0<abs(angle_rad)<=math.pi/2:
            raise ValueError('turn_angle_outside_envelope')
        if isinstance(minimum_yaw_rate,bool) or not math.isfinite(minimum_yaw_rate) or not 0<minimum_yaw_rate<=.18:
            raise ValueError('invalid_minimum_yaw_rate')
        self.minimum_rate=float(minimum_yaw_rate)
        self.started=float(time_s);self.goal=float(angle_rad);self.time=None
        self.reference=HeadingReference(body_rotation,up)
        self.yaw=0.;self.settling=None;self.refinements=0
        self.outcome=self.reason=None
        self.require_sustained_readiness=require_sustained_readiness
        self.ready_times=[];self.ready_camera=None

    def rotation(self):
        return self.reference.rotation(self.yaw)

    def fresh_ready(self,time_s,observed_at_s,ready,error):
        """One second and11 distinct current camera samples; never tick counts."""
        if (observed_at_s is None or not math.isfinite(observed_at_s)
                or observed_at_s>time_s+1e-8 or time_s-observed_at_s>.150001
                or self.ready_camera is not None and observed_at_s<self.ready_camera-1e-8):
            raise ValueError('turn_ready_camera_time_invalid')
        previous=self.ready_camera;self.ready_camera=observed_at_s
        if (not ready or abs(error)>.05 or self.settling is None
                or observed_at_s<self.settling-1e-8):
            self.ready_times=[]
        elif previous is None or observed_at_s>previous+1e-8:
            if previous is not None and observed_at_s-previous>.150001:self.ready_times=[]
            self.ready_times.append(observed_at_s)
        return len(self.ready_times)>=11 and self.ready_times[-1]-self.ready_times[0]>=1.-1e-8

    def update(self,time_s,body_rotation,ready,observed_at_s=None):
        dt=time_s-(self.time if self.time is not None else self.started)
        if not math.isfinite(time_s) or not 0<=dt<=.020001 or (self.time is not None and dt<=0):
            raise ValueError('turn_measurement_time_gap')
        self.time=time_s
        self.yaw=self.reference.angle(body_rotation)
        error=self.goal-self.yaw;rate=0.
        if self.outcome is None:
            if time_s-self.started>=10.-1e-8:
                self.outcome,self.reason='failed','measured_turn_timeout'
            else:
                if self.settling is None and abs(error)<=.02:self.settling=time_s
                stable_ready=(self.fresh_ready(time_s,observed_at_s,ready,error)
                              if self.require_sustained_readiness else ready)
                if self.settling is not None and time_s-self.settling>=1.-1e-8:
                    if abs(error)<=.05 and stable_ready:
                        self.outcome,self.reason='completed','measured_heading_and_hold'
                    elif abs(error)>.05:
                        if self.refinements>=2:
                            self.outcome,self.reason='failed','turn_refinement_budget_exhausted'
                        else:self.settling=None;self.refinements+=1
                if self.outcome is None and self.settling is None:
                    rate=math.copysign(min(.18,max(self.minimum_rate,.8*abs(error))),error)
        result=dict(time_s=time_s,yaw_rad=self.yaw,target_rad=self.goal,error_rad=error,
                    yaw_rate_command_rad_s=rate,phase='turn_settle' if self.settling is not None else 'turn_drive',
                    outcome=self.outcome,reason=self.reason,refinements=self.refinements)
        if self.require_sustained_readiness:
            result['ready_window']=dict(fresh_samples=len(self.ready_times),
                first_camera_s=self.ready_times[0] if self.ready_times else None,
                last_camera_s=self.ready_times[-1] if self.ready_times else None)
        return result
