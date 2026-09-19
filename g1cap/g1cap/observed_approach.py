"""Camera-time front clearance with gyro-updated direction; metres/seconds.

This is a local command limiter, not collision planning or certified braking.
The plane is sampled at camera time; leg geometry updates with each encoder
packet. Measured plane-offset speed predicts pelvis translation during its age;
unobserved acceleration, slip, calibration error and occluded legs are not
bounded. The existing 3 cm margin / 0.1 s lookahead remain uncalibrated.
"""
import math
import numpy as np
from .toolkit.arena_approach import MARGIN, LOOKAHEAD, APPROACH_TIME


def vector(value):
    result=np.asarray(value,float)
    if result.shape!=(3,) or not np.isfinite(result).all():
        raise ValueError('expected three finite values')
    return result


class ApproachEstimate:
    def __init__(self):
        self.time=None
        self.gyro=None
        self.previous=None
        self.normal=self.up=None
        self.closing=None
        self.origin_closing=None
        self.points=None
        self.reason='initializing_front'

    def invalidate(self,reason):
        self.previous=None
        self.normal=self.up=None
        self.closing=None
        self.origin_closing=None
        self.reason=reason

    def advance_imu(self,time_s,gyro,points_body):
        omega=vector(gyro)
        points=np.asarray(points_body,float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)==0 or not np.isfinite(points).all():
            raise ValueError('current calibrated lower-body points required')
        if not math.isfinite(time_s):raise ValueError('invalid IMU time')
        if self.time is not None:
            dt=time_s-self.time
            if dt<=0:raise ValueError('IMU time must advance')
            if dt>.100001:self.invalidate('imu_time_gap')
            elif self.normal is not None:
                # A scene-fixed normal rotates opposite the measured body.
                turn=-.5*(self.gyro+omega)*dt
                angle=np.linalg.norm(turn)
                if angle>1e-12:
                    axis=turn/angle
                    def rotate(v):
                        return (v*np.cos(angle)+np.cross(axis,v)*np.sin(angle)
                                +axis*(axis@v)*(1-np.cos(angle)))
                    self.normal,self.up=rotate(self.normal),rotate(self.up)
        self.time,self.gyro=float(time_s),omega.copy()
        self.points=points.copy()

    def observe(self,time_s,offset_body_m,normal_body,up_body):
        normal,up=vector(normal_body),vector(up_body)
        if (self.time is None or abs(time_s-self.time)>1e-8 or
                not math.isfinite(offset_body_m) or min(np.linalg.norm(normal),np.linalg.norm(up))<1e-8):
            raise ValueError('invalid or unsynchronized front observation')
        normal=normal/np.linalg.norm(normal);up=up/np.linalg.norm(up)
        if abs(normal@up)>.2:
            self.invalidate('front_not_horizontal');return
        self.closing=None
        self.origin_closing=None
        clearance_m=-float(np.max(self.points@normal))-offset_body_m
        if self.previous is not None:
            t,clearance,offset=self.previous
            if time_s<=t:raise ValueError('camera time must advance')
            if time_s-t<=.150001:
                self.closing=(clearance-clearance_m)/(time_s-t)
                self.origin_closing=(offset_body_m-offset)/(time_s-t)
        self.previous=(float(time_s),float(clearance_m),float(offset_body_m))
        self.normal,self.up=normal,up
        self.reason='initializing_closing_rate'

    def select_front(self,candidates,time_s):
        """Associate fresh lines in the current pelvis frame, never a cached edge.

        Initialize with one >=15 cm segment. A prior <=150 ms old permits one
        >=7.5 cm segment within 5 degrees and 3 cm of predicted normal/offset.
        Gyro propagates orientation; measured offset rate predicts translation.
        These gates assume ideal RGB-D and bounded inter-frame motion; they are
        not calibrated error bounds. Missing/ambiguous matches stay unavailable.
        """
        unavailable=dict(status='unavailable')
        if self.time is None or not math.isfinite(time_s) or abs(self.time-time_s)>1e-8:
            return dict(unavailable,reason='unsynchronized_imu')
        if self.previous is None or not 0<=time_s-self.previous[0]<=.150001:
            matching=[c for c in candidates if c['observed_span_m']>=.15]
            mode='initialize'
        else:
            age=time_s-self.previous[0]
            offset=self.previous[2]+(self.origin_closing or 0.)*age
            matching=[c for c in candidates if c['observed_span_m']>=.075
                      and np.dot(c['normal_body'],self.normal)>=np.cos(np.deg2rad(5.))
                      and abs(c['offset_body_m']-offset)<=.03]
            mode='track'
        if len(matching)!=1:
            reason=('ambiguous_tracked_front' if mode=='track' else 'ambiguous_front') if matching else 'front_not_visible'
            return dict(unavailable,reason=reason,matching_count=len(matching))
        return dict(status='observed_candidate',candidate=matching[0],mode=mode)

    def feedback(self,now_s):
        unavailable=dict(status='unavailable',reason=self.reason)
        if self.previous is None or self.closing is None:return unavailable
        t,observed_clearance,offset=self.previous;age=now_s-t
        if not math.isfinite(now_s) or abs(now_s-self.time)>1e-8:
            return dict(unavailable,reason='unsynchronized_imu')
        if not 0<=age<=.150001:return dict(unavailable,reason='stale_front')
        # Native navigation is horizontal forward/left, not tilted pelvis XY.
        forward=np.array([1.,0.,0.]);forward-=self.up*(forward@self.up)
        if np.linalg.norm(forward)<1e-6:return dict(unavailable,reason='invalid_heading')
        forward/=np.linalg.norm(forward);left=np.cross(self.up,forward)
        n=np.array([self.normal@forward,self.normal@left]);n/=np.linalg.norm(n)
        # Body-relative limb motion is measured, not extrapolated from the
        # previous image. Only pelvis translation toward the plane is predicted.
        clearance=-float(np.max(self.points@self.normal))-offset-max(0.,self.origin_closing)*age
        return dict(status='available',observed_at_s=t,age_s=age,clearance_m=clearance,
                    observed_clearance_m=observed_clearance,origin_closing_speed_m_s=self.origin_closing,
                    closing_speed_m_s=self.closing,normal_navigation_xy=n.tolist())

    def limit(self,command,now_s):
        info=self.feedback(now_s)
        if info['status']!='available':raise ValueError('sensor_approach_unavailable')
        command=vector(command);n=np.array(info['normal_navigation_xy']);tangent=np.array([-n[1],n[0]])
        reserve=(info['clearance_m']-MARGIN-
                 LOOKAHEAD*max(0.,info['closing_speed_m_s']))
        cap=max(0.,reserve/APPROACH_TIME)
        side_scale=np.clip(reserve/MARGIN,0.,1.)
        inward=command[:2]@n
        planar=n*min(inward,cap)+tangent*(command[:2]@tangent)*side_scale
        limited=[*planar,command[2]*side_scale]
        info.update(inward_cap_m_s=cap,requested_inward_m_s=float(inward),
                    lateral_yaw_scale=float(side_scale),
                    intervened=bool(np.max(np.abs(np.array(limited)-command))>1e-7))
        return limited,info


def observe_front(geometry,estimate,packet,camera,support,up_body):
    """One synchronized RGB-D/body observation; no task geometry is accepted."""
    from .rgbd_table import front_candidates
    if packet['step']!=camera['step'] or abs(packet['time_s']-camera['time_s'])>1e-8:
        raise ValueError('front camera/encoder timestamps differ')
    rgb=np.asarray(camera['rgb'],float)
    result=front_candidates(camera['depth_m'],camera['calibration']['intrinsic'],support,
        np.ptp(rgb,axis=2)<25,
        lambda points:geometry.exclude(points,packet,camera['calibration']),min_span=.075)
    if result['status']=='unavailable':
        estimate.invalidate(result['reason'])
        return result
    candidates=[]
    for line in result['lines']:
        normal,offset,clearance=geometry.front_clearance(line,packet,camera['calibration'])
        candidates.append(dict(line,normal_body=normal.tolist(),offset_body_m=offset,clearance_m=clearance))
    selected=estimate.select_front(candidates,packet['time_s'])
    if selected['status']=='unavailable':
        estimate.invalidate(selected['reason'])
        return selected
    candidate=selected['candidate']
    estimate.observe(packet['time_s'],candidate['offset_body_m'],candidate['normal_body'],up_body)
    line={k:v for k,v in candidate.items() if k not in ('normal_body','offset_body_m','clearance_m')}
    return dict(status='observed_candidate',line=line,planar_points=result['planar_points'],
                association=selected['mode'],observed_at_s=packet['time_s'],
                clearance_m=candidate['clearance_m'],normal_body=candidate['normal_body'],
                offset_body_m=candidate['offset_body_m'])
