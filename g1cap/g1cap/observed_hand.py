"""Hand clearance to an observed tabletop/front half-plane, in pelvis metres.

Body/Dex3 geometry updates at encoder rate. Gyro rotates the observed planes;
only recent closing translation is extrapolated between images. Unknown table
sides/legs, acceleration, slip and calibration error are not bounded here.
"""
import itertools
import math
import numpy as np
from .observed_approach import vector


def clipped_gap(points,up,top_offset,front,front_offset):
    """Lowest box-corner hull point reaching the front's 3 cm development band."""
    points=np.asarray(points,float)
    signed=points@front+front_offset+.03
    if np.max(signed)<0:return None
    heights=points@up+top_offset
    minimum=float(np.min(heights[signed>=0]))
    if np.min(signed)<0:
        # All corner pairs include every polytope edge. Extra internal chords
        # remain inside the hull and cannot invent a lower feasible point.
        edges=np.array(list(itertools.combinations(range(len(points)),2)))
        a,b=edges.T;cross=signed[a]*signed[b]<0;a,b=a[cross],b[cross]
        if len(a):
            fraction=-signed[a]/(signed[b]-signed[a])
            minimum=min(minimum,float(np.min(heights[a]+fraction*(heights[b]-heights[a]))))
    return minimum


def hand_shapes(model,bounds,body,hands):
    """Known collision bounds placed only by synchronized measured joint angles."""
    from .hand_sensors import measured_joint_positions
    from .sensor_kinematics import frame_in_body
    names=list(model.names)[1:]
    joints=dict(zip(names,measured_joint_positions(body,hands,names),strict=True))
    result=[]
    for name,shapes in bounds.items():
        if 'hand' not in name and 'wrist' not in name:continue
        pose=frame_in_body(model,name,joints)
        for shape in shapes:
            points=np.array(list(itertools.product(*zip(shape['min'],shape['max']))))
            result.append(dict(name=name,points=points@pose[:3,:3].T+pose[:3,3]))
    return result


class HandClearanceEstimate:
    def __init__(self):
        self.time=self.gyro=None
        self.shapes=[]
        self.invalidate('initializing_hand_clearance')

    def invalidate(self,reason):
        self.observed_at=None
        self.up=self.front=self.offsets=self.rates=None
        self.reason=reason

    def advance(self,time_s,gyro,shapes):
        omega=vector(gyro)
        if not math.isfinite(time_s):raise ValueError('invalid hand measurement time')
        if not shapes:raise ValueError('measured hand geometry required')
        for shape in shapes:
            points=np.asarray(shape['points'])
            if points.ndim!=2 or points.shape[1]!=3 or len(points)<2 or not np.isfinite(points).all():
                raise ValueError('invalid measured hand bounds')
        if self.time is not None:
            dt=time_s-self.time
            if dt<=0:raise ValueError('hand measurement time must advance')
            if dt>.100001:self.invalidate('hand_time_gap')
            elif self.up is not None:
                turn=-.5*(self.gyro+omega)*dt;angle=np.linalg.norm(turn)
                if angle>1e-12:
                    axis=turn/angle
                    def rotate(v):
                        return v*np.cos(angle)+np.cross(axis,v)*np.sin(angle)+axis*(axis@v)*(1-np.cos(angle))
                    self.up,self.front=rotate(self.up),rotate(self.front)
        self.time,self.gyro=float(time_s),omega.copy()
        self.shapes=[dict(name=s['name'],points=np.asarray(s['points'],float).copy()) for s in shapes]

    def observe(self,time_s,up,top_offset,front,front_offset):
        up,front=vector(up),vector(front)
        if (self.time is None or abs(time_s-self.time)>1e-8 or
                not all(math.isfinite(v) for v in [top_offset,front_offset]) or
                abs(np.linalg.norm(up)-1)>.001 or abs(np.linalg.norm(front)-1)>.001):
            raise ValueError('invalid or unsynchronized hand planes')
        if abs(up@front)>.2:self.invalidate('inconsistent_hand_planes');return
        offsets=np.array([top_offset,front_offset],float);rates=None
        if self.observed_at is not None:
            dt=time_s-self.observed_at
            if dt<=0:raise ValueError('hand camera time must advance')
            if dt<=.150001:rates=(offsets-self.offsets)/dt
        self.observed_at=float(time_s);self.offsets=offsets;self.rates=rates
        self.up,self.front=up,front;self.reason='initializing_plane_motion'

    def feedback(self,now_s):
        if self.rates is None:return dict(status='unavailable',reason=self.reason)
        age=now_s-self.observed_at
        if not math.isfinite(now_s) or abs(now_s-self.time)>1e-8:
            return dict(status='unavailable',reason='unsynchronized_hand_measurement')
        if not 0<=age<=.150001:return dict(status='unavailable',reason='stale_hand_planes')
        # Smaller top offset means the pelvis descended. Larger front offset
        # means it approached the table. Never credit inferred retreat/lift.
        top_offset=self.offsets[0]+min(0.,self.rates[0])*age
        front_offset=self.offsets[1]+max(0.,self.rates[1])*age
        margins=[]
        for shape in self.shapes:
            margin=clipped_gap(shape['points'],self.up,top_offset,self.front,front_offset)
            if margin is not None:margins.append((margin,shape['name']))
        margin,name=min(margins) if margins else (None,None)
        return dict(status='available',time_s=now_s,observed_at_s=self.observed_at,age_s=age,
            margin_m=margin,limiting_link=name,up_body=self.up.tolist(),
            top_offset_rate_m_s=float(self.rates[0]),front_offset_rate_m_s=float(self.rates[1]))


def observe_hand_planes(estimate,model,body,camera,support,front):
    """Reuse the synchronized RGB-D-derived support/front; never scene metadata."""
    from .sensor_kinematics import camera_in_body
    if body['step']!=camera['step'] or abs(body['time_s']-camera['time_s'])>1e-8:
        raise ValueError('hand camera/body timestamps differ')
    if support is None or support['status']!='observed_candidate' or front['status']!='observed_candidate':
        estimate.invalidate('hand_support_or_front_unavailable');return
    if abs(front['observed_at_s']-body['time_s'])>1e-8:raise ValueError('hand front timestamp differs')
    joints=dict(zip(body['joint_names'],body['q_rad'],strict=True))
    transform=camera_in_body(model,joints,camera['calibration'])
    up=transform[:3,:3]@np.asarray(support['normal_camera'])
    offset=float(support['offset_m']-up@transform[:3,3])
    estimate.observe(body['time_s'],up,offset,front['normal_body'],front['offset_body_m'])
