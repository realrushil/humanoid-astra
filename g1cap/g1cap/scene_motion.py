"""Scene motion from RGB-D floor geometry, encoders and gyro; no contact truth.

G1 rev1.0 sole sphere centers are known robot geometry in ankle-roll coordinates
(metres), with 5 mm radii. Proximity within 8 mm is a NO-SLIP HYPOTHESIS, not
measured contact/load. Rolling spheres, sliding, calibration error and invisible
floors can invalidate it. Consensus limits and timing are simulation assumptions.
"""
from collections import deque
import math
import numpy as np

SOLE_POINTS=np.array([[-.05,.025,-.03],[-.05,-.025,-.03],[.12,.03,-.03],[.12,-.03,-.03]])


def combine_grasp_motion(grasp,scene):
    """Keep local grasp geometry usable during acquisition if odometry is lost.

    Scene-relative completion still requires both observations. An active wrist
    hold separately requires its original continuous motion segment.
    """
    quiet=(scene['status']=='available' and all(scene.get(k,float('inf'))<=limit for k,limit in
        [('max_box_speed_m_s',.05),('max_box_rotation_rad_s',.2),('max_body_planar_speed_m_s',.05)]))
    return dict(grasp,wrist_relative_ready=grasp.get('ready',False),
        settled_retention=bool(grasp.get('retention_stable',False) and quiet),
        scene_status=scene['status'],ready=(grasp['status']=='available' and
            grasp.get('ready',False) and scene['status']=='available' and scene['ready']))


def _rigid(value):
    t=np.asarray(value,float)
    if (t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1])
            or not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5)
            or not np.isclose(np.linalg.det(t[:3,:3]),1.,atol=1e-5)):
        raise ValueError('expected a finite proper rigid transform')
    return t


def _vector(value):
    v=np.asarray(value,float)
    if v.shape!=(3,) or not np.isfinite(v).all():raise ValueError('expected three finite values')
    return v


def _time(value):
    if isinstance(value,bool) or not math.isfinite(value) or value<0:raise ValueError('invalid time')
    return float(value)


def observed_floor(planes,camera_in_body,feet_in_body,up_body):
    """Select an observed level plane near both modeled ankles, in body metres.

    Ankle height 5 mm–20 cm is an explicit flat-ground/swing-height envelope.
    Planes must come from the current RGB-D frame; no stored table height enters.
    """
    cam=_rigid(camera_in_body);feet=[_rigid(f) for f in feet_in_body]
    if len(feet)!=2:raise ValueError('two foot transforms required')
    ankles=np.array([f[:3,3] for f in feet]);up=_vector(up_body)
    if np.linalg.norm(up)<1e-8:raise ValueError('zero up direction')
    up=up/np.linalg.norm(up);candidates=[]
    for plane in planes:
        n=cam[:3,:3]@plane['normal'];d=plane['offset_m']-n@cam[:3,3]
        if n@up<0:n,d=-n,-d
        if n@up<math.cos(math.radians(20)):continue
        heights=ankles@n+d
        if heights.min()<.005 or heights.max()>.20:continue
        candidates.append(dict(status='observed_candidate',normal_body=n.tolist(),offset_m=float(d)))
    if len(candidates)!=1:return dict(status='unavailable',reason='floor_plane_ambiguity')
    return candidates[0]


class StanceMotion:
    """Gyro at 50 Hz, point hypotheses at camera time (nominally 10 Hz).

    body_in_segment maps body points into axes fixed at the first IMU packet,
    with origin at the pelvis when this local segment initializes. Segment IDs
    change after lost support/time continuity; poses from different segments
    cannot be combined without an independently observed alignment.
    """
    def __init__(self):
        self.rotation=np.eye(3);self.imu_time=None;self.gyro=None
        self.previous=None;self.position=np.zeros(3);self.segment=0;self.active=False

    def advance_imu(self,time_s,gyro):
        t=_time(time_s);omega=_vector(gyro)
        if self.imu_time is not None:
            dt=t-self.imu_time
            if not 0<dt<=.1+1e-9:raise ValueError('IMU time must advance by at most 0.1 s')
            v=(omega+self.gyro)*(.5*dt);angle=np.linalg.norm(v)
            if angle>1e-12:
                x,y,z=v/angle;k=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
                self.rotation=self.rotation@(np.eye(3)+np.sin(angle)*k+(1-np.cos(angle))*k@k)
        self.imu_time=t;self.gyro=omega.copy()

    def update(self,time_s,feet_in_body,floor):
        t=_time(time_s)
        if self.imu_time is None or abs(t-self.imu_time)>1e-8:
            raise ValueError('foot geometry requires camera-time IMU orientation')
        if self.previous is not None:
            dt=t-self.previous['time_s']
            if dt<=0:raise ValueError('camera time must increase')
            if dt>.150001:self.previous=None;self.active=False
        feet=[_rigid(f) for f in feet_in_body]
        if len(feet)!=2:raise ValueError('two foot transforms required')
        points=np.concatenate([SOLE_POINTS@f[:3,:3].T+f[:3,3] for f in feet])
        gaps=None;eligible=np.zeros(8,dtype=bool)
        if floor['status']=='observed_candidate':
            n=_vector(floor['normal_body']);d=float(floor['offset_m'])
            if not np.isfinite(d) or not np.isclose(np.linalg.norm(n),1.,atol=1e-5):raise ValueError('invalid floor plane')
            gaps=points@n+d-.005;eligible=np.abs(gaps)<.008
        common=np.flatnonzero(eligible&self.previous['eligible']) if self.previous is not None else np.array([],int)
        delta=None;disagreement=None
        if len(common):
            suggestions=(self.previous['points'][common]@self.previous['rotation'].T
                         -points[common]@self.rotation.T)
            residual=np.linalg.norm(suggestions-np.median(suggestions,axis=0),axis=1)
            keep=residual<.010;disagreement=float(residual.max())
            if np.count_nonzero(keep)>=max(1,len(common)/2):
                delta=suggestions[keep].mean(axis=0);common=common[keep]
            else:common=np.array([],int)
        pose=None;status='unavailable'
        if len(common):
            if not self.active:
                self.segment+=1;self.position=np.zeros(3);self.active=True;status='initialized_local_segment'
            else:self.position+=delta;status='tracked_local_segment'
            pose=np.eye(4);pose[:3,:3]=self.rotation;pose[:3,3]=self.position
        else:self.active=False
        self.previous=dict(time_s=t,eligible=eligible.copy(),points=points.copy(),rotation=self.rotation.copy())
        return dict(status=status,time_s=t,segment=self.segment,
            body_in_segment=pose.tolist() if pose is not None else None,
            translation_increment_initial_axes_m=delta.tolist() if delta is not None else None,
            gaps_m=gaps.reshape(2,4).tolist() if gaps is not None else None,
            eligible=eligible.tolist(),used_points=common.tolist(),foot_disagreement_m=disagreement,
            assumption='near-floor no-slip sole points; not measured contact')


class SceneStability:
    """One second of scene-relative motion plus independent visual grasp readiness.

    Camera-rate differences can miss faster peaks. 5 cm/s translation and 0.2 rad/s
    rotation limits do not establish equivalence with the 50 Hz truth scorer.
    """
    def __init__(self):
        self.history=deque();self.segment=None;self.up=None

    def update(self,time_s,motion,box,camera_in_body,up_body,*,grasp_ready):
        t=_time(time_s)
        result=dict(status='unavailable',time_s=t,ready=False,reason='scene_motion_unavailable',segment=motion['segment'])
        if motion['status']=='unavailable' or box['status']!='accepted':
            self.history.clear();return result
        if abs(motion['time_s']-t)>1e-8 or abs(box['observed_at_s']-t)>1e-8:
            raise ValueError('scene stability requires same-time body and box estimates')
        body=_rigid(motion['body_in_segment']);cam=body@_rigid(camera_in_body)
        if self.history:
            dt=t-self.history[-1]['time_s']
            if dt<=0:raise ValueError('scene time must increase')
            if dt>.150001:self.history.clear()
        if motion['segment']!=self.segment:
            self.history.clear();self.segment=motion['segment'];self.up=body[:3,:3]@_vector(up_body)
            if np.linalg.norm(self.up)<1e-8:raise ValueError('zero up direction')
            self.up/=np.linalg.norm(self.up)
        self.history.append(dict(time_s=t,box_position=cam[:3,:3]@box['center_camera_m']+cam[:3,3],
            box_rotation=cam[:3,:3]@box['axes_camera'],body_position=body[:3,3].copy()))
        # Preserve the sample before the one-second boundary for full coverage.
        while len(self.history)>2 and self.history[1]['time_s']<=t-1.+1e-9:
            self.history.popleft()
        result.update(status='available',reason='need_one_second')
        if len(self.history)>=2 and t-self.history[0]['time_s']>=.999:
            speed=[];angular=[];body_speed=[]
            for a,b in zip(self.history,list(self.history)[1:]):
                dt=b['time_s']-a['time_s']
                speed.append(float(np.linalg.norm(b['box_position']-a['box_position'])/dt))
                angular.append(float(math.acos(np.clip((np.trace(a['box_rotation'].T@b['box_rotation'])-1)/2,-1,1))/dt))
                d=b['body_position']-a['body_position'];body_speed.append(float(np.linalg.norm(d-self.up*(self.up@d))/dt))
            ready=bool(grasp_ready and max(speed)<=.05 and max(angular)<=.2 and max(body_speed)<=.05)
            result.update(ready=ready,reason='scene_and_grasp_stable' if ready else 'scene_or_grasp_unsettled',
                max_box_speed_m_s=max(speed),max_box_rotation_rad_s=max(angular),max_body_planar_speed_m_s=max(body_speed))
        return result
