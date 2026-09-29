"""Disposable source-only turn guard; declared envelopes, not hardware bounds."""
import itertools
import numpy as np
from g1cap.source_stop import SourceSides
from g1cap.sensor_robot_geometry import RobotGeometry
from ._batched_kinematics import frames_in_body
from g1cap.hand_sensors import measured_joint_positions

class TurnPlane:
    """Enclose at most +/-0.05rad additional yaw, plus articulated deviations.

    Fixed query-body frame. The 10cm front/2cm down/12cm total assumptions
    include camera age, gait, tracking and stopping. None is calibrated.
    Radius times yaw chord bounds displacement; normal perpendicular to floor
    up determines nominal closing, so level-top tangent yaw loses no height.
    """
    def __init__(self,source,rotation,up):
        if not isinstance(source,SourceSides) or source.memory.invalid:
            raise ValueError('turn_source_sides_unavailable')
        self.source=source;self.rotation=np.asarray(rotation);self.up=np.asarray(up)
        if self.up.shape!=(3,) or not np.isfinite(self.up).all() or abs(np.linalg.norm(self.up)-1)>1e-6:
            raise ValueError('turn_floor_up_unavailable')
        top=(self.rotation.T@source.top.rotation@source.top.normal) if source.top is not None else None
        pose=np.asarray(source.motion['body_in_segment']);R=pose[:3,:3]@source.gyro.T@self.rotation
        front=R.T@source.memory.normal
        self.top_tangent=np.linalg.norm(top-self.up*(top@self.up)) if top is not None else 0.
        self.front_tangent=np.linalg.norm(front-self.up*(front@self.up))
        self.chord=2*np.sin(.05/2)
        self.lipschitz=source.lipschitz*(1+self.chord)
    def bounds(self,points,now,rotation):
        if not np.allclose(rotation,self.rotation,atol=1e-10):raise ValueError('turn_query_frame_changed')
        points=np.asarray(points);radius=max(np.linalg.norm(points-np.outer(points@self.up,self.up),axis=1))
        travel=radius*self.chord
        return self.source.bounds(points,now,rotation,
            front_closing_m=.10+self.front_tangent*travel,
            top_closing_m=.02+self.top_tangent*travel,
            point_displacement_m=.12+travel)

class SourceTurnProbe:
    def __init__(self,urdf,bounds_file,names):
        self.geometry=RobotGeometry(urdf,bounds_file,names)
        if self.geometry.spheres:raise ValueError('turn_requires_measured_fingers')
        self.names=list(names)
    def screen(self,stop,packet,hands,rotation,seed,proposed,previous):
        now=packet['time_s']
        if not 0<=now-seed['time_s']<=.050001:raise ValueError('turn_probe_camera_stale')
        if stop.retention is None or not stop.retention['available'] or not 0<=now-stop.retention['time_s']<=.050001:
            raise ValueError('turn_probe_retention_unavailable')
        if (len(proposed)!=50 or not np.isfinite(proposed).all() or any(abs(v)>1e-12 for v in proposed[43:45])
                or abs(proposed[45])>.180001):raise ValueError('turn_probe_navigation_outside_envelope')
        if len(previous)!=50 or not np.isfinite(previous).all():raise ValueError('turn_probe_previous_command_invalid')
        arm_indices={self.names.index(name) for name in stop.path.arm_names}
        if any(abs(proposed[i]-previous[i])>1e-12 for i in range(50)
               if i not in arm_indices and i not in (43,44,45)):
            raise ValueError('turn_probe_nonarm_reference_changed')
        if stop.up_world is None:raise ValueError('turn_probe_floor_unavailable')
        plane=TurnPlane(stop.plane,rotation,np.asarray(rotation).T@stop.up_world)
        q=np.asarray(measured_joint_positions(packet,hands,stop.model_names));requested=q.copy()
        for name in stop.path.arm_names:
            requested[stop.path.model.joints[stop.path.model.getJointId(name)].idx_q]=proposed[self.names.index(name)]
        path=stop.path.screen(q,requested,plane,now,rotation)
        arm_lower=path['lower_m']-(plane.lipschitz-1)*path['interpolation_allowance_m']
        joints=dict(zip(stop.model_names,q));groups={}
        transforms=frames_in_body(self.geometry.model,self.geometry.rigid,joints)
        for link in self.geometry.rigid:
            T=transforms[link];points=[]
            for bound in self.geometry.bounds[link]:
                corners=np.array(list(itertools.product(*zip(bound['min'],bound['max']))))
                points.extend(corners@T[:3,:3].T+T[:3,3])
            if points:groups[link]=np.array(points)
        box=seed['box']
        if box['status']!='accepted':raise ValueError('turn_probe_box_unavailable')
        half=np.asarray(box['dimensions_m'])/2
        corners=np.array(list(itertools.product(*zip(-half,half))))@np.asarray(box['axes_camera']).T+box['center_camera_m']
        C=np.asarray(seed['camera_transform']);points=corners@C[:3,:3].T+C[:3,3]
        groups['observed_box']=points@np.asarray(seed['gyro_rotation']).T@np.asarray(rotation)
        margins={name:float(min(plane.bounds(points,now,rotation))) for name,points in groups.items()}
        margins['proposed_arm_path']=arm_lower
        limiting=min(margins,key=margins.get)
        return dict(status='clear' if margins[limiting]>=.03 else 'insufficient_margin',
            lower_m=margins[limiting],limiting_part=limiting,part_lower_m=margins,
            time_s=now,observed_at_s=seed['time_s'],path=path,
            assumed_front_closing_m=.10,assumed_top_closing_m=.02,
            assumed_point_displacement_m=.12,assumed_remaining_yaw_rad=.05,
            source_only=True,destination_clearance_qualified=False)
