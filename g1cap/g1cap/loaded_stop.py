"""Sensor-screened loaded stopping, using the existing paired wrist owner.

This module only evaluates a proposed arm reference. It does not write robot
commands, fit a second source tracker or infer grasp force. Bounds assume the
same static level floor/table, ideal gyro/calibration and the declared vertical
acceleration envelope. A reference-path screen is not a dynamic tracking tube.
"""
import itertools,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
from .scene_motion import _rigid

def rotation(value):
    pose=np.eye(4);pose[:3,:3]=np.asarray(value,float)
    return _rigid(pose)[:3,:3]

class TopPlane:
    def __init__(self,normal,offset,anchor,anchor_error,normal_difference,time_s,gyro_rotation):
        self.normal=np.asarray(normal,float);self.anchor=np.asarray(anchor,float)
        self.offset=float(offset);self.error=float(anchor_error);self.normal_difference=float(normal_difference)
        self.time=float(time_s);self.rotation=rotation(gyro_rotation)
        if (self.normal.shape!=(3,) or self.anchor.shape!=(3,) or self.rotation.shape!=(3,3)
                or not np.isfinite(np.r_[self.normal,self.anchor,self.offset,self.error,self.normal_difference,self.time]).all()
                or abs(np.linalg.norm(self.normal)-1)>1e-6 or min(self.error,self.normal_difference)<0):
            raise ValueError('invalid top-plane evidence')
    def bounds(self,points,now,gyro_rotation):
        age=now-self.time
        if not np.isfinite(age) or not 0<=age<=.150001:raise ValueError('top_plane_stale')
        relative=rotation(gyro_rotation).T@self.rotation
        normal=relative@self.normal;anchor=relative@self.anchor
        points=np.asarray(points,float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)==0 or not np.isfinite(points).all():raise ValueError('invalid query points')
        # Rotation and fit uncertainty only. The caller separately subtracts
        # the explicit vertical-motion allowance through age plus the next tick.
        allowance=self.error+self.normal_difference*np.linalg.norm(points-anchor,axis=1)
        return points@normal+self.offset-allowance

class ArmPath:
    def __init__(self,urdf,bounds_file):
        self.model=pin.buildModelFromUrdf(str(urdf));self.data=self.model.createData()
        self.arm_names=[n for n in self.model.names if any(k in n for k in ('shoulder','elbow','wrist'))]
        self.arm_ids=np.array([self.model.joints[self.model.getJointId(n)].idx_q for n in self.arm_names])
        bounds=json.loads(Path(bounds_file).read_text());parents={}
        for j in ET.parse(urdf).getroot().findall('joint'):
            parents[j.find('child').attrib['link']]=j
        self.shapes=[]
        for link,shapes in bounds.items():
            if not any(k in link for k in ('shoulder','elbow','wrist','hand')):continue
            for shape in shapes:
                corners=np.array(list(itertools.product(*zip(shape['min'],shape['max']))))
                # For each upstream revolute joint, radius to any link corner
                # is bounded by chain translation lengths + local corner norm.
                radius=max(np.linalg.norm(corners,axis=1));node=link;reach={}
                while node in parents:
                    j=parents[node];name=j.attrib['name']
                    if name in self.arm_names:reach[self.model.joints[self.model.getJointId(name)].idx_q]=radius
                    origin=j.find('origin');translation=np.fromstring(origin.attrib.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
                    radius+=np.linalg.norm(translation);node=j.find('parent').attrib['link']
                self.shapes.append((link,self.model.getFrameId(link),corners,reach))
    def points(self,q):
        pin.framesForwardKinematics(self.model,self.data,q)
        return [(link,corners@self.data.oMf[frame].rotation.T+self.data.oMf[frame].translation) for link,frame,corners,_ in self.shapes]
    def screen(self,measured,requested,plane,now,rotation,*,samples=17):
        if samples<2:raise ValueError('at least two path samples required')
        measured=np.asarray(measured,float);requested=np.asarray(requested,float)
        if (measured.shape!=(self.model.nq,) or requested.shape!=(self.model.nq,)
                or not np.isfinite(measured).all() or not np.isfinite(requested).all()):raise ValueError('invalid joint path')
        delta=requested-measured
        other=np.ones(self.model.nq,bool);other[self.arm_ids]=False
        if np.any(abs(delta[other])>1e-12):raise ValueError('only paired arms may change')
        allowances=[sum(radius*abs(delta[index]) for index,radius in reach.items())/(2*(samples-1)) for _,_,_,reach in self.shapes]
        lower=float('inf');limiting=None;sample_min=float('inf')
        for fraction in np.linspace(0,1,samples):
            for (link,points),allowance in zip(self.points(measured+fraction*delta),allowances):
                value=float(np.min(plane.bounds(points,now,rotation)))
                sample_min=min(sample_min,value)
                if value-allowance<lower:lower=value-allowance;limiting=link
        return dict(lower_m=lower,limiting_link=limiting,sampled_lower_m=sample_min,
                    interpolation_allowance_m=float(max(allowances)),samples=samples,
                    assumption='joint-linear arm reference path; measured waist/fingers fixed; not dynamic tracking tube')


class LoadedStopClearance:
    """Consume fresh source evidence; screen zero-navigation paired correction.

    Source identity belongs to SourcePlane. Retention is a visual hypothesis
    from opposed nearby wrists and attitude, independent of task-height query
    precision. Missing evidence cannot be replaced by old source geometry.
    """
    def __init__(self,urdf,bounds_file,action_joint_names):
        self.path=ArmPath(urdf,bounds_file);self.names=list(action_joint_names)
        self.model_names=list(self.path.model.names)[1:]
        self.source_time=self.camera_time=None
        self.up_world=None;self.acc=[];self.heights=[]
        self.plane=None;self.retention=None

    def measure(self,packet,seed,rotation,up_body,source):
        from .level_source_plane import floor_distances
        from .sensor_kinematics import box_in_robot_frames
        t=packet['time_s'];camera_t=seed['time_s'];rotation=np.asarray(rotation,float)
        if not np.isfinite([t,camera_t]).all() or not 0<=t-camera_t<=.150001:
            raise ValueError('stop_camera_time_unavailable')
        if self.acc and t<=self.acc[-1][0]:raise ValueError('stop_measurement_time_must_advance')
        camera=np.asarray(seed['camera_transform']);box=seed['box']
        if self.camera_time!=camera_t:
            self.camera_time=camera_t;self.plane=None
            floor=floor_distances(seed['floor_points'],[-camera[:3,:3].T@camera[:3,3]])
            self.retention=dict(time_s=camera_t,available=False)
            if floor['status']=='measured':
                n=camera[:3,:3]@np.array(floor['normal']);sign=1 if n@up_body>0 else -1
                if self.up_world is None:self.up_world=np.asarray(seed['gyro_rotation'])@(sign*n)
                self.heights.append([camera_t,sign*floor['distances_m'][0],floor['bounds_m'][0]])
            if box['status']=='accepted':
                relative=box_in_robot_frames(self.path.model,box,seed['packet'],seed['calibration'])
                center=np.asarray(relative['box_center_pelvis_m']);axes=np.asarray(relative['box_axes_pelvis'])
                wrists=[(np.asarray(relative['wrist_positions_pelvis_m'][side])-center)@axes for side in ('left','right')]
                axis=int(np.argmax(abs(wrists[0]-wrists[1])))
                opposed=wrists[0][axis]*wrists[1][axis]<0 and abs(wrists[0][axis]-wrists[1][axis])>=box['dimensions_m'][axis]
                near=max(np.linalg.norm(p) for p in wrists)<=.35
                attitude=np.asarray(up_body)[2]>=np.cos(np.deg2rad(15))
                self.retention=dict(time_s=camera_t,available=bool(opposed and near and attitude),
                    opposing_near_wrists=bool(opposed and near),attitude_ok=bool(attitude))
        if self.up_world is not None:
            self.acc.append([t,float(self.up_world@rotation@np.array(packet['specific_force_m_s2'])-9.81)])
        if source is None:
            self.plane=None;self.source_time=None
        elif self.source_time!=camera_t:
            self.source_time=camera_t
            if source['observed_at_s']!=camera_t:raise ValueError('stop_source_time_mismatch')
            if source['status']=='observed_candidate' and box['status']=='accepted':
                quality=source['prediction_quality'];normal=camera[:3,:3]@np.array(source['normal_camera'])
                anchor=camera[:3,:3]@np.array(box['center_camera_m'])+camera[:3,3]
                self.plane=TopPlane(normal,source['offset_m']-normal@camera[:3,3],anchor,
                    quality['height_perturbation_m'],2*np.sin(np.deg2rad(quality['floor_normal_perturbation_deg'])/2),
                    camera_t,seed['gyro_rotation'])
        self.acc=[row for row in self.acc if row[0]>=t-1.]
        self.heights=[row for row in self.heights if row[0]>=t-1.]

    def screen(self,packet,hands,rotation,proposed):
        from .hand_sensors import measured_joint_positions
        from .vertical_motion import predict_vertical
        t=packet['time_s']
        if (self.retention is None or not self.retention['available']
                or not 0<=t-self.retention['time_s']<=.150001):raise ValueError('stop_visual_retention_unavailable')
        if self.plane is None:raise ValueError('stop_top_unavailable')
        if len(proposed)!=50 or not np.isfinite(proposed).all() or any(abs(v)>1e-12 for v in proposed[43:46]):
            raise ValueError('loaded_stop_requires_zero_navigation')
        q=np.asarray(measured_joint_positions(packet,hands,self.model_names));requested=q.copy()
        for name in self.path.arm_names:
            index=self.path.model.joints[self.path.model.getJointId(name)].idx_q
            requested[index]=proposed[self.names.index(name)]
        path=self.path.screen(q,requested,self.plane,t,rotation)
        motion=predict_vertical(t,self.plane.time,self.heights,self.acc)
        lower=path['lower_m']-motion['downward_allowance_m']
        return dict(status='clear' if lower>=.03 else 'insufficient_margin',lower_m=lower,
            required_margin_m=.03,geometric_lower_m=path['lower_m'],limiting_link=path['limiting_link'],
            retention=self.retention,path=path,vertical_motion=motion)
