"""Depth exclusion from calibrated robot geometry and body encoders.

Unknown finger chains use conservative rotation-independent reach spheres. No
finger position, world pose, table geometry or simulator label is supplied.
"""
import itertools,json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
from .sensor_kinematics import frame_in_body,camera_in_body


class RobotGeometry:
    def __init__(self,urdf,bounds_file,joint_names):
        self.model=pin.buildModelFromUrdf(str(urdf))
        self.bounds=json.loads(Path(bounds_file).read_text())
        joints=ET.parse(urdf).getroot().findall('joint');children={}
        for j in joints:children.setdefault(j.find('parent').attrib['link'],[]).append(j)
        def translation(j):return np.fromstring(j.find('origin').attrib.get('xyz','0 0 0'),sep=' ')
        def radius(link):
            # Triangle inequality: rotations preserve point/offset norm, so
            # this encloses every configuration, not just sampled joint angles.
            own=max((np.linalg.norm(v) for s in self.bounds.get(link,[])
                     for v in itertools.product(*zip(s['min'],s['max']))),default=0.)
            return max([own]+[np.linalg.norm(translation(j))+radius(j.find('child').attrib['link'])
                              for j in children.get(link,[])])
        probe=dict.fromkeys(joint_names,0.)
        self.rigid=[];self.spheres=[]
        for link in self.bounds:
            try:frame_in_body(self.model,link,probe)
            except ValueError as error:
                if 'missing ancestor joint measurements' not in str(error):raise
            else:self.rigid.append(link)
        for j in joints:
            if j.attrib['type']=='fixed' or j.attrib['name'] in joint_names:continue
            parent=j.find('parent').attrib['link'];child=j.find('child').attrib['link']
            try:frame_in_body(self.model,parent,probe)
            except ValueError as error:
                if 'missing ancestor joint measurements' not in str(error):raise
            else:self.spheres.append(dict(parent=parent,origin=translation(j),radius=radius(child),root_joint=j.attrib['name']))

    def exclude(self,points_camera,packet,calibration):
        joints=dict(zip(packet['joint_names'],packet['q_rad'],strict=True))
        camera=camera_in_body(self.model,joints,calibration)
        body=np.asarray(points_camera)@camera[:3,:3].T+camera[:3,3]
        excluded=np.zeros(len(body),bool)
        for link in self.rigid:
            pose=frame_in_body(self.model,link,joints)
            local=(body-pose[:3,3])@pose[:3,:3]
            for bounds in self.bounds[link]:
                # 5 mm is an uncalibrated model/sensing development margin.
                excluded|=((local>=np.array(bounds['min'])-.005)&(local<=np.array(bounds['max'])+.005)).all(axis=1)
        for sphere in self.spheres:
            pose=frame_in_body(self.model,sphere['parent'],joints)
            center=pose[:3,:3]@sphere['origin']+pose[:3,3]
            excluded|=np.linalg.norm(body-center,axis=1)<=sphere['radius']+.005
        return excluded

    def lower_body_points(self,packet):
        """Current pelvis-frame collision-bound corners from body encoders."""
        from .toolkit.arena_approach import GUARDED_BODIES
        joints=dict(zip(packet['joint_names'],packet['q_rad'],strict=True))
        points=[]
        for name in GUARDED_BODIES:
            pose=frame_in_body(self.model,name,joints)
            for bounds in self.bounds[name]:
                corners=np.array(list(itertools.product(*zip(bounds['min'],bounds['max']))))
                points.extend(corners@pose[:3,:3].T+pose[:3,3])
        return np.array(points)

    def front_clearance(self,line,packet,calibration):
        """Signed pelvis/leg clearance to the observed camera-time front plane.

        The visible front is conservatively extended sideways. This does not
        observe hidden table legs or certify future gait/arm collision clearance.
        """
        joints=dict(zip(packet['joint_names'],packet['q_rad'],strict=True))
        camera=camera_in_body(self.model,joints,calibration)
        normal=camera[:3,:3]@np.asarray(line['normal_camera'])
        offset=float(line['offset_m']-normal@camera[:3,3])
        clearance=-float(np.max(self.lower_body_points(packet)@normal+offset))
        return normal,offset,clearance
