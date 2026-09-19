"""Opt-in single-box RGB-D pickup/hold diagnostic frontend.

Assumes one brown cuboid, initially upright on a visible neutral planar support.
Color masks are an explicit prototype association rule, not semantic recognition.
No scene config, object truth, contact flags or runtime simulator poses enter.
An explicit carry transition tracks the observed source plane after the box
leaves its footprint. This is a height reference, not current load support.
"""
from copy import deepcopy
import numpy as np

from .rgbd_box_geometry import initialize_box, observed_support_gap
from .rgbd_box_tracking import box_image_points, refine_box_pose
from .visual_box_state import BoxEstimateStream
from .visual_grasp import VisualGraspWindow
from .rgbd_geometry import depth_points,fit_planes
from .scene_motion import StanceMotion,SceneStability,observed_floor,combine_grasp_motion
from .carry_frame import FloorCarryFrame
from .source_plane import SourcePlane


class ArenaBoxPerception:
    def __init__(self,model):
        self.model=model
        self.track=BoxEstimateStream()
        self.grasp=VisualGraspWindow()
        self.motion=StanceMotion()
        self.scene_stability=SceneStability()
        self.carry_frame=FloorCarryFrame()
        self.latest_carry_frame=None
        self.source_plane=None
        self.carry_seed=None
        self.robot_geometry=None
        self.last_step=-1
        self.latest_motion=None
        self.latest=dict(status='unavailable',reason='not_initialized')

    def begin_carry(self,now_s):
        """Associate the source from fresh measured geometry at tool admission.

        Repeated carry requests preserve the source identity. Acquisition may
        explicitly start a new association; a tracking loss never does so.
        """
        if self.source_plane is not None:return
        seed=self.carry_seed
        if seed is None or not 0<=now_s-seed['time_s']<=.150001:
            raise ValueError('carry_source_observation_unavailable')
        source=SourcePlane()
        result=source.observe(seed['time_s'],seed['frame'],seed['camera_transform'],
            seed['box'],seed['initial'],points=self._source_points(seed),floor_points=seed['floor_points'])
        if result['status']!='observed_candidate':
            raise ValueError('carry_source_observation_unavailable')
        self.source_plane=source

    def _source_points(self,seed):
        """Exclude known robot surfaces from this camera-time depth cloud."""
        if self.robot_geometry is None:
            from pathlib import Path
            from .sensor_robot_geometry import RobotGeometry
            assets=Path(__file__).resolve().parent/'assets'
            self.robot_geometry=RobotGeometry(assets/'arena_g1_rev1_0_kinematics.urdf',
                assets/'arena_g1_rev1_0_bounds.json',seed['packet']['joint_names'])
        points=seed['points']
        return points[~self.robot_geometry.exclude(points,seed['packet'],seed['calibration'])]

    def begin_acquisition(self):
        """Reset source association only, preserving physical/sensor history."""
        self.source_plane=None

    def advance_imu(self,time_s,gyro):
        """Every recorded 50 Hz packet, including steps between camera frames."""
        self.motion.advance_imu(time_s,gyro)

    def control_frame(self,now_s):
        """Internal current-IMU frame; published camera estimates remain intact."""
        if self.motion.imu_time is None or abs(now_s-self.motion.imu_time)>1e-8:
            raise ValueError('floor_control_imu_time_mismatch')
        return self.carry_frame.at_imu(now_s,self.motion.rotation)

    def feedback(self,now_s):
        box=self.track.observe(now_s)
        if box['status']!='accepted' or self.latest['status']!='available':
            return dict(status='unavailable',reason=(box['reason'] if box['status']!='accepted'
                else self.latest['reason']),raised=False,ready=False,attitude_ok=False)
        return dict(deepcopy(self.latest),age_s=box['age_s'],
                    gap_reference='source_height_plane' if self.source_plane is not None else 'observed_under_box_plane')

    def update(self,packet,camera,up_body):
        """Process each new camera frame with encoders from that same time.

        Caller supplies an IMU/depth-derived up estimate at the measurement time.
        Between frames feedback retains its original timestamp and expires.
        """
        from .sensor_kinematics import box_in_robot_frames,camera_in_body,frame_in_body
        if camera['step']==self.last_step:return None
        if camera['step']!=packet['step'] or abs(camera['time_s']-packet['time_s'])>1e-8:
            raise ValueError('visual update requires synchronized camera and encoders')
        self.last_step=camera['step'];t=camera['time_s'];k=camera['calibration']['intrinsic']
        rgb=np.asarray(camera['rgb'],float);depth=camera['depth_m']
        mask=(rgb[...,0]>1.2*rgb[...,1])&(rgb[...,1]>1.1*rgb[...,2])
        neutral=(rgb.max(axis=2)-rgb.min(axis=2)<25)&~mask
        joints=dict(zip(packet['joint_names'],packet['q_rad'],strict=True))
        transform=camera_in_body(self.model,joints,camera['calibration'])
        feet=[frame_in_body(self.model,s+'_ankle_roll_link',joints) for s in ('left','right')]
        # One current background fit supplies both support gap and floor/stance
        # hypotheses. Visible geometry is not a complete scene collision map.
        background=depth_points(depth,k)[neutral]
        planes=fit_planes(background,max_planes=4,min_points=200)
        floor=observed_floor(planes,transform,feet,up_body)
        motion=self.motion.update(t,feet,floor)
        self.latest_motion=motion
        self.latest_carry_frame=self.carry_frame.update(t,self.motion.rotation,floor)
        initial=fit=None
        if self.track.epoch==0:
            initial=initialize_box(mask,neutral,depth,k)
            if initial['status']=='cuboid_candidate':
                self.track.initialize(time_s=t,now_s=t,center=initial['center_camera_m'],
                    rotation=initial['axes_camera'],size=initial['dimensions_m'])
        else:
            seed=self.track.prediction_seed(t)
            if seed is not None:
                points,edges=box_image_points(mask,depth,k)
                center,rotation,fit=refine_box_pose(points,edges,seed['center_camera_m'],
                    seed['axes_camera'],seed['dimensions_m'])
                fit['total_face_points']=len(points)
                self.track.submit(time_s=t,now_s=t,center=center,rotation=rotation,
                    size=seed['dimensions_m'],quality=fit)
        box=self.track.observe(t);relative=support=None
        if box['status']=='accepted':
            relative=box_in_robot_frames(self.model,box,packet,camera['calibration'])
            support=observed_support_gap(neutral,depth,k,box,transform[:3,:3].T@up_body,planes=planes)
        floor_points=np.empty((0,3))
        if floor['status']=='observed_candidate':
            # Recover current inliers of the selected measured floor. Its
            # offset only selects depth samples; source height is fitted anew.
            normal_body=np.asarray(floor['normal_body'])
            normal_camera=transform[:3,:3].T@normal_body
            offset=floor['offset_m']+normal_body@transform[:3,3]
            floor_points=background[np.isfinite(background).all(axis=1)&
                                    (abs(background@normal_camera+offset)<=.003)]
        self.carry_seed=dict(time_s=t,frame=self.latest_carry_frame,camera_transform=transform,
            box=box,initial=support,points=background,floor_points=floor_points,
            packet=packet,calibration=camera['calibration'],gyro_rotation=self.motion.rotation.copy())
        height_reference=support
        if self.source_plane is not None:
            height_reference=self.source_plane.observe(t,self.latest_carry_frame,transform,
                box,support,points=self._source_points(self.carry_seed),floor_points=floor_points)
        if relative is not None and height_reference is not None and height_reference['status']=='observed_candidate':
            self.latest=self.grasp.add(dict(relative,time_s=t,gap_m=height_reference['gap_m'],
                dimensions_m=box['dimensions_m'],up_body=np.asarray(up_body).tolist()))
        else:
            self.grasp.samples.clear()
            self.latest=dict(status='unavailable',reason=box['reason'] if relative is None or height_reference is None
                             else height_reference['reason'])
        scene=self.scene_stability.update(t,motion,box,transform,up_body,
            grasp_ready=self.latest.get('ready',False))
        self.latest=combine_grasp_motion(self.latest,scene)
        return dict(step=camera['step'],time_s=t,up_body=np.asarray(up_body).tolist(),
            initialization=initial,fit=fit,box=box,relative=relative,support=support,
            height_reference=height_reference,carry_frame=self.latest_carry_frame,
            source_obstacle=self.source_plane.geometry if self.source_plane is not None else None,
            floor=floor,motion=motion,scene_stability=scene,
            feedback=self.feedback(t))
