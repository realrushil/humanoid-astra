"""Sensor-frame transforms must use measured ancestor joints, not world state."""
import importlib
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import pinocchio as pin
except ImportError:
    pin = None

URDF = '''<robot name="camera_test">
<link name="pelvis"/><link name="torso"/><link name="head"/><link name="leg"/>
<link name="left_wrist_yaw_link"/><link name="right_wrist_yaw_link"/>
<joint name="waist" type="revolute"><parent link="pelvis"/><child link="torso"/>
<origin xyz="0 0 0.5"/><axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
<joint name="head_mount" type="fixed"><parent link="torso"/><child link="head"/><origin xyz="0.1 0 0"/></joint>
<joint name="hip" type="revolute"><parent link="pelvis"/><child link="leg"/>
<axis xyz="0 1 0"/><limit lower="-3" upper="3" effort="1" velocity="1"/></joint>
<joint name="left_mount" type="fixed"><parent link="pelvis"/><child link="left_wrist_yaw_link"/><origin xyz="0 0.2 0.3"/></joint>
<joint name="right_mount" type="fixed"><parent link="pelvis"/><child link="right_wrist_yaw_link"/><origin xyz="0 -0.2 0.3"/></joint></robot>'''


@unittest.skipIf(pin is None,'native Pinocchio is required')
class SensorKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        path=Path(self.tmp.name)/'robot.urdf';path.write_text(URDF)
        self.model=pin.buildModelFromUrdf(str(path))
        try:
            self.module=importlib.import_module('g1cap.sensor_kinematics')
        except ModuleNotFoundError:
            self.fail('Sensor-only body-frame kinematics is missing')
        self.calibration=dict(parent_frame='head',offset_position_m=[.2,0,0],
                              offset_quaternion_xyzw=[0,0,0,1],offset_convention='ros')

    def test_named_ancestor_feedback_moves_camera_and_unrelated_joint_is_unneeded(self):
        pose=self.module.camera_in_body(self.model,{'waist':np.pi/2},self.calibration)
        np.testing.assert_allclose(pose[:3,3],[0,.3,.5],atol=1e-12)
        np.testing.assert_allclose(pose[:3,:3]@[1,0,0],[0,1,0],atol=1e-12)
        again=self.module.camera_in_body(self.model,{'waist':np.pi/2,'hip':.8},self.calibration)
        np.testing.assert_array_equal(pose,again)

    def test_missing_ancestor_is_not_filled_with_neutral_pose(self):
        with self.assertRaisesRegex(ValueError,'waist'):
            self.module.camera_in_body(self.model,{'hip':.1},self.calibration)
        with self.assertRaises(ValueError):
            self.module.camera_in_body(self.model,{'waist':float('nan')},self.calibration)

    def test_camera_mount_quaternion_is_xyzw_and_not_silently_reinterpreted(self):
        self.calibration['offset_quaternion_xyzw']=[0,0,2**-.5,2**-.5]
        pose=self.module.camera_in_body(self.model,{'waist':0.},self.calibration)
        np.testing.assert_allclose(pose[:3,:3]@[1,0,0],[0,1,0],atol=1e-12)
        self.calibration['offset_convention']='opengl'
        with self.assertRaises(ValueError):
            self.module.camera_in_body(self.model,{'waist':0.},self.calibration)

    def test_rejects_unknown_frames_and_floating_base_dependency(self):
        with self.assertRaises(ValueError):
            self.module.frame_in_body(self.model,'missing',{'waist':0.})
        path=Path(self.tmp.name)/'robot.urdf'
        floating=pin.buildModelFromUrdf(str(path),pin.JointModelFreeFlyer())
        with self.assertRaises(ValueError):
            self.module.camera_in_body(floating,{'waist':0.},self.calibration)

    def measured_box_inputs(self):
        from g1cap.arena_sensors import proprioception_packet
        p=proprioception_packet(step=50,time_s=1.,joint_names=['waist'],q=[float(np.pi/2)],
            dq=[0.],tau_est=[0.],gyro=[0.,0.,0.],accel=[0.,0.,9.81])
        e=dict(status='accepted',observed_at_s=1.,age_s=0.,track_epoch=1,
               center_camera_m=[.1,0,.4],axes_camera=np.eye(3).tolist(),dimensions_m=[.2,.2,.2])
        return e,p

    def test_box_and_both_wrists_use_only_same_time_encoder_kinematics(self):
        e,p=self.measured_box_inputs()
        state=self.module.box_in_robot_frames(self.model,e,p,self.calibration)
        self.assertEqual(state['status'],'accepted')
        np.testing.assert_allclose(state['box_center_pelvis_m'],[0,.4,.9],atol=1e-12)
        np.testing.assert_allclose(state['box_center_wrist_m']['left'],[0,.2,.6],atol=1e-12)
        np.testing.assert_allclose(state['box_center_wrist_m']['right'],[0,.6,.6],atol=1e-12)
        self.assertNotIn('contact',state);self.assertNotIn('mass',state)

    def test_relative_geometry_rejects_mixed_times_and_extra_truth_packet(self):
        e,p=self.measured_box_inputs();p['time_s']=1.02
        state=self.module.box_in_robot_frames(self.model,e,p,self.calibration)
        self.assertEqual(state['reason'],'unsynchronized_encoders')
        self.assertIsNone(state['box_center_pelvis_m'])
        e,p=self.measured_box_inputs();p['box_mass']=.1
        with self.assertRaises(ValueError):self.module.box_in_robot_frames(self.model,e,p,self.calibration)
        e,p=self.measured_box_inputs();e['status']='unavailable'
        state=self.module.box_in_robot_frames(self.model,e,p,self.calibration)
        self.assertIsNone(state['box_center_pelvis_m'])
