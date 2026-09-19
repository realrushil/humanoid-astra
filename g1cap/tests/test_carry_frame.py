import unittest
try:
 import numpy as np
except ImportError:
 np=None
if np is not None:
 from g1cap.carry_frame import FloorCarryFrame

@unittest.skipIf(np is None,'NumPy runtime required')
class FrameTests(unittest.TestCase):
 def test_height_changes_but_xy_is_explicitly_not_a_position_measurement(self):
  f=FloorCarryFrame();floor=dict(status='observed_candidate',normal_body=[0,0,1],offset_m=.75)
  a=f.update(0,np.eye(3),floor);floor['offset_m']=.77;b=f.update(.1,np.eye(3),floor)
  np.testing.assert_allclose(np.array(b['body_in_control_frame'])[:3,3],[0,0,.02])
  self.assertFalse(b['horizontal_position_observed']);self.assertEqual(a['segment'],b['segment'])
 def test_observed_tilt_corrects_drift_without_claiming_measured_heading(self):
  f=FloorCarryFrame();floor=dict(status='observed_candidate',normal_body=[0,0,1],offset_m=.75)
  f.update(0,np.eye(3),floor);a=.08;floor['normal_body']=[np.sin(a),0,np.cos(a)]
  r=f.update(.1,np.eye(3),floor);rotation=np.array(r['body_in_control_frame'])[:3,:3]
  np.testing.assert_allclose(rotation@floor['normal_body'],[0,0,1],atol=1e-12)
 def test_missing_floor_and_gap_create_new_frame_not_false_continuity(self):
  f=FloorCarryFrame();floor=dict(status='observed_candidate',normal_body=[0,0,1],offset_m=.75)
  a=f.update(0,np.eye(3),floor)
  self.assertEqual(f.update(.1,np.eye(3),dict(status='unavailable'))['status'],'unavailable')
  b=f.update(.2,np.eye(3),floor);self.assertNotEqual(a['segment'],b['segment'])
  c=f.update(.4,np.eye(3),floor);self.assertNotEqual(b['segment'],c['segment'])

@unittest.skipIf(np is None,'NumPy runtime required')
class PropagatedFrameTests(unittest.TestCase):
 def setUp(self):
  self.f=FloorCarryFrame();self.floor=dict(status='observed_candidate',normal_body=[0,0,1],offset_m=.75)
  self.observed=self.f.update(0,np.eye(3),self.floor)
 def test_control_rotation_propagates_without_refreshing_camera_height(self):
  a=.03;r=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1.]])
  result=self.f.at_imu(.02,r)
  np.testing.assert_allclose(np.array(result['body_in_control_frame'])[:3,:3],r,atol=1e-12)
  self.assertEqual(result['time_s'],.02);self.assertEqual(result['observed_at_s'],0)
  self.assertEqual(result['height_age_s'],.02)
  self.assertEqual(self.observed['time_s'],0)
  self.assertEqual(np.array(result['body_in_control_frame'])[:3,3].tolist(),[0.,0.,0.])
  result['body_in_control_frame'][2][3]=999
  self.assertEqual(self.f.at_imu(.04,r)['body_in_control_frame'][2][3],0.)
 def test_new_depth_corrects_held_height_without_changing_source_timestamp(self):
  floor=dict(self.floor,offset_m=.762)
  observed=self.f.update(.1,np.eye(3),floor)
  self.assertEqual(self.f.at_imu(.1,np.eye(3))['body_in_control_frame'],observed['body_in_control_frame'])
  result=self.f.at_imu(.12,np.eye(3));self.assertAlmostEqual(result['body_in_control_frame'][2][3],.012)
  self.assertEqual(result['observed_at_s'],.1)
 def test_expired_future_and_missing_floor_cannot_supply_a_control_frame(self):
  for t in (-.01,.16,float('nan'),True):
   with self.assertRaises(ValueError):self.f.at_imu(t,np.eye(3))
  self.f.update(.1,np.eye(3),dict(status='unavailable'))
  with self.assertRaises(ValueError):self.f.at_imu(.1,np.eye(3))
  recovered=self.f.update(.2,np.eye(3),self.floor)
  self.assertNotEqual(self.observed['segment'],recovered['segment'])
 def test_rotation_must_be_a_finite_proper_transform(self):
  for rotation in (np.zeros((3,3)),np.full((3,3),np.nan),np.eye(4)):
   with self.assertRaises(ValueError):self.f.at_imu(.02,rotation)
 def test_perception_requires_current_imu_and_retains_camera_record(self):
  from g1cap.arena_perception import ArenaBoxPerception
  p=ArenaBoxPerception(None);p.carry_frame=self.f;p.latest_carry_frame=self.observed
  p.advance_imu(0,[0,0,0]);p.advance_imu(.02,[0,.1,0])
  result=p.control_frame(.02);self.assertEqual(result['observed_at_s'],0)
  self.assertEqual(p.latest_carry_frame,self.observed)
  with self.assertRaises(ValueError):p.control_frame(.04)

if __name__=='__main__':unittest.main()
