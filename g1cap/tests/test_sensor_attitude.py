"""Attitude from permitted measurements must reject missing/stale feedback."""
import importlib,unittest
try:
 import numpy as np
except ImportError:
 np=None

@unittest.skipIf(np is None,'numpy required')
class SensorAttitudeTests(unittest.TestCase):
 def module(self):
  try:return importlib.import_module('g1cap.sensor_attitude')
  except ModuleNotFoundError:self.fail('Sensor attitude estimator is missing')

 def test_gyro_rotates_gravity_in_body_frame_without_pose_input(self):
  m=self.module();s=m.GravityEstimate([0,0,1],0.,[0,.5,0])
  for i in range(1,51):s.update(i*.02,[0,.5,0],[0,0,0])
  np.testing.assert_allclose(s.up,[-np.sin(.5),0,np.cos(.5)],atol=1e-12)
  q=m.up_quaternion(s.up)
  w,x,y,z=q
  np.testing.assert_allclose([2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)],s.up,atol=1e-12)

 def test_large_specific_force_does_not_become_tilt(self):
  m=self.module();s=m.GravityEstimate([0,0,1],0.,[0,0,0])
  for i in range(1,51):s.update(i*.02,[0,0,0],[15,0,9.81])
  np.testing.assert_allclose(s.up,[0,0,1])
  self.assertFalse(s.accel_used)

 def test_slow_accelerometer_correction_and_stale_rejection(self):
  m=self.module();s=m.GravityEstimate([.1,0,1],0.,[0,0,0]);before=abs(s.up[0])
  for i in range(1,51):s.update(i*.02,[0,0,0],[0,0,9.81])
  self.assertLess(abs(s.up[0]),before);self.assertTrue(s.accel_used)
  for t in [1.,.5,2.]:
   with self.assertRaises(ValueError):s.update(t,[0,0,0],[0,0,9.81])
  with self.assertRaises(ValueError):m.GravityEstimate([0,0,0],0.,[0,0,0])

 def test_bootstrap_uses_observed_level_plane_and_camera_calibration(self):
  m=self.module();k=[[80,0,32],[0,80,32],[0,0,1]]
  a=.35;r=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])@np.diag([1,-1,-1])
  v,u=np.indices((64,64));rays=np.stack(((u-32)/80,(v-32)/80,np.ones_like(u)),axis=-1)
  n=r.T@np.array([0,0,1]);depth=-1/(rays@n)
  t=np.eye(4);t[:3,:3]=r;t[:3,3]=[0,0,1]
  np.testing.assert_allclose(m.initial_up_from_depth(depth,k,t),[0,0,1],atol=1e-8)
  with self.assertRaises(ValueError):m.initial_up_from_depth(depth*np.nan,k,t)

 def test_homie_adapter_reorders_body_feedback_and_rejects_oracle_fields(self):
  m=self.module()
  from g1cap.arena_sensors import proprioception_packet
  p=proprioception_packet(step=1,time_s=.02,joint_names=['arm','hip'],q=[.2,.4],dq=[.1,.3],tau_est=[1,2],gyro=[.01,.02,.03],accel=[0,0,9.81])
  obs=m.homie_observation(p,['hip','finger','arm'],['arm','hip'],[0,0,1])
  self.assertEqual(set(obs),{'q','dq','floating_base_pose','floating_base_vel'})
  np.testing.assert_array_equal(obs['q'],[[.4,0,.2]])
  np.testing.assert_array_equal(obs['floating_base_pose'],[[0,0,0,1,0,0,0]])
  np.testing.assert_array_equal(obs['floating_base_vel'],[[0,0,0,.01,.02,.03]])
  with self.assertRaises(ValueError):m.homie_observation(dict(p,root_pose=[1,2,3]),['hip','finger','arm'],['arm','hip'],[0,0,1])
