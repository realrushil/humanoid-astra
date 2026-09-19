import unittest
try:
 import numpy as np
except ImportError:
 np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class SceneMotionTests(unittest.TestCase):
 def test_scene_loss_withholds_readiness_without_discarding_observed_grasp_geometry(self):
  from g1cap.scene_motion import combine_grasp_motion
  grasp=dict(status='available',time_s=1.,raised=True,ready=True,attitude_ok=True,gap_m=.08)
  result=combine_grasp_motion(grasp,dict(status='unavailable',ready=False))
  self.assertEqual(result['status'],'available');self.assertTrue(result['raised'])
  self.assertFalse(result['ready']);self.assertEqual(result['scene_status'],'unavailable')
  self.assertTrue(result['wrist_relative_ready']);self.assertTrue(grasp['ready'])
  ready=combine_grasp_motion(grasp,dict(status='available',ready=True))
  self.assertTrue(ready['ready'])
  unknown=combine_grasp_motion(dict(status='unavailable',reason='missing_box'),dict(status='available',ready=True))
  self.assertEqual(unknown['status'],'unavailable');self.assertFalse(unknown['ready'])
 def feet(self):
  feet=np.repeat(np.eye(4)[None],2,axis=0);feet[:,2,3]=.035;feet[:,1,3]=[.1,-.1]
  return feet
 def floor(self):return dict(status='observed_candidate',normal_body=[0,0,1],offset_m=0.)
 def update(self,observer,t,feet,floor=None):
  observer.advance_imu(t,[0,0,0])
  return observer.update(t,feet,self.floor() if floor is None else floor)
 def test_gyro_rotation_and_clock_are_explicit(self):
  from g1cap.scene_motion import StanceMotion
  m=StanceMotion();m.advance_imu(0,[0,.5,0])
  for i in range(1,11):m.advance_imu(i*.1,[0,.5,0])
  np.testing.assert_allclose(m.rotation,[[np.cos(.5),0,np.sin(.5)],[0,1,0],[-np.sin(.5),0,np.cos(.5)]],atol=1e-12)
  with self.assertRaises(ValueError):m.advance_imu(1.,[0,.5,0])
  with self.assertRaises(ValueError):m.advance_imu(1.3,[0,.5,0])
 def test_floor_selection_uses_feet_and_observed_gravity(self):
  from g1cap.scene_motion import observed_floor
  planes=[dict(normal=np.array([0.,0,1]),offset_m=0.),dict(normal=np.array([0.,0,1]),offset_m=-.8)]
  f=observed_floor(planes,np.eye(4),self.feet(),[0,0,1])
  self.assertEqual(f['status'],'observed_candidate');self.assertAlmostEqual(f['offset_m'],0.)
  self.assertEqual(observed_floor([],np.eye(4),self.feet(),[0,0,1])['status'],'unavailable')
 def test_stance_switch_and_rolling_toe_do_not_anchor_the_ankle(self):
  from g1cap.scene_motion import StanceMotion
  m=StanceMotion();feet=self.feet()
  self.assertEqual(self.update(m,0,feet)['status'],'unavailable')
  self.assertEqual(self.update(m,.1,feet)['status'],'initialized_local_segment')
  moved=feet.copy();moved[:,0,3]-=.01;moved[1,2,3]+=.1
  r=self.update(m,.2,moved);self.assertAlmostEqual(r['body_in_segment'][0][3],.01)
  a=np.radians(12);rotation=np.array([[np.cos(a),0,np.sin(a)],[0,1,0],[-np.sin(a),0,np.cos(a)]])
  pivot=np.array([.12,0,-.03]);rolled=moved.copy();rolled[0,:3,:3]=rotation;rolled[0,:3,3]+=pivot-rotation@pivot
  r=self.update(m,.3,rolled)
  np.testing.assert_allclose(np.array(r['body_in_segment'])[:3,3],[.01,0,0],atol=1e-12)
  self.assertTrue(all(i in [2,3] for i in r['used_points']))
 def test_lost_support_never_bridges_a_pose_or_segment(self):
  from g1cap.scene_motion import StanceMotion
  m=StanceMotion();feet=self.feet()
  self.update(m,0,feet);self.update(m,.1,feet)
  missing=self.update(m,.2,feet,dict(status='unavailable'))
  self.assertIsNone(missing['body_in_segment'])
  self.assertEqual(self.update(m,.3,feet)['status'],'unavailable')
  r=self.update(m,.4,feet);self.assertEqual(r['segment'],2)
  self.assertEqual(r['status'],'initialized_local_segment')
 def test_disagreeing_point_motion_is_not_a_pose(self):
  from g1cap.scene_motion import StanceMotion
  m=StanceMotion();feet=self.feet();self.update(m,0,feet);self.update(m,.1,feet)
  feet[1,0,3]+=.2;r=self.update(m,.2,feet)
  self.assertEqual(r['status'],'unavailable');self.assertIsNone(r['body_in_segment'])
 def stability_sample(self,window,t,x=0,segment=1,status='tracked_local_segment'):
  body=np.eye(4);body[0,3]=x
  motion=dict(status=status,segment=segment,body_in_segment=body.tolist(),time_s=t)
  box=dict(status='accepted',observed_at_s=t,center_camera_m=[0,0,1],axes_camera=np.eye(3).tolist())
  return window.update(t,motion,box,np.eye(4),[0,0,1],grasp_ready=True)
 def test_rigidly_held_box_moving_through_scene_is_not_stable(self):
  from g1cap.scene_motion import SceneStability
  moving=SceneStability();quiet=SceneStability()
  for i in range(11):
   t=i*.1;r=self.stability_sample(moving,t,.1*t);q=self.stability_sample(quiet,t)
  self.assertFalse(r['ready']);self.assertAlmostEqual(r['max_box_speed_m_s'],.1)
  self.assertTrue(q['ready'])
 def test_scene_stability_expires_on_loss_or_new_segment(self):
  from g1cap.scene_motion import SceneStability
  m=SceneStability()
  for i in range(11):r=self.stability_sample(m,i*.1)
  self.assertTrue(r['ready'])
  self.assertFalse(self.stability_sample(m,1.1,status='unavailable')['ready'])
  self.assertFalse(self.stability_sample(m,1.2,segment=2)['ready'])
 def test_scene_readiness_needs_elapsed_second_at_each_rate(self):
  from g1cap.scene_motion import SceneStability
  for hz in (10,25,50):
   m=SceneStability()
   for i in range(hz):r=self.stability_sample(m,i/hz)
   self.assertFalse(r['ready'],hz)
   self.assertTrue(self.stability_sample(m,1.)['ready'],hz)
  m=SceneStability()
  for t in (0,.1,.2,.24,.28,.32,.36,.4,.44,.48,.52,.56,.6,.64,.68,.72,.76,.8,.84,.88,.92,.96,1.):
   r=self.stability_sample(m,t)
  self.assertTrue(r['ready'])
 def test_fast_scene_drift_and_epoch_change_reject(self):
  from g1cap.scene_motion import SceneStability
  m=SceneStability()
  for i in range(51):r=self.stability_sample(m,i*.02,x=.1*i*.02)
  self.assertFalse(r['ready'])
  m=SceneStability()
  for i in range(51):r=self.stability_sample(m,i*.02)
  self.assertTrue(r['ready'])
  self.assertFalse(self.stability_sample(m,1.02,segment=2)['ready'])
  self.assertFalse(self.stability_sample(m,1.3,segment=2)['ready'])
