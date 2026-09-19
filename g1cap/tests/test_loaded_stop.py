import unittest
try:
 import numpy as np
 import pinocchio
except ImportError:
 np=None
if np is not None:
 from g1cap.loaded_stop import TopPlane,ArmPath

@unittest.skipIf(np is None,"native numerical runtime required")
class PlaneTests(unittest.TestCase):
 def test_no_front_is_required_but_top_age_is_explicit(self):
  plane=TopPlane([0,0,1],-.7,[0,0,.7],.003,.01,1.,np.eye(3))
  result=plane.bounds([[0,0,.9]],1.08,np.eye(3))
  self.assertAlmostEqual(result[0],.2-.003-.002)
  with self.assertRaises(ValueError):plane.bounds([[0,0,.9]],1.16,np.eye(3))
 def test_body_rotation_is_propagated_without_inventing_translation(self):
  plane=TopPlane([0,0,1],-.7,[0,0,.7],.003,0,1.,np.eye(3))
  rotation=np.array([[1,0,0],[0,0,-1],[0,1,0]])
  a=plane.bounds([[0,0,.9]],1.0,np.eye(3))
  b=plane.bounds(np.array([[0,0,.9]])@rotation,1.0,rotation)
  np.testing.assert_allclose(a,b)
 def test_nonfinite_rotation_or_points_rejected(self):
  with self.assertRaises(ValueError):TopPlane([0,0,1],0,[0,0,0],.003,0,0.,np.full((3,3),np.nan))
  plane=TopPlane([0,0,1],0,[0,0,0],.003,0,0.,np.eye(3))
  with self.assertRaises(ValueError):plane.bounds([[0,0,float("nan")]],0.,np.eye(3))
  with self.assertRaises(ValueError):plane.bounds([[0,0,1]],0.,2*np.eye(3))
 def test_lower_hand_is_rejected_geometrically(self):
  plane=TopPlane([0,0,1],-.7,[0,0,.7],.003,0,0.,np.eye(3))
  self.assertLess(plane.bounds([[0,0,.69]],0.,np.eye(3))[0],0)
 def test_invalid_and_future_measurements_rejected(self):
  with self.assertRaises(ValueError):TopPlane([0,0,2],0,[0,0,0],.003,0,0.,np.eye(3))
  plane=TopPlane([0,0,1],0,[0,0,0],.003,0,1.,np.eye(3))
  with self.assertRaises(ValueError):plane.bounds([[0,0,1]],.9,np.eye(3))

@unittest.skipIf(np is None,"native numerical runtime required")
class PathTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.path=ArmPath('g1cap/assets/arena_g1_rev1_0_kinematics.urdf','g1cap/assets/arena_g1_rev1_0_bounds.json')
 def test_reach_bounds_cover_dense_unsampled_joint_paths(self):
  rng=np.random.default_rng(104);path=self.path
  for _ in range(12):
   q=np.zeros(path.model.nq);end=q.copy();end[path.arm_ids]=rng.uniform(-.8,.8,len(path.arm_ids))
   plane=TopPlane([0,0,1],0,[0,0,0],.003,.01,0.,np.eye(3))
   lower=path.screen(q,end,plane,0.,np.eye(3),samples=3)['lower_m']
   actual=min(float(np.min(plane.bounds(p,0.,np.eye(3)))) for f in np.linspace(0,1,81) for _,p in path.points(q+f*(end-q)))
   self.assertLessEqual(lower,actual+1e-12)
 def test_nonfinite_path_rejected(self):
  path=self.path;q=np.zeros(path.model.nq);q[path.arm_ids[0]]=np.nan
  plane=TopPlane([0,0,1],0,[0,0,0],.003,0,0.,np.eye(3))
  with self.assertRaises(ValueError):path.screen(q,q,plane,0.,np.eye(3))
 def test_measured_and_requested_joint_states_both_checked(self):
  path=self.path;q=np.zeros(path.model.nq);end=q.copy();end[path.arm_ids]=.5
  plane=TopPlane([0,0,1],0,[0,0,0],.003,0,0.,np.eye(3))
  result=path.screen(q,end,plane,0.,np.eye(3))
  for config in (q,end):
   endpoint=min(float(min(plane.bounds(p,0.,np.eye(3)))) for _,p in path.points(config))
   self.assertLessEqual(result['lower_m'],endpoint)
  self.assertGreater(result['interpolation_allowance_m'],0)
if __name__=='__main__':unittest.main()

@unittest.skipIf(np is None,"native numerical runtime required")
class LoadedStopEvidenceTests(unittest.TestCase):
 def setUp(self):
  from g1cap.loaded_stop import LoadedStopClearance
  self.stop=LoadedStopClearance('g1cap/assets/arena_g1_rev1_0_kinematics.urdf','g1cap/assets/arena_g1_rev1_0_bounds.json',[])
 def test_future_and_stale_images_cannot_enter_height_history(self):
  for camera_t in (1.1,.5):
   with self.assertRaises(ValueError):self.stop.measure({'time_s':1.},{'time_s':camera_t},np.eye(3),[0,0,1],None)
  self.assertFalse(self.stop.heights)
 def test_missing_source_clears_previous_plane_even_between_images(self):
  self.stop.camera_time=.9;self.stop.plane=object()
  self.stop.measure({'time_s':1.},{'time_s':.9,'camera_transform':np.eye(4),'box':{'status':'unavailable'}},np.eye(3),[0,0,1],None)
  self.assertIsNone(self.stop.plane)
 def test_stop_path_cannot_authorize_navigation(self):
  self.stop.retention=dict(time_s=1.,available=True);self.stop.plane=object()
  action=[0.]*50;action[43]=.1
  with self.assertRaisesRegex(ValueError,'requires_zero_navigation'):
   self.stop.screen({'time_s':1.},None,np.eye(3),action)
