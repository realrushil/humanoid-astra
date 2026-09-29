"""Guard against subtracting motion allowances after choosing the free side."""
import unittest
try:
 import numpy as np
 from g1cap.source_stop import SourceSides
 from g1cap.loaded_stop import TopPlane
 from g1cap.remembered_source_plane import RememberedSourcePlane
except ImportError:
 np=None

@unittest.skipIf(np is None,'native numerical runtime required')
class DirectionalClearanceTests(unittest.TestCase):
 def plane(self,top_error=0.,front_angle=0.):
  motion=dict(status='tracked_local_segment',time_s=1.,segment=1,body_in_segment=np.eye(4).tolist())
  memory=RememberedSourcePlane(1.,motion,[1,0,0],0,[0,0,0],plane_error_m=0.,plane_angle_rad=front_angle,max_age_s=10.)
  top=TopPlane([0,0,1],0,[0,0,0],0.,top_error,1.,np.eye(3))
  return SourceSides(top,memory,motion,np.eye(3),1.)
 def evaluate(self,p,points,front,top,norm):
  return p.bounds(points,1.,np.eye(3),front_closing_m=front,top_closing_m=top,point_displacement_m=norm)
 def test_top_allows_tangent_motion_but_subtracts_downward_motion(self):
  # A 10cm planar deviation should not consume 10cm of vertical reserve.
  result=self.evaluate(self.plane(),[[-1,0,.08]],.10,.02,.11)
  self.assertAlmostEqual(result[0],.06)
 def test_side_choice_happens_after_unequal_allowances(self):
  # Front starts better; a large front-closing allowance makes top better.
  result=self.evaluate(self.plane(),[[.3,0,.08]],.20,.01,.21)
  self.assertAlmostEqual(result[0],.07)
 def test_uncertain_normal_penalizes_even_nominally_tangent_motion(self):
  result=self.evaluate(self.plane(top_error=.1),[[-1,0,.08]],.10,.02,.11)
  self.assertAlmostEqual(result[0],.08-.1*(1+.08**2)**.5-.02-.011)
 def test_corners_cannot_independently_choose_free_sides(self):
  result=self.evaluate(self.plane(),[[1,0,-1],[-1,0,1]],0,0,0)
  self.assertLess(min(result),0)
 def test_front_normal_uncertainty_uses_actual_memory_angle(self):
  p=self.plane(front_angle=.2);p.top=None
  original=p.bounds([[1,0,0]],1.,np.eye(3))[0]
  result=self.evaluate(p,[[1,0,0]],.02,.01,.10)
  self.assertAlmostEqual(original-result[0],.02+.10*(2*np.sin(np.deg2rad(2.)/2)+2*np.sin(.1)))
 def test_lost_front_uses_only_top_with_top_allowance(self):
  p=self.plane();p.memory.invalid=True
  self.assertAlmostEqual(self.evaluate(p,[[1,0,.08]],0,.03,.05)[0],.05)
 def test_invalid_allowances_reject_before_any_side_can_pass(self):
  for values in ((-.01,0,0),(0,float('nan'),0),(0,0,float('inf'))):
   with self.subTest(values=values),self.assertRaises(ValueError):self.evaluate(self.plane(),[[1,0,1]],*values)
