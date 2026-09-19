import unittest
try:
 import numpy as np
except ImportError:
 raise unittest.SkipTest("native numerical dependencies unavailable")
from g1cap.toolkit.box_alignment import AlignmentPath

class AlignmentTests(unittest.TestCase):
 def test_common_rotation_preserves_both_grasp_transforms(self):
  a=np.radians(35);R=np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
  center=np.array([.1,.2,.4]);p=AlignmentPath(center,R)
  for fraction in (0.,.25,1.):
   D=p.delta(fraction)
   for wrist in (np.array([.2,.1,.5]),np.array([.2,.3,.5])):
    moved=center+D@(wrist-center)
    np.testing.assert_allclose((D@R).T@(moved-center),R.T@(wrist-center),atol=1e-12)
  self.assertAlmostEqual(abs((p.delta(1)@R)[2,2]),1.)
 def test_invalid_and_excessive_rotation_rejected(self):
  with self.assertRaises(ValueError):AlignmentPath([0,0,float('nan')],np.eye(3))
  with self.assertRaises(ValueError):AlignmentPath([0,0,0],np.zeros((3,3)))
  R=np.array([[1/np.sqrt(2),-1/np.sqrt(2),0],[1/np.sqrt(6),1/np.sqrt(6),-2/np.sqrt(6)],[1/np.sqrt(3)]*3])
  with self.assertRaisesRegex(ValueError,'45-degree'):AlignmentPath([0,0,0],R)
 def test_identity_and_fraction_bounds(self):
  p=AlignmentPath([0,0,0],np.eye(3));np.testing.assert_allclose(p.delta(.5),np.eye(3))
  with self.assertRaises(ValueError):p.delta(1.1)
if __name__=='__main__':unittest.main()
