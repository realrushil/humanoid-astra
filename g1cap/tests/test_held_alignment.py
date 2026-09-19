import unittest
from types import SimpleNamespace
from collections import deque
try:
 import numpy as np
 from scipy.spatial.transform import Rotation
except ImportError:
 raise unittest.SkipTest("native numerical dependencies unavailable")
from g1cap.arena_alignment import HeldAlignment
from g1cap.toolkit.box_alignment import AlignmentPath

class TransitionTests(unittest.TestCase):
 def make(self):
  c=object.__new__(HeldAlignment)
  c.box_rotation=np.eye(3);c.path=AlignmentPath([0,0,0],np.eye(3));c.started=0.
  c.phase='place_align';c.method='temporary_align';c.result=None;c.history=deque(maxlen=51);c.measurements=[]
  c.last_action=[.1]*43+[0.,0.,0.,.75,0.,0.,0.]
  def forbidden(*args):raise AssertionError('IK called after transition to verification')
  c.wrists=SimpleNamespace(targets={'left':SimpleNamespace(translation=np.zeros(3))},command=forbidden)
  return c
 def row(self,i):
  return dict(time=i*.02,step=i,box_quat=[1,0,0,0],box_pos=[0,0,0],root_pos=[0,0,0],
   root_quat=[1,0,0,0],joint_pos=[0]*43,wrists_world={'left':{'pos':[0,0,0]}},
   surfaces={'source':{'clearance_m':.08,'robot_contact_peak_N':0}},stance_clear=True,
   tilt=0.,bilateral=True,clearance=.08,loaded_contacts={'samples':4,'minimum_hand_N':10.,'maximum_hand_N':20.,'minimum_total_foot_N':100.})
 def test_measured_alignment_enters_hold_and_never_runs_ik_there(self):
  c=self.make();self.assertIsNone(c.update(self.row(1)))
  self.assertEqual(c.phase,'place_align_hold')
  saved=list(c.last_action)
  self.assertEqual(c.command(self.row(2)),saved)
  for i in range(2,151):self.assertIsNone(c.update(self.row(i)))
  c.update(self.row(151))
  self.assertEqual(c.result,('completed','aligned_hold'))
 def test_force_limit_remains_active_during_verification(self):
  c=self.make();c.update(self.row(1));r=self.row(2);r['loaded_contacts']['maximum_hand_N']=25.1
  self.assertEqual(c.update(r),('failed','preparation_force_limit'))
  self.assertEqual(c.update(self.row(3)),('failed','preparation_force_limit'))
if __name__=='__main__':unittest.main()
