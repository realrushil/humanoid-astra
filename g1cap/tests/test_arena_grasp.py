import unittest
from test_arena_retreat import state
from g1cap.arena_grasp import GripPreparation
ACTION=[0.]*46+[.75,0.,0.,0.]
def sample(i,peak=15.):
 r=state(i);r['loaded_contacts']['maximum_hand_N']=peak
 return r
class Motion:
 def __init__(self):self.calls=0
 def command(self,*args):self.calls+=1;return args[-2]
class PreparationTests(unittest.TestCase):
 def make(self,peak=15.):
  motion=Motion();c=GripPreparation(sample(50,peak),ACTION,lambda *a,**kw:motion)
  return c,motion
 def test_force_margin_freezes_references_and_never_restarts(self):
  c,m=self.make();c.command(sample(50));self.assertEqual(m.calls,1)
  c.update(sample(51,20.));before=list(c.last_action)
  for i in range(52,60):c.command(sample(i,16.));c.update(sample(i,16.))
  self.assertEqual(m.calls,1);self.assertEqual(c.last_action,before)
 def test_already_high_contact_adds_no_tightening(self):
  c,m=self.make(21.);c.command(sample(50,21.));self.assertEqual(m.calls,0)
 def test_force_ceiling_failure_is_latched(self):
  c,m=self.make();self.assertEqual(c.update(sample(51,26.)),('failed','preparation_force_limit'))
  self.assertEqual(c.update(sample(52)),('failed','preparation_force_limit'))
 def test_three_second_dwell_and_stable_hold_are_required(self):
  c,m=self.make();c.update(sample(51,20.))
  for i in range(52,200):self.assertIsNone(c.update(sample(i,19.)))
  self.assertEqual(c.update(sample(200,19.)),('completed','grasp_preparation_verified'))
if __name__=='__main__':unittest.main()
