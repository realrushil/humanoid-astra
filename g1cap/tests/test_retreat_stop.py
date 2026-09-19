import unittest
from g1cap.retreat_stop import RetreatStop

class StopTests(unittest.TestCase):
 def sample(self,c,t,x,ready=False):return c.update(t,t,x,ready)
 def test_stop_needs_settled_distance_and_does_not_finish_while_moving(self):
  c=RetreatStop(.2,0)
  self.sample(c,0,0)
  self.assertEqual(self.sample(c,.1,.15)['phase'],'settle')
  self.assertIsNone(self.sample(c,.2,.20)['outcome'])
  self.assertIsNone(self.sample(c,1.2,.20)['outcome'])
  self.assertEqual(self.sample(c,1.3,.20,True)['outcome'],'completed')
 def test_short_stop_can_refine_without_resetting_progress_or_clock(self):
  c=RetreatStop(.2,0);self.sample(c,0,0);self.sample(c,.1,.15)
  r=self.sample(c,1.2,.16,True)
  self.assertEqual(r['phase'],'drive');self.assertEqual(r['refinements'],1)
  r=self.sample(c,1.3,.16,True);self.assertGreater(r['speed_m_s'],0)
  self.assertEqual(r['remaining_m'],.2-.16)
  with self.assertRaisesRegex(ValueError,'timeout'):self.sample(c,12.01,.16)
 def test_overrun_and_wrong_direction_are_failures(self):
  for x,reason in [(.231,'overshoot'),(-.031,'wrong_direction')]:
   with self.assertRaisesRegex(ValueError,reason):self.sample(RetreatStop(.2,0),0,x)
 def test_camera_age_and_regressions_reject(self):
  c=RetreatStop(.2,0)
  with self.assertRaises(ValueError):c.update(.2,0,0,False)
  c=RetreatStop(.2,0);self.sample(c,.1,.01)
  with self.assertRaises(ValueError):c.update(.2,.05,.02,False)
 def test_magnitude_and_rate_limit_and_cached_frame_not_double_counted(self):
  c=RetreatStop(.2,0);speeds=[]
  for i in range(30):
   r=c.update(i*.02,(i//5)*.1,0.,False);speeds.append(r['speed_m_s'])
  self.assertLessEqual(max(speeds),.12)
  self.assertTrue(all(b-a<=.00600001 for a,b in zip(speeds,speeds[1:])))
  self.assertEqual(len(c.history),3)
 def test_refinements_bounded(self):
  c=RetreatStop(.2,0);self.sample(c,0,0)
  for i,x in enumerate([.10,.13,.15]):
   c.phase='settle';c.phase_started=i*2
   if i<2:self.sample(c,i*2+1.1,x,True)
   else:
    with self.assertRaisesRegex(ValueError,'refinement_limit'):self.sample(c,i*2+1.1,x,True)
if __name__=='__main__':unittest.main()
