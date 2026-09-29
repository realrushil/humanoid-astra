"""Unwired approach contract; these tests provide a synthetic path screen."""
import math,unittest
from g1cap.sensor_destination_approach import SensorDestinationApproach

def sample(t,x=0.,yaw=0.,ready=True):
 c,s=math.cos(yaw),math.sin(yaw)
 return dict(navigation=dict(time_s=t,yaw_segment_rad=yaw,segment=1),destination=dict(status='observed_destination_candidate',observed_at_s=t,
  direction_status='available',direction_segment_xy=[1.,0.],segment_id=1,view_epoch=1),
  grasp=dict(status='available',time_s=t,raised=True,ready=ready,opposing_near_wrists=True,attitude_ok=True),
  frame=dict(status='tracked_local_segment',time_s=t,segment=1,
   body_in_segment=[[c,-s,0,x],[s,c,0,0],[0,0,1,0],[0,0,0,1]]))

class SensorDestinationApproachTests(unittest.TestCase):
 def setUp(self):self.current=sample(0.);self.wrist=True;self.screened=[]
 def screen(self,t,s,a):
  self.screened.append(a.copy());return dict(status='clear',observed_at_s=s['destination']['observed_at_s'])
 def controller(self):return SensorDestinationApproach(0.,.20,lambda _:self.current,lambda:self.wrist,screen=self.screen)
 def test_missing_screen_cannot_construct_motion(self):
  with self.assertRaises((TypeError,ValueError)):SensorDestinationApproach(0.,.2,lambda _:self.current,lambda:True)
 def test_direction_converts_to_current_navigation_heading(self):
  self.current=sample(0.,yaw=math.pi/2);c=self.controller();self.current=sample(.02,yaw=math.pi/2)
  a=c.command(.02,[0.]*50);self.assertAlmostEqual(a[43],0.,places=8);self.assertLess(a[44],0.);self.assertEqual(a[45],0.)
  self.assertEqual(a,self.screened[-1])
 def test_held_camera_uses_current_imu_navigation_heading(self):
  c=self.controller();self.current=sample(0.)
  self.current['navigation']=dict(time_s=.02,yaw_segment_rad=math.pi/2,segment=1)
  a=c.command(.02,[0.]*50);self.assertAlmostEqual(a[43],0.,places=8);self.assertLess(a[44],0.)
 def test_old_navigation_heading_rejects(self):
  c=self.controller();self.current=sample(.02);self.current['navigation']['time_s']=0.
  with self.assertRaises(ValueError):c.command(.02,[0.]*50)
 def test_moving_retention_does_not_require_stationary_readiness(self):
  c=self.controller();self.current=sample(.02,ready=False);self.assertGreater(c.command(.02,[0.]*50)[43],0.)
 def test_admission_requires_stationary_grasp(self):
  self.current=sample(0.,ready=False)
  with self.assertRaisesRegex(ValueError,'initial_grasp'):self.controller()
 def test_wrist_handoff_is_zero_then_times_out(self):
  c=self.controller();self.wrist=False
  for i in range(1,8):
   self.current=sample(i*.02);self.assertEqual(c.command(i*.02,[0.]*50)[43:46],[0.,0.,0.])
  self.current=sample(.16)
  with self.assertRaisesRegex(ValueError,'wrist'):c.command(.16,[0.]*50)
 def test_bad_path_screen_latches_failure(self):
  c=self.controller();c.screen=lambda *args:dict(status='unavailable');self.current=sample(.02)
  with self.assertRaisesRegex(ValueError,'path'):c.command(.02,[0.]*50)
  c.screen=self.screen;self.current=sample(.04)
  with self.assertRaises(ValueError):c.command(.04,[0.]*50)
 def test_stale_path_certificate_cannot_advance(self):
  c=self.controller();c.screen=lambda *args:dict(status='clear',observed_at_s=0.);self.current=sample(.02)
  with self.assertRaisesRegex(ValueError,'path'):c.command(.02,[0.]*50)
 def test_candidate_loss_is_latched(self):
  c=self.controller();self.current=sample(.02);self.current['destination']['status']='unavailable'
  with self.assertRaises(ValueError):c.command(.02,[0.]*50)
  self.current=sample(.04)
  with self.assertRaises(ValueError):c.command(.04,[0.]*50)
 def test_segment_epoch_and_timestamp_mismatch_reject(self):
  for field,value in [('segment_id',2),('view_epoch',2),('observed_at_s',.01)]:
   self.current=sample(0.);c=self.controller();self.current=sample(.02);self.current['destination'][field]=value
   with self.subTest(field=field),self.assertRaises(ValueError):c.command(.02,[0.]*50)
 def test_reflection_and_nonrigid_frame_reject(self):
  for value in (-1,2):
   self.current=sample(0.);self.current['frame']['body_in_segment'][0][0]=value
   with self.assertRaises(ValueError):self.controller()
 def test_clock_jump_and_reversal_reject(self):
  for t in (-.02,.2):
   self.current=sample(0.);c=self.controller();self.current=sample(t)
   with self.assertRaises(ValueError):c.command(t,[0.]*50)
 def test_progress_reversal_and_overshoot_reject(self):
  for x in (-.04,.24):
   self.current=sample(0.);c=self.controller();self.current=sample(.02,x)
   with self.assertRaises(ValueError):c.command(.02,[0.]*50)
 def test_native_update_then_command_order_moves_and_settles(self):
  c=self.controller();a=c.command(0.,[0.]*50);x=0.;maximum=0.;outcome=None
  for i in range(1,231):
   t=i*.02;x+=a[43]*.02;self.current=sample(t,x)
   outcome=c.update(t)
   if outcome is not None:break
   a=c.command(t,a);maximum=max(maximum,a[43])
  self.assertGreater(maximum,.05)
  self.assertEqual(outcome,('completed','destination_approach_and_hold'))
 def test_requires_one_second_after_zero_navigation(self):
  c=self.controller()
  for i in range(1,11):
   t=i*.02;self.current=sample(t,i*.02);c.command(t,[0.]*50);self.assertIsNone(c.update(t))
  for i in range(11,56):
   t=i*.02;self.current=sample(t,.20);self.assertEqual(c.command(t,[0.]*50)[43:46],[0.,0.,0.]);self.assertIsNone(c.update(t))
  for i in range(56,70):
   t=i*.02;self.current=sample(t,.20);c.command(t,[0.]*50);result=c.update(t)
   if result is not None:break
  self.assertEqual(result,('completed','destination_approach_and_hold'))
 def test_updates_without_current_zero_command_cannot_complete(self):
  c=self.controller();self.current=sample(.02,.20);c.command(.02,[0.]*50);c.update(.02)
  self.current=sample(.06,.20)
  with self.assertRaisesRegex(ValueError,'command_stale'):c.update(.06)
 def test_repeated_camera_never_completes_settling(self):
  c=self.controller();self.current=sample(.02,.20);c.command(.02,[0.]*50)
  self.assertIsNone(c.update(.02))
  for i in range(2,7):
   self.current['navigation']['time_s']=i*.02
   c.command(i*.02,[0.]*50);self.assertIsNone(c.update(i*.02))
  with self.assertRaises(ValueError):c.update(.18)
 def test_settling_drift_cancels_readiness_window(self):
  c=self.controller()
  for i in range(1,51):
   t=i*.02;self.current=sample(t,.20);c.command(t,[0.]*50);self.assertIsNone(c.update(t))
  self.current=sample(1.02,.20,ready=False);c.command(1.02,[0.]*50);self.assertIsNone(c.update(1.02))
 def test_lost_opposition_stops_even_if_ready_flag_true(self):
  c=self.controller();self.current=sample(.02);self.current['grasp']['opposing_near_wrists']=False
  with self.assertRaises(ValueError):c.command(.02,[0.]*50)
if __name__=='__main__':unittest.main()
