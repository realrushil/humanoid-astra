import unittest
from test_arena_retreat import state
from g1cap.arena_placement import PlaceOnSurface,release_ready
ACTION=[0.]*46+[.75,0.,0.,0.]
def obs(i,supported=False,released=False):
 r=state(i);r['robot_source_peak_N']=0.;r['tilt']=0.
 r['surfaces']={'destination':dict(contained=True,supported=supported,clearance_m=0. if supported else .08,robot_contact_peak_N=0.,box_upward_N=1. if supported else 0.,bounds=dict(min=[-.5,-.5,-.04],max=[.5,.5,0.]))}
 r['hand_forces_N']={'left':0. if released else 10.,'right':0. if released else 10.}
 r['bilateral']=not released
 return r
class DummyWrists:
 def command(self,*args):return args[-2]
class PlacementTests(unittest.TestCase):
 def test_no_withdrawal_until_measured_destination_support_is_sustained(self):
  calls=[]
  c=PlaceOnSurface(obs(50),ACTION,'destination',lambda r,t:calls.append(r['time']) or DummyWrists())
  for i in range(51,75):c.update(obs(i))
  self.assertEqual(c.stage,'lower');self.assertFalse(calls)
  for i in range(75,90):c.update(obs(i,True))
  self.assertEqual(c.stage,'supported_hold');self.assertFalse(calls)
  for i in range(90,191):c.update(obs(i,True))
  self.assertEqual(c.stage,'withdraw');self.assertEqual(len(calls),1)
 def test_source_support_does_not_substitute_for_destination(self):
  c=PlaceOnSurface(obs(50),ACTION,'destination',lambda r,t:DummyWrists())
  r=obs(51,False,True);r['supported']=True
  self.assertEqual(c.update(r),('failed','retention_lost_before_support'))
  self.assertEqual(c.update(obs(52,True)),('failed','retention_lost_before_support'))
 def test_unreachable_lowering_and_uncontained_box_rejected(self):
  r=obs(50);r['surfaces']['destination']['clearance_m']=.15
  with self.assertRaisesRegex(ValueError,'lowering'):PlaceOnSurface(r,ACTION,'destination',lambda *a:None)
  r=obs(50);r['surfaces']['destination']['contained']=False
  with self.assertRaisesRegex(ValueError,'contained'):PlaceOnSurface(r,ACTION,'destination',lambda *a:None)
 def test_final_release_needs_both_hands_and_full_stable_second(self):
  rows=[obs(i,True,True) for i in range(51)]
  self.assertTrue(release_ready(rows,'destination'));rows[-1]['hand_forces_N']['right']=2.
  self.assertFalse(release_ready(rows,'destination'));self.assertFalse(release_ready(rows[:20],'destination'))
 def test_withdrawal_can_settle_within_geometry_but_final_support_is_required(self):
  c=PlaceOnSurface(obs(50),ACTION,'destination',lambda *a:DummyWrists())
  for i in range(51,166):c.update(obs(i,True))
  self.assertEqual(c.stage,'withdraw')
  r=obs(166);r['surfaces']['destination']['clearance_m']=.00008
  self.assertIsNone(c.update(r))
  for i in range(167,470):
   r=obs(i,True,True);r['surfaces']['destination']['supported']=False
   c.update(r)
  self.assertEqual(c.result,('failed','release_not_verified'))
 def test_withdrawal_never_allows_box_outside_height_envelope(self):
  c=PlaceOnSurface(obs(50),ACTION,'destination',lambda *a:DummyWrists())
  for i in range(51,166):c.update(obs(i,True))
  r=obs(166);r['surfaces']['destination']['clearance_m']=.0101
  self.assertEqual(c.update(r),('failed','box_left_selected_surface'))
 def test_support_required_through_prewithdrawal_dwell(self):
  c=PlaceOnSurface(obs(50),ACTION,'destination',lambda *a:DummyWrists())
  for i in range(51,66):c.update(obs(i,True))
  r=obs(66);r['surfaces']['destination']['clearance_m']=.00008
  self.assertEqual(c.update(r),('failed','selected_support_lost'))
 def test_navigation_is_zero_and_pelvis_lowering_is_bounded(self):
  c=PlaceOnSurface(obs(50),ACTION,'destination',lambda *a:None)
  a=c.command(obs(500));self.assertEqual(a[43:46],[0.,0.,0.]);self.assertAlmostEqual(a[46],.61)
if __name__=='__main__':unittest.main()


from g1cap.arena_control import BoxControl
class PlacementDispatchTests(unittest.TestCase):
 def control(self):
  c=BoxControl(ACTION,lambda o:ACTION,lambda *a:DummyWrists(),
   placement_factory=lambda r,a,s:PlaceOnSurface(r,a,s,lambda *args:DummyWrists()))
  for i in range(51):c.update(obs(i))
  return c
 def test_owner_placement_factory_receives_actual_action_and_selected_surface(self):
  c=self.control();calls=[]
  def placement(row,action,surface_id):
   calls.append((row,list(action),surface_id))
   return PlaceOnSurface(row,action,surface_id,lambda *a:DummyWrists())
  c.placement_factory=placement
  r=obs(50);before=list(c.last_action)
  self.assertEqual(c.start('place_box',{'surface_id':'destination'},r)['status'],'running')
  self.assertEqual(len(calls),1);self.assertIs(calls[0][0],r)
  self.assertEqual(calls[0][1],before);self.assertEqual(calls[0][2],'destination')
 def test_missing_selected_surface_rejected_and_fault_during_placement_latches(self):
  c=self.control();r=obs(50);del r['surfaces']
  self.assertEqual(c.start('place_box',{'surface_id':'destination'},r)['reason'],'missing_surface_observation')
  c=self.control();self.assertEqual(c.start('place_box',{'surface_id':'destination'},obs(50))['status'],'running')
  r=obs(51);r['surfaces']['destination']['robot_contact_peak_N']=6.
  self.assertEqual(c.update(r)['reason'],'forbidden_destination_contact')
  self.assertEqual(c.command(r)[43:46],[0.,0.,0.])
 def test_source_return_uses_same_selected_support_controller(self):
  c=self.control();r=obs(50)
  r['surfaces']['source']=r['surfaces'].pop('destination')
  self.assertEqual(c.start('return_box_to_source',{},r)['status'],'running')
  self.assertEqual(c.placement.surface_id,'source')
  self.assertEqual(c.phase,'place_lower')
 def test_successful_release_clears_preparation_ownership(self):
  c=self.control();c.grasp_prepared=True;c.preparation=object()
  self.assertEqual(c.start('place_box',{'surface_id':'destination'},obs(50))['status'],'running')
  for i in range(51,480):
   r=obs(i,True,True);c.command(r);c.update(r)
   if c.result:break
  self.assertEqual(c.result['status'],'completed',c.result)
  self.assertFalse(c.grasp_prepared);self.assertIsNone(c.preparation)
