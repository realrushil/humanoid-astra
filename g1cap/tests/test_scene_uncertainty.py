import unittest
import numpy as np
from g1cap.scene_uncertainty import SceneUncertainty
from g1cap.remembered_source_plane import RememberedSourcePlane

def frame(t,x=0.,yaw=0.,segment=1):
 c,s=np.cos(yaw),np.sin(yaw)
 return dict(status='tracked_local_segment',time_s=t,segment=segment,
             body_in_segment=[[c,-s,0,x],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
def ledger(**kw):
 return SceneUncertainty(translation_growth=(.02,.005,.0005),rotation_growth=(.01,.01,.0001),
   max_translation_m=kw.get('limit',.20),max_rotation_rad=.15)
class SceneUncertaintyTests(unittest.TestCase):
 def test_loop_accumulates_and_queries_do_not_update(self):
  u=ledger();u.observe(0.,frame(0.));capture=u.snapshot()
  u.observe(.1,frame(.1,1.,.2));u.observe(.2,frame(.2))
  a=u.bounds(capture,.2);self.assertEqual(a,u.bounds(capture,.2))
  self.assertAlmostEqual(a['distance_m'],2.);self.assertAlmostEqual(a['rotation_rad'],.4)
  self.assertAlmostEqual(a['translation_error_m'],.1421)
  self.assertAlmostEqual(a['rotation_error_rad'],np.deg2rad(2)+.02402)
 def test_identical_repeat_no_growth_changed_repeat_invalidates(self):
  u=ledger();u.observe(0.,frame(0.));c=u.snapshot();u.observe(0.,frame(0.))
  self.assertEqual(u.bounds(c,0.)['distance_m'],0.)
  with self.assertRaises(ValueError):u.observe(0.,frame(0.,.1))
  with self.assertRaises(ValueError):u.observe(.1,frame(.1))
 def test_continuity_failures_latch(self):
  cases=[(-.1,frame(-.1)),(.2,frame(.2)),(.1,frame(.1,segment=2)),(float('nan'),frame(.1))]
  reflected=frame(.1);reflected['body_in_segment'][0][0]=-1;cases.append((.1,reflected))
  mismatch=frame(.1);mismatch['time_s']=.2;cases.append((.1,mismatch))
  for t,f in cases:
   with self.subTest(t=t):
    u=ledger();u.observe(0.,frame(0.));c=u.snapshot()
    with self.assertRaises(ValueError):u.observe(t,f)
    with self.assertRaises(ValueError):u.bounds(c,.1)
 def test_query_clock_and_foreign_capture(self):
  u=ledger();v=ledger();u.observe(0.,frame(0.));v.observe(0.,frame(0.))
  for t in [-.1,.2,float('nan'),True]:
   with self.assertRaises(ValueError):u.bounds(u.snapshot(),t)
  with self.assertRaises(ValueError):u.bounds(v.snapshot(),0.)
 def test_refresh_does_not_reset_ledger_and_exhaustion_is_per_capture(self):
  u=ledger(limit=.12);u.observe(0.,frame(0.));old=u.snapshot()
  u.observe(.1,frame(.1,1.));fresh=u.snapshot()
  with self.assertRaises(ValueError):u.bounds(old,.1)
  self.assertAlmostEqual(u.bounds(fresh,.1)['translation_error_m'],.10)
  self.assertEqual(fresh.distance_m,1.)
 def test_configuration_rejects_bad_rates_and_ceilings(self):
  for rate in [(-1,0,0),(0,0,float('nan')),(0,0), (True,0,0)]:
   with self.assertRaises(ValueError):SceneUncertainty(translation_growth=rate,rotation_growth=(.01,.01,.0001),max_translation_m=.2,max_rotation_rad=.15)
  with self.assertRaises(ValueError):ledger(limit=.09)
 def test_memory_outlives_ten_seconds_without_renewing_capture(self):
  u=ledger();u.observe(0.,frame(0.))
  m=RememberedSourcePlane(0.,frame(0.),[1,0,0],0,[0,0,0],plane_error_m=.003,plane_angle_rad=0,max_age_s=10.,uncertainty=u)
  for i in range(1,111):u.observe(i*.1,frame(i*.1))
  r=m.query(11.,frame(11.),[[1,0,0]],translation_error_m=.1,rotation_error_rad=np.deg2rad(2))
  self.assertEqual(r['observed_at_s'],0.);self.assertEqual(r['age_s'],11.)
  self.assertGreater(r['uncertainty_m'][0],.103)
  self.assertAlmostEqual(r['pose_translation_error_m'],.1055)
 def test_added_uncertainty_never_improves_clearance(self):
  u=ledger();u.observe(0.,frame(0.));args=(0.,frame(0.),[1,0,0],0,[0,0,0]);kw=dict(plane_error_m=.003,plane_angle_rad=.01,max_age_s=10.)
  old=RememberedSourcePlane(*args,**kw);new=RememberedSourcePlane(*args,**kw,uncertainty=u)
  u.observe(.1,frame(.1,.01));points=[[1,0,0],[1,1,0]]
  a=old.query(.1,frame(.1,.01),points,translation_error_m=.1,rotation_error_rad=np.deg2rad(2))
  b=new.query(.1,frame(.1,.01),points,translation_error_m=.1,rotation_error_rad=np.deg2rad(2))
  self.assertTrue(np.all(np.asarray(b['lower_m'])<=a['lower_m']))
 def test_prepared_and_batched_screen_include_growth(self):
  from g1cap.source_stop import SourceSides
  from g1cap._prepared_source_sides import prepare
  u=ledger();u.observe(0.,frame(0.))
  m=RememberedSourcePlane(0.,frame(0.),[1,0,0],0,[0,0,0],plane_error_m=.003,plane_angle_rad=.01,max_age_s=10.,uncertainty=u)
  u.observe(.1,frame(.1,.1,.1));source=SourceSides(None,m,frame(.1,.1,.1),np.eye(3),.1)
  prepared=prepare(source,.1,np.eye(3));points=np.array([[1.,0,0],[1.,1,0]])
  kwargs=dict(front_closing_m=.1,point_displacement_m=.12)
  expected=source.bounds(points,.1,np.eye(3),**kwargs)
  np.testing.assert_allclose(prepared.bounds(points,.1,np.eye(3),**kwargs),expected,atol=1e-12)
  np.testing.assert_allclose(prepared.bounds_many(points[None],.1,np.eye(3),**kwargs)[0],expected,atol=1e-12)
  expected_difference=2*np.sin(u.bounds(m.capture,.1)['rotation_error_rad']/2)+2*np.sin(m.angle/2)
  self.assertAlmostEqual(source.front_normal_difference,expected_difference)
 def test_bank_outlives_expiry_and_bounds_storage(self):
  from g1cap._source_references import SourceFrontReferences
  u=ledger();owner=object();bank=SourceFrontReferences(owner,uncertainty=u)
  candidate=dict(normal_body=[1,0,0],anchor_body=[.5,0,0],line=dict(observed_span_m=.2,points=50,rms_m=0.,endpoints_camera_m=[]))
  bank.observe(0.,[candidate],owner,frame(0.))
  for i in range(1,111):bank.observe(i*.1,[],owner,frame(i*.1))
  self.assertIsNone(bank.failure)
  self.assertEqual([m.time for m in bank.available(11.,owner,frame(11.))],[0.])
  self.assertEqual(bank.observe(11.1,[candidate],owner,frame(11.1))['status'],'observed_source_front')
  for i in range(112,251):bank.observe(i*.1,[candidate],owner,frame(i*.1))
  self.assertLessEqual(len(bank.history),10)
  self.assertGreater(u.snapshot().time_s,24.)
 def test_shared_ledger_loss_invalidates_bank(self):
  from g1cap._source_references import SourceFrontReferences
  u=ledger();owner=object();bank=SourceFrontReferences(owner,uncertainty=u)
  c=dict(normal_body=[1,0,0],anchor_body=[.5,0,0],line=dict(observed_span_m=.2,points=50,rms_m=0.,endpoints_camera_m=[]))
  bank.observe(0.,[c],owner,frame(0.))
  bank.observe(.1,[],owner,frame(.1,segment=2))
  self.assertEqual(bank.available(.1,owner,frame(.1,segment=2)),[])
 def test_cached_top_age_does_not_rewind_uncertainty_query(self):
  from g1cap.source_stop import SourceSides
  from g1cap.loaded_stop import TopPlane
  from g1cap._prepared_source_sides import prepare
  u=ledger();u.observe(0.,frame(0.))
  m=RememberedSourcePlane(0.,frame(0.),[1,0,0],0,[0,0,0],plane_error_m=.003,plane_angle_rad=.01,max_age_s=10.,uncertainty=u)
  top=TopPlane([0,0,1],0,[0,0,0],.003,.01,0.,np.eye(3))
  u.observe(.1,frame(.1))
  source=SourceSides(top,m,frame(.1),np.eye(3),.1)
  self.assertEqual(source.time,0.)  # vertical prediction retains original top age
  prepared=prepare(source,.1,np.eye(3))
  self.assertFalse(m.invalid);self.assertIsNotNone(prepared.front_data)
  points=np.array([[1.,0.,1.],[1.,1.,1.]])
  np.testing.assert_allclose(prepared.bounds(points,.1,np.eye(3)),source.bounds(points,.1,np.eye(3)),atol=1e-12)
if __name__=='__main__':unittest.main()
