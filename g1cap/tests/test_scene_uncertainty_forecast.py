import unittest
from types import SimpleNamespace
import numpy as np
from test_scene_uncertainty import ledger,frame
from test_scene_uncertainty_runtime import candidate
from g1cap._source_references import SourceFrontReferences
from g1cap._source_reference_screen import screen_references

FORECAST=dict(elapsed_s=2.,travel_m=1.,rotation_rad=2.)
class ForecastTests(unittest.TestCase):
 def test_forecast_increases_error_without_advancing_observation(self):
  u=ledger();u.observe(0.,frame(0.));c=u.snapshot()
  future=u.forecast(c,0.,**FORECAST)
  self.assertAlmostEqual(future['translation_error_m'],.131)
  self.assertAlmostEqual(future['rotation_error_rad'],np.deg2rad(2)+.0302)
  self.assertEqual(u.snapshot(),c);self.assertEqual(u.bounds(c,0.)['elapsed_s'],0.)
 def test_future_exhaustion_does_not_invalidate_present_stopping(self):
  u=ledger(limit=.12);u.observe(0.,frame(0.));c=u.snapshot()
  with self.assertRaises(ValueError):u.forecast(c,0.,**FORECAST)
  self.assertEqual(u.bounds(c,0.)['translation_error_m'],.1)
 def test_forecast_rejects_negative_nan_bool_and_zero_horizon(self):
  for key in FORECAST:
   for value in [-1,float('nan'),True]:
    u=ledger();u.observe(0.,frame(0.));args=dict(FORECAST);args[key]=value
    with self.assertRaises(ValueError):u.forecast(u.snapshot(),0.,**args)
  u=ledger();u.observe(0.,frame(0.))
  with self.assertRaises(ValueError):u.forecast(u.snapshot(),0.,elapsed_s=0,travel_m=0,rotation_rad=0)
 def fixture(self,limit=.2):
  u=ledger(limit=limit);u.observe(0.,frame(0.));owner=object();bank=SourceFrontReferences(owner,uncertainty=u)
  bank.observe(0.,[candidate()],owner,frame(0.))
  stop=SimpleNamespace(plane=None,front_memory=None,motion=frame(0.),front_gyro=np.eye(3))
  def evaluate():
   value=float(min(stop.plane.bounds([[-.6,0,0]],0.,np.eye(3))))
   return dict(status='clear' if value>=.03 else 'insufficient_margin',lower_m=value)
  return u,owner,bank,stop,evaluate
 def test_screen_uses_forecast_and_restores_original_objects(self):
  u,owner,bank,stop,evaluate=self.fixture()
  args=(stop,bank,owner,dict(time_s=0.),np.eye(3),evaluate)
  current=screen_references(*args);future=screen_references(*args,forecast=FORECAST)
  self.assertLess(future['lower_m'],current['lower_m'])
  self.assertEqual(future['source_observed_at_s'],0.)
  self.assertIsNone(stop.plane);self.assertIsNone(stop.front_memory)
  self.assertFalse(bank.latest.invalid)
 def test_rejected_future_still_allows_current_screen(self):
  u,owner,bank,stop,evaluate=self.fixture(limit=.12)
  args=(stop,bank,owner,dict(time_s=0.),np.eye(3),evaluate)
  with self.assertRaises(ValueError):screen_references(*args,forecast=FORECAST)
  self.assertFalse(bank.latest.invalid)
  self.assertEqual(screen_references(*args)['status'],'clear')
 def test_runtime_requires_explicit_forecast_but_stop_reads_present(self):
  from g1cap.scene_uncertainty_runtime import configure_scene_uncertainty,navigation_forecast
  config=dict(translation_growth=[.02,.005,.0005],rotation_growth=[.01,.01,.0001],max_translation_m=.20,max_rotation_rad=.15)
  with self.assertRaises(ValueError):configure_scene_uncertainty(dict(scene_uncertainty=config))
  u,f=configure_scene_uncertainty(dict(scene_uncertainty=config,scene_uncertainty_forecast=FORECAST))
  w=SimpleNamespace(scene_uncertainty=u,scene_uncertainty_forecast=f)
  self.assertEqual(navigation_forecast(w),FORECAST)
  self.assertIsNone(navigation_forecast(w,stopping=True))
  self.assertEqual(configure_scene_uncertainty({}),(None,None))
if __name__=='__main__':unittest.main()
