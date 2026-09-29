import unittest
from types import SimpleNamespace
from test_scene_uncertainty import frame,ledger
from g1cap.scene_uncertainty_runtime import update_scene_uncertainty
from g1cap._source_references import SourceFrontReferences
from g1cap.departure_reference import initialize_reference

def candidate():return dict(normal_body=[1,0,0],anchor_body=[.5,0,0],line=dict(observed_span_m=.2,points=50,rms_m=0.,endpoints_camera_m=[]))
class RuntimeUncertaintyTests(unittest.TestCase):
 def world(self):return SimpleNamespace(scene_uncertainty=ledger(),loaded_stop_clearance=SimpleNamespace())
 def test_initial_unavailable_is_not_a_latched_loss(self):
  w=self.world();owner=object();bank=SourceFrontReferences(owner,uncertainty=w.scene_uncertainty)
  m=dict(status='unavailable',time_s=0.,segment=0)
  update_scene_uncertainty(w,m)
  self.assertEqual(bank.observe(0.,[],owner,m)['status'],'unavailable');self.assertIsNone(bank.failure)
  update_scene_uncertainty(w,frame(.02));bank.observe(.02,[candidate()],owner,frame(.02))
  self.assertEqual(len(bank.available(.02,owner,frame(.02))),1)
 def test_both_tables_share_one_ledger_and_repeated_capture_is_idempotent(self):
  w=self.world();a=object();b=object();left=SourceFrontReferences(a,uncertainty=w.scene_uncertainty);right=SourceFrontReferences(b,uncertainty=w.scene_uncertainty)
  for t in [0.,.1]:
   m=frame(t,t);update_scene_uncertainty(w,m)
   left.observe(t,[candidate()],a,m);right.observe(t,[candidate()],b,m)
  self.assertAlmostEqual(w.scene_uncertainty.distance,.1)
  self.assertIs(w.loaded_stop_clearance.uncertainty,w.scene_uncertainty)
  self.assertIs(left.latest.uncertainty,right.latest.uncertainty)
 def test_departure_reference_reuses_ledger_without_reset(self):
  w=self.world();owner=SimpleNamespace(plane=True);w.box_perception=SimpleNamespace(source_plane=owner)
  update_scene_uncertainty(w,frame(0.));update_scene_uncertainty(w,frame(.1,.1))
  w.departure_source_input=dict(time_s=.1,owner=owner,motion=frame(.1,.1),candidates=[candidate()])
  bank=initialize_reference(w,.1)
  self.assertIs(bank.uncertainty,w.scene_uncertainty);self.assertAlmostEqual(w.scene_uncertainty.distance,.1)
  self.assertIs(initialize_reference(w,.1),bank)
 def test_tracking_loss_after_initialization_is_latched(self):
  w=self.world();update_scene_uncertainty(w,frame(0.));c=w.scene_uncertainty.snapshot()
  update_scene_uncertainty(w,dict(status='unavailable',time_s=.1,segment=1))
  update_scene_uncertainty(w,frame(.12))
  with self.assertRaises(ValueError):w.scene_uncertainty.bounds(c,.12)
 def test_legacy_world_is_unchanged(self):
  w=SimpleNamespace();self.assertIsNone(update_scene_uncertainty(w,frame(0.)))
if __name__=='__main__':unittest.main()
