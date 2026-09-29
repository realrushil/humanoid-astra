import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
from g1cap.source_stop import SourceStopClearance,SourceSides
from g1cap.loaded_stop import LoadedStopClearance,TopPlane
from g1cap.remembered_source_plane import RememberedSourcePlane

def motion(t,segment=1):return dict(status='tracked_local_segment',time_s=t,segment=segment,body_in_segment=np.eye(4).tolist())
def top(t):return TopPlane([0,0,1],0,[0,0,0],.003,.01,t,np.eye(3))
class CacheTests(unittest.TestCase):
 def setUp(self):
  self.p=SourceStopClearance.__new__(SourceStopClearance);p=self.p
  p.plane=None;p.front_memory=None;p.source_owner=NS(plane=object());p.front_lost=False;p.motion=motion(1.);p.front_gyro=np.eye(3);p.heights=[];p.cached_top=None;p.cached_top_segment=None
 def measure(self,t,source=None,floor=True):
  p=self.p
  def parent(this,packet,seed,*args):
   this.plane=source
   if floor:this.heights.append([t,.75,.003])
  with patch.object(LoadedStopClearance,'measure',parent):
   p.measure(dict(time_s=t),dict(time_s=t,box=dict(status='accepted')),np.eye(3),[0,0,1],source)
 def test_gap_retains_original_object_and_time_then_expires(self):
  p=self.p;a=top(1.);self.measure(1.,a)
  for t in (1.02,1.14):
   p.motion=motion(t);self.measure(t);self.assertIs(p.plane,a);self.assertEqual(p.plane.time,1.)
  p.motion=motion(1.16);self.measure(1.16);self.assertIsNone(p.plane)
  p.motion=motion(1.18);self.measure(1.18);self.assertIsNone(p.plane)
 def test_fresh_measurement_replaces_cached_object(self):
  self.measure(1.,top(1.));self.p.motion=motion(1.02);a=top(1.02);self.measure(1.02,a);self.assertIs(self.p.plane,a)
 def test_floor_segment_and_identity_loss_discard_cache(self):
  for loss in ('floor','segment','identity','motion'):
   with self.subTest(loss=loss):
    self.setUp();p=self.p;self.measure(1.,top(1.));p.motion=motion(1.02,2 if loss=='segment' else 1)
    if loss=='identity':p.source_owner.plane=None
    if loss=='motion':p.motion=dict(status='unavailable')
    self.measure(1.02,floor=loss!='floor');self.assertIsNone(p.plane)
    p.motion=motion(1.04);p.source_owner.plane=object();self.measure(1.04);self.assertIsNone(p.plane)
 def test_owner_change_discards_cache_before_query(self):
  p=self.p;self.measure(1.,top(1.));p.observe_front({},motion(1.02),np.eye(3),NS(plane=object()),False);self.measure(1.02);self.assertIsNone(p.plane)
 def test_sides_preserve_oldest_timestamp_for_vertical_prediction(self):
  memory=RememberedSourcePlane(1.,motion(1.),[1,0,0],0,[0,0,0],plane_error_m=.003,plane_angle_rad=.01,max_age_s=10.)
  side=SourceSides(top(1.),memory,motion(1.08),np.eye(3),1.08)
  self.assertEqual(side.time,1.)
  self.assertEqual(SourceSides(None,memory,motion(1.08),np.eye(3),1.08).time,1.08)
if __name__=='__main__':unittest.main()
