import unittest
from types import SimpleNamespace as NS
import numpy as np
from g1cap.source_stop import SourceSides
from g1cap.remembered_source_plane import RememberedSourcePlane
from g1cap.departure_travel import screen_travel, tableward_navigation

def yaw(a):
 c,s=np.cos(a),np.sin(a);return np.array([[c,-s,0],[s,c,0],[0,0,1.]])

def plane(rotation=None,up=None):
 pose=np.eye(4);motion=dict(status='tracked_local_segment',segment=1,time_s=1.,body_in_segment=pose.tolist())
 memory=RememberedSourcePlane(1.,motion,[-1,0,0],0.,[0,0,0],plane_error_m=.003,plane_angle_rad=.01,max_age_s=10.)
 return SourceSides(None,memory,motion,np.eye(3),1.)

class TravelTests(unittest.TestCase):
 def test_current_heading_maps_segment_normal_to_native_axes(self):
  n=tableward_navigation(plane(),1.02,yaw(np.pi/2),np.array([0,0,1.]))
  np.testing.assert_allclose(n,[0,-1],atol=1e-12)
 def test_tilted_pelvis_uses_horizontal_heading(self):
  a=.3;c,s=np.cos(a),np.sin(a);R=np.array([[c,0,s],[0,1,0],[-s,0,c]])
  n=tableward_navigation(plane(),1.02,R,R.T@np.array([0,0,1.]))
  np.testing.assert_allclose(n,[1,0],atol=1e-12)
 def test_bad_reference_or_frame_is_rejected(self):
  for change,now in ((lambda p:None,11.01),(lambda p:setattr(p.memory,'invalid',True),1.02),
       (lambda p:p.motion.update(segment=2),1.02),(lambda p:None,float('nan')),
       (lambda p:p.motion.update(time_s=.7),1.02)):
   with self.subTest(now=now):
    p=plane();change(p)
    with self.assertRaises(ValueError):tableward_navigation(p,now,np.eye(3),np.array([0,0,1.]))
 def world(self):
  stop=NS(plane=plane(),up_world=np.array([0,0,1.]))
  w=NS(loaded_stop_clearance=stop,box_perception=NS(motion=NS(rotation=yaw(np.pi/2)),carry_seed={'marker':'camera'}),sensor_recorder=NS(latest_hands={'marker':'hands'}))
  w.screen_measured_source=lambda packet,evaluate:evaluate()
  return w
 def test_screens_actual_arms_and_maximum_retreat_without_mutation(self):
  w=self.world();previous=[0.]*50;proposed=[0.]*50;proposed[15]=.3;calls=[]
  def screen(*args):
   calls.append(args);return dict(status='clear',lower_m=.04)
  result=screen_travel(w,dict(time_s=1.02),NS(screen=screen),proposed,previous)
  np.testing.assert_allclose(calls[0][-2][43:46],[0,.08,0],atol=1e-12)
  self.assertEqual(calls[0][-2][15],.3);self.assertIs(calls[0][-1],previous)
  self.assertEqual(proposed[43:46],[0,0,0]);self.assertEqual(result['status'],'clear')
  np.testing.assert_allclose(result['front_normal_xy'],[0,-1],atol=1e-12)
 def test_clearance_rejection_is_preserved(self):
  result=screen_travel(self.world(),dict(time_s=1.02),NS(screen=lambda *a:dict(status='insufficient_margin',lower_m=.02)),[0.]*50,[0.]*50)
  self.assertEqual(result['status'],'insufficient_margin')
 def test_missing_reference_never_calls_probe(self):
  w=self.world();w.loaded_stop_clearance.plane=None;calls=[]
  with self.assertRaises(ValueError):screen_travel(w,dict(time_s=1.02),NS(screen=lambda *a:calls.append(a)),[0.]*50,[0.]*50)
  self.assertFalse(calls)

class RuntimeTravelTests(unittest.TestCase):
 def test_live_edge_loss_uses_complete_travel_certificate_then_stops_on_rejection(self):
  import io,json
  from unittest.mock import patch
  from g1cap.departure_runtime import RuntimeDeparture
  from g1cap.source_departure_adapter import SensorDeparture
  from g1cap.source_departure import SourceDeparture
  w=TravelTests().world();packet=dict(time_s=1.02);w.sensor_recorder.latest_packet=packet
  w.loaded_stop_evidence_error=None;w.loaded_stop_clearance.screen=lambda *a:dict(status='clear',lower_m=.06)
  w.control=NS(carry_feedback=lambda now:dict(status='available',time_s=now,track_epoch=1,segment=1,retained=True,settled=True))
  w.box_perception.control_frame=lambda now:dict(segment=1)
  w.approach=NS(feedback=lambda now:dict(status='unavailable'))
  w.observed_hand=NS(feedback=lambda now:dict(status='unavailable'))
  w.files={'source-departure':io.StringIO()}
  w.screen_measured_source=lambda packet,evaluate,**kw:evaluate()
  runtime=RuntimeDeparture.__new__(RuntimeDeparture);runtime.world=w;runtime.travel_probe=NS();runtime.probe=NS()
  initial=dict(track_epoch=1,segment=1,retained=True)
  runtime.controller=SourceDeparture(1.,-.5,initial)
  adapter=SensorDeparture(1.,-.5,initial,runtime.measure)
  certificate=dict(status='clear',lower_m=.04,front_normal_xy=[1.,0.],direction_basis='admitted_measured_source_front')
  with patch('g1cap.departure_runtime.screen_departure_sweep',side_effect=lambda *a,**kw:dict(status='insufficient_margin',lower_m=-.1,observed_at_s=packet['time_s'])), patch('g1cap.departure_travel.screen_travel',side_effect=lambda *a:dict(certificate)) as travel:
   action=adapter.screened_command(1.02,[0.]*50,[0.]*50)
   self.assertLess(action[43],0.);self.assertEqual(travel.call_count,1)
   row=json.loads(w.files['source-departure'].getvalue().splitlines()[-1])
   self.assertEqual(row['travel']['status'],'clear');self.assertFalse(row['evidence']['live_front_available'])
   certificate.update(status='insufficient_margin',lower_m=.02);packet['time_s']=1.04
   stopped=adapter.screened_command(1.04,[0.]*50,action)
   self.assertEqual(stopped[43:46],[0.,0.,0.]);self.assertEqual(adapter.controller.phase,'settle')
   certificate.update(status='clear',lower_m=.04);packet['time_s']=1.06
   self.assertEqual(adapter.screened_command(1.06,[0.]*50,stopped)[43:46],[0.,0.,0.])

 def test_available_live_path_keeps_its_existing_direction(self):
  import io,json
  from unittest.mock import patch
  from g1cap.departure_runtime import RuntimeDeparture
  from g1cap.source_departure_adapter import SensorDeparture
  from g1cap.source_departure import SourceDeparture
  w=TravelTests().world();packet=dict(time_s=1.02);w.sensor_recorder.latest_packet=packet
  w.loaded_stop_evidence_error=None;w.loaded_stop_clearance.screen=lambda *a:dict(status='clear',lower_m=.06)
  w.control=NS(carry_feedback=lambda now:dict(status='available',time_s=now,track_epoch=1,segment=1,retained=True,settled=True))
  w.box_perception.control_frame=lambda now:dict(segment=1)
  w.approach=NS(feedback=lambda now:dict(status='available',normal_navigation_xy=[.6,.8]))
  w.observed_hand=NS(feedback=lambda now:dict(status='available'))
  w.files={'source-departure':io.StringIO()}
  w.screen_measured_source=lambda packet,evaluate,**kw:evaluate()
  runtime=RuntimeDeparture.__new__(RuntimeDeparture);runtime.world=w;runtime.travel_probe=NS();runtime.probe=NS()
  initial=dict(track_epoch=1,segment=1,retained=True)
  runtime.controller=SourceDeparture(1.,-.5,initial)
  adapter=SensorDeparture(1.,-.5,initial,runtime.measure)
  with patch('g1cap.departure_runtime.screen_departure_sweep',return_value=dict(status='insufficient_margin',lower_m=-.1,observed_at_s=1.02)), patch('g1cap.departure_travel.screen_travel',return_value=dict(status='insufficient_margin',lower_m=-.15,front_normal_xy=[1.,0.],direction_basis='admitted_measured_source_front')) as travel:
   action=adapter.screened_command(1.02,[0.]*50,[0.]*50)
   np.testing.assert_allclose(action[43:46],[-.0036,-.0048,0],atol=1e-12)
   travel.assert_not_called()

if __name__=='__main__':unittest.main()
