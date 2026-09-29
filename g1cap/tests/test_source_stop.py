import unittest
from types import SimpleNamespace
try:
 import numpy as np
 from g1cap.source_stop import SourceStopClearance as FrontStopProbe,SourceSides
 from g1cap.loaded_stop import TopPlane
 from g1cap.remembered_source_plane import RememberedSourcePlane
except ImportError:
 np=None
def motion(t=1.,segment=1):return dict(status='tracked_local_segment',time_s=t,segment=segment,body_in_segment=np.eye(4).tolist())
def front(t=1.,association='track'):return dict(status='observed_candidate',observed_at_s=t,association=association,normal_body=[-1.,0,0],offset_body_m=0.)

def probe():
 p=FrontStopProbe.__new__(FrontStopProbe)
 p.source_owner=None;p.front_memory=None;p.front_lost=False;p.motion=None;p.front_gyro=None
 return p
@unittest.skipIf(np is None,"native numerical runtime required")
class FrontTests(unittest.TestCase):
 def test_stale_front_cannot_be_retimed(self):
  p=probe();p.observe_front(front(),motion(2.),np.eye(3),SimpleNamespace(initialized=True,plane=object()),True);self.assertIsNone(p.front_memory)
 def test_new_source_owner_starts_empty(self):
  p=probe();a=SimpleNamespace(initialized=True,plane=object());p.observe_front(front(),motion(),np.eye(3),a,True)
  p.observe_front({},motion(1.04),np.eye(3),SimpleNamespace(initialized=True,plane=object()),False);self.assertIsNone(p.front_memory)
 def test_dropouts_and_segment_changes_latch(self):
  for m in ({'status':'unavailable'},motion(1.04,2)):
   p=probe();a=SimpleNamespace(initialized=True,plane=object());p.observe_front(front(),motion(),np.eye(3),a,True)
   p.observe_front({},m,np.eye(3),a,False);self.assertTrue(p.front_memory.invalid)
 def test_loss_even_when_top_missing_freezes_original(self):
  p=probe();a=SimpleNamespace(initialized=True,plane=object());p.observe_front(front(),motion(),np.eye(3),a,True)
  p.observe_front({},motion(1.04),np.eye(3),a,False)
  p.observe_front(front(1.08),motion(1.08),np.eye(3),a,True)
  self.assertEqual(p.front_memory.time,1.)
 def test_source_tracker_identity_loss_invalidates_original_front(self):
  p=probe();a=SimpleNamespace(initialized=True,plane=object())
  p.observe_front(front(),motion(),np.eye(3),a,True)
  a.plane=None
  p.observe_front({},motion(1.04),np.eye(3),a,False)
  self.assertTrue(p.front_memory.invalid)
 def test_duplicate_capture_does_not_renew_memory(self):
  p=probe();a=SimpleNamespace(initialized=True,plane=object());p.observe_front(front(),motion(),np.eye(3),a,True)
  p.observe_front(front(),motion(1.04),np.eye(3),a,True);self.assertEqual(p.front_memory.time,1.)
@unittest.skipIf(np is None,"native numerical runtime required")
class SideTests(unittest.TestCase):
 def side(self,top=True,m=None,now=1.):
  t=TopPlane([0,0,1],0,[0,0,0],0,0,now,np.eye(3)) if top else None
  a=RememberedSourcePlane(1.,motion(),[1,0,0],0,[0,0,0],plane_error_m=.003,plane_angle_rad=np.deg2rad(.5),max_age_s=10.)
  return SourceSides(t,a,motion(now) if m is None else m,np.eye(3),now)
 def test_front_can_screen_without_top(self):
  self.assertGreater(min(self.side(False).bounds([[1,0,-1]],1.,np.eye(3))),.03)
 def test_no_top_and_bad_motion_fails(self):
  for m in ({'status':'unavailable'},motion(.5),motion(1.1),motion(1.,2)):
   with self.assertRaises(ValueError):self.side(False,m).bounds([[1,0,-1]],1.,np.eye(3))
 def test_missing_front_can_use_fresh_top(self):
  u=self.side();u.memory.invalid=True
  self.assertEqual(u.bounds([[1,0,-1]],1.,np.eye(3))[0],-1.)
 def test_expiry_between_cameras(self):
  with self.assertRaises(ValueError):self.side(False,now=11.).bounds([[1,0,-1]],11.02,np.eye(3))
 def test_corner_straddling_fails(self):
  self.assertLess(min(self.side().bounds([[1,0,-1],[-1,0,1]],1.,np.eye(3))),0)


@unittest.skipIf(np is None,"native numerical runtime required")
class EndToEndTests(unittest.TestCase):
 def prepare(self,mode):
  import json
  from pathlib import Path
  from copy import deepcopy
  from g1cap.sensor_kinematics import camera_in_body
  from g1cap.hand_sensors import measured_joint_positions
  f=json.loads(Path('tests/fixtures/source-stop-retention.json').read_text())
  stop=FrontStopProbe('g1cap/assets/arena_g1_rev1_0_kinematics.urdf','g1cap/assets/arena_g1_rev1_0_bounds.json',f['action_joint_names'])
  packet=deepcopy(f['packet']);hands=deepcopy(f['hands']);box=deepcopy(f['visual']['box'])
  camera=camera_in_body(stop.path.model,dict(zip(packet['joint_names'],packet['q_rad'])),f['calibration'])
  n=np.array(f['visual']['floor']['normal_body']);d=f['visual']['floor']['offset_m']
  tangent=np.cross(n,[1,0,0]);tangent/=np.linalg.norm(tangent);other=np.cross(n,tangent)
  points=np.array([-d*n+x*tangent+y*other for x in np.linspace(-1,1,20) for y in np.linspace(-1,1,20)])
  floor=(points-camera[:3,3])@camera[:3,:3]
  packet['specific_force_m_s2']=(9.81*n).tolist()
  owner=SimpleNamespace(initialized=True,plane=object())
  for step in range(31):
   t=step*.02;packet.update(time_s=t,step=step);hands.update(time_s=t,step=step);box['observed_at_s']=t
   if mode=='retention_loss' and step==30:box['status']='unavailable'
   if mode=='identity_loss' and step==30:owner.plane=None
   observation=dict(front(t),normal_body=[1.,0,0],offset_body_m=-1.) if step==0 else {}
   stop.observe_front(observation,motion(t),np.eye(3),owner,step==0)
   seed=dict(time_s=t,camera_transform=camera,box=box,packet=deepcopy(packet),calibration=f['calibration'],floor_points=floor,gyro_rotation=np.eye(3))
   stop.measure(packet,seed,np.eye(3),n,None)
  if mode=='expired':stop.front_memory.max_age=.5
  q=measured_joint_positions(packet,hands,f['action_joint_names']);action=list(q)+[0.]*7
  return stop,packet,hands,action
 def test_clear_front_without_any_top_passes_real_measure_screen(self):
  stop,p,h,a=self.prepare('clear')
  self.assertIsNone(stop.plane.top)
  result=stop.screen(p,h,np.eye(3),a)
  self.assertEqual(result['status'],'clear')
  self.assertEqual(result['source_reference'],'remembered_front')
 def test_expiry_retention_and_identity_loss_fail_real_path(self):
  for mode in ('expired','retention_loss','identity_loss'):
   with self.subTest(mode=mode):
    stop,p,h,a=self.prepare(mode)
    with self.assertRaises(ValueError):stop.screen(p,h,np.eye(3),a)
