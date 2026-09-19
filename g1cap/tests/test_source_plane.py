import unittest

try:
 import numpy as np
except ImportError:
 np=None
if np is not None:
 from g1cap.source_plane import SourcePlane

@unittest.skipIf(np is None,'NumPy runtime required')
class TrackingTests(unittest.TestCase):
 def setUp(self):
  self.track=SourcePlane();self.cam=np.eye(4)
  self.original=dict(status='observed_candidate',normal_camera=[0,0,1],offset_m=-.7,gap_m=.1)
  self.box=dict(status='accepted',center_camera_m=[1.,0,.9],axes_camera=np.eye(3).tolist(),dimensions_m=[.2]*3)
  x,y=np.meshgrid(np.linspace(-.5,.5,30),np.linspace(-.5,.5,30));self.points=np.c_[x.ravel(),y.ravel(),np.full(x.size,.7)]
  self.floor=self.points.copy();self.floor[:,:2]*=4;self.floor[:,2]=0
  self.track.observe(0,self.motion(0),self.cam,self.box,self.original,points=self.points,floor_points=self.floor)
 def motion(self,t,segment=1):return dict(status='tracked_local_segment',segment=segment,time_s=t,body_in_control_frame=np.eye(4).tolist())
 def observe(self,t,points,segment=1):return self.track.observe(t,self.motion(t,segment),self.cam,self.box,dict(status='unavailable'),points=points,floor_points=self.floor)
 def test_box_leaving_footprint_does_not_delete_visible_table(self):
  r=self.observe(.1,self.points);self.assertEqual(r['status'],'observed_candidate');self.assertAlmostEqual(r['gap_m'],.1)
  self.assertTrue(r['height_reference_only'])
 def test_initialization_requires_fresh_plane_evidence(self):
  track=SourcePlane()
  r=track.observe(0,self.motion(0),self.cam,self.box,self.original,points=np.empty((0,3)),floor_points=self.floor)
  self.assertEqual(r['status'],'unavailable')
  self.assertFalse(track.initialized)
 def test_partial_plane_is_accepted_after_explicit_association(self):
  points=self.points.copy();points[:,0]*=.3;points[:,1]*=.1
  r=self.observe(.1,points)
  self.assertEqual(r['status'],'observed_candidate')
  self.assertLess(min(r['observed_spans_m']),.25)
 def test_missing_floor_invalidates_existing_association(self):
  r=self.track.observe(.1,self.motion(.1),self.cam,self.box,self.original,
      points=self.points,floor_points=None)
  self.assertEqual(r['status'],'unavailable')
  self.assertEqual(self.observe(.2,self.points)['status'],'unavailable')
 def test_initial_ambiguous_geometry_is_not_installed(self):
  track=SourcePlane()
  r=track.observe(0,self.motion(0),self.cam,self.box,self.original,
      points=np.r_[self.points,self.points+[0,0,.02]],floor_points=self.floor)
  self.assertEqual(r['status'],'unavailable');self.assertFalse(track.initialized)
 def test_missing_plane_is_unavailable_not_cached(self):
  self.assertEqual(self.observe(.1,np.empty((0,3)))['status'],'unavailable')
  self.assertEqual(self.observe(.2,self.points)['status'],'observed_candidate')
 def test_segment_or_time_gap_requires_new_initial_association(self):
  self.assertEqual(self.observe(.1,self.points,2)['status'],'unavailable')
  self.setUp();self.assertEqual(self.observe(.2,self.points)['status'],'unavailable')
 def test_distant_floor_and_ambiguous_parallel_planes_rejected(self):
  self.assertEqual(self.observe(.1,self.points-[0,0,.5])['status'],'unavailable')
  r=self.observe(.2,np.r_[self.points,self.points+[0,0,.02]])
  self.assertEqual(r['status'],'unavailable');self.assertEqual(r['candidates'],2)
 def test_missing_box_still_allows_table_but_not_box_height(self):
  self.box=dict(status='unavailable');r=self.observe(.1,self.points)
  self.assertEqual(r['status'],'observed_candidate');self.assertIsNone(r['gap_m'])
 def test_lost_frame_does_not_silently_reassociate_another_table_under_box(self):
  frame=self.motion(.1);frame['status']='unavailable'
  self.track.observe(.1,frame,self.cam,self.box,self.original,points=self.points,floor_points=self.floor)
  r=self.track.observe(.2,self.motion(.2,2),self.cam,self.box,self.original,points=self.points,floor_points=self.floor)
  self.assertEqual(r['status'],'unavailable')
 def test_imprecise_box_height_preserves_fresh_obstacle_geometry(self):
  self.box['center_camera_m']=[10.,0.,.9]
  result=self.observe(.1,self.points)
  self.assertEqual(result['status'],'unavailable')
  self.assertEqual(result['reason'],'source_height_precision_insufficient')
  self.assertEqual(self.track.geometry['status'],'observed_candidate')
  self.assertGreater(self.track.geometry['prediction_quality']['height_perturbation_m'],.01)
  self.assertEqual(self.track.geometry['observed_at_s'],.1)
  self.observe(.2,np.empty((0,3)))
  self.assertIsNone(self.track.geometry)
 def test_initial_association_keeps_strict_height_precision(self):
  self.box['center_camera_m']=[10.,0.,.9];track=SourcePlane()
  result=track.observe(0,self.motion(0),self.cam,self.box,self.original,points=self.points,floor_points=self.floor)
  self.assertEqual(result['status'],'unavailable');self.assertFalse(track.initialized)
  self.assertIsNone(track.geometry)
if __name__=='__main__':unittest.main()
