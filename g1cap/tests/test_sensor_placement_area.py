"""Coverage must reject unknown interior cells and mismatched camera samples."""
import unittest
import numpy as np
from g1cap.sensor_placement_area import observe_footprint

class PlacementAreaTests(unittest.TestCase):
    def setUp(self):
        self.rgb=np.full((9,9,3),[50,90,50],np.uint8)
        self.depth=np.ones((9,9))
        self.k=np.array([[10.,0,4],[0,10.,4],[0,0,1.]])
        self.candidate=dict(status='observed_destination_candidate',observed_at_s=1.,normal_camera=[0,0,-1],offset_m=1.)
        self.corners=np.array([[-.1,-.1,1],[.1,-.1,1],[.1,.1,1],[-.1,.1,1]])
    def call(self,**kw):
        args=dict(candidate=self.candidate,rgb=self.rgb,depth_m=self.depth,intrinsic=self.k,corners_camera_m=self.corners,image_time_s=1.,now_s=1.1,erosion_pixels=0)
        args.update(kw);return observe_footprint(**args)
    def test_observed_rectangle_is_evidence_only(self):
        r=self.call();self.assertEqual(r['status'],'observed_footprint');self.assertEqual(r['tested_pixels'],9);self.assertFalse(r['placement_authorized']);self.assertEqual(r['frame'],'camera_optical_at_observation_time')
    def test_missing_interior_depth_unknown(self):
        self.depth[4,4]=np.nan;r=self.call();self.assertEqual(r['status'],'unknown');self.assertEqual(r['unknown_pixels'],1)
    def test_non_green_interior_unknown(self):
        self.rgb[4,4]=0;self.assertEqual(self.call()['status'],'unknown')
    def test_off_plane_interior_unknown(self):
        self.depth[4,4]=1.02;self.assertEqual(self.call()['status'],'unknown')
    def test_erosion_checks_neighbor_outside_footprint(self):
        self.depth[4,2]=np.nan;self.assertEqual(self.call()['status'],'observed_footprint');self.assertEqual(self.call(erosion_pixels=1)['status'],'unknown')
    def test_boundary_clipping_unknown(self):
        corners=self.corners.copy();corners[:,0]-=.4;self.assertEqual(self.call(corners_camera_m=corners)['reason'],'footprint_outside_image')
    def test_stale_future_and_mismatched_sample_unknown(self):
        for kw in [dict(now_s=1.151),dict(now_s=.99),dict(image_time_s=1.01)]:
            with self.subTest(kw=kw):self.assertEqual(self.call(**kw)['status'],'unknown')
    def test_unavailable_candidate_unknown(self):
        self.assertEqual(self.call(candidate={'status':'unavailable'})['status'],'unknown')
    def test_invalid_footprints_rejected(self):
        for corners in [self.corners[[0,2,1,3]],self.corners[[0,0,2,3]],self.corners+np.array([0,0,.01]),np.full((4,3),np.nan)]:
            with self.subTest(corners=corners):self.assertEqual(self.call(corners_camera_m=corners)['status'],'unknown')
    def test_invalid_calibration_and_plane_rejected(self):
        for k in [np.zeros((3,3)),self.k+np.diag([np.nan,0,0])]:self.assertEqual(self.call(intrinsic=k)['status'],'unknown')
        c=dict(self.candidate,normal_camera=[0,0,-2]);self.assertEqual(self.call(candidate=c)['status'],'unknown')
    def test_invalid_settings_rejected(self):
        for erosion in [-1,True,1.5,100]:self.assertEqual(self.call(erosion_pixels=erosion)['status'],'unknown')
    def test_clock_nan_rejected(self):
        self.assertEqual(self.call(now_s=float('nan'))['status'],'unknown')
    def test_reversed_winding_valid(self):
        self.assertEqual(self.call(corners_camera_m=self.corners[::-1])['status'],'observed_footprint')

    def test_rotated_footprint_does_not_require_untouched_bbox_pixels(self):
        corners=np.array([[0,-.2,1],[.2,0,1],[0,.2,1],[-.2,0,1]])
        self.depth[2,2]=np.nan
        self.assertEqual(self.call(corners_camera_m=corners)['status'],'observed_footprint')
        self.depth[4,4]=np.nan
        self.assertEqual(self.call(corners_camera_m=corners)['status'],'unknown')
    def test_exact_boundary_tangency_is_unknown_on_every_edge(self):
        for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
            corners=self.corners.copy();corners[:,axis]+=sign*.35
            self.assertEqual(self.call(corners_camera_m=corners)['reason'],'footprint_outside_image')
            corners[:,axis]-=sign*.0001
            self.assertEqual(self.call(corners_camera_m=corners)['status'],'observed_footprint')
    def test_tilted_plane_projects_depth_consistently(self):
        v,u=np.indices(self.depth.shape)
        self.depth=1/(1-.2*(u-4)/10)
        normal=np.array([.2,0,-1]);scale=np.linalg.norm(normal)
        self.candidate.update(normal_camera=(normal/scale).tolist(),offset_m=1/scale)
        self.corners[:,2]=1+.2*self.corners[:,0]
        self.assertEqual(self.call()['status'],'observed_footprint')
        self.depth[4,4]+=.02
        self.assertEqual(self.call()['status'],'unknown')
