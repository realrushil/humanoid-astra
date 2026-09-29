import unittest
import numpy as np
from g1cap import destination_surface as surface

class DestinationTopTests(unittest.TestCase):
    def scene(self):
        yy,xx=np.indices((80,100));mask=np.zeros((80,100),bool);mask[26:60,10:90]=True
        rgb=np.zeros((80,100,3),np.uint8);rgb[mask]=[50,150,60]
        k=np.array([[100.,0,50],[0,100.,20],[0,0,1.]])
        depth=np.full(mask.shape,np.nan)
        top=mask&(yy<40);side=mask&~top
        depth[top]=.2/((yy[top]-20)/100);depth[side]=1.
        return rgb,depth,k,top
    def observe(self,rgb,depth,k,up=(0.,-1.,0.)):
        return surface.observe_level_destination_surface(2.,rgb,depth,k,up)
    def test_level_top_selected_without_fascia_extending_footprint(self):
        rgb,d,k,top=self.scene();old=surface.observe_destination_surface(2.,rgb,surface.green_destination_mask(rgb),d,k)
        self.assertEqual(old['status'],'unavailable')
        r=self.observe(rgb,d,k)
        self.assertEqual(r['status'],'observed_destination_candidate')
        self.assertFalse(r['complete_footprint_observed'])
        self.assertEqual(r['extent_kind'],'visible_plane_patch_only')
        self.assertAlmostEqual(abs(r['normal_camera'][1]),1.,places=5)
        self.assertEqual(r['selection'],'unique_observed_level_color_plane')
    def test_two_level_surfaces_rejected(self):
        rgb,d,k,top=self.scene();yy,xx=np.indices(d.shape)
        second=top&(xx>=50);d[second]=.3/((yy[second]-20)/100)
        self.assertEqual(self.observe(rgb,d,k)['status'],'unavailable')
    def test_vertical_only_rejected(self):
        rgb,d,k,top=self.scene();d[np.isfinite(d)]=1.
        self.assertEqual(self.observe(rgb,d,k)['status'],'unavailable')
    def test_plane_budget_cannot_hide_second_level_candidate(self):
        y,x=np.indices((100,200));rgb=np.full((100,200,3),[50,150,60],np.uint8)
        k=np.array([[100.,0.,100.],[0.,100.,-20.],[0.,0.,1.]])
        depth=np.zeros((100,200));depth[x<50]=4.
        depth[(x>=50)&(x<100)]=5.;depth[(x>=100)&(x<150)]=6.
        first=(x>=150)&(y<60);second=(x>=150)&(y>=60)
        depth[first]=.2/((y[first]+20)/100)
        depth[second]=.6/((y[second]+20)/100)
        self.assertEqual(self.observe(rgb,depth,k)['status'],'unavailable')
    def test_missing_depth_cannot_be_hidden_by_selection(self):
        rgb,d,k,top=self.scene();d[:,::2]=np.nan
        self.assertEqual(self.observe(rgb,d,k)['status'],'unavailable')
    def test_multiple_color_objects_rejected(self):
        rgb,d,k,top=self.scene();rgb[3:8,3:8]=[50,150,60];d[3:8,3:8]=1.
        self.assertEqual(self.observe(rgb,d,k)['status'],'unavailable')
    def test_invalid_up_rejected(self):
        rgb,d,k,top=self.scene()
        for up in ([0,0,0],[0,-2,0],[0,float('nan'),0],[1,0]):
            with self.subTest(up=up),self.assertRaises(ValueError):self.observe(rgb,d,k,up)
    def test_no_color_is_unavailable(self):
        rgb,d,k,top=self.scene();rgb[:]=100
        self.assertEqual(self.observe(rgb,d,k)['status'],'unavailable')
    def test_cached_observation_uses_camera_time_up(self):
        rgb,d,k,_=self.scene();camera=dict(step=4,time_s=2.,rgb=rgb,depth_m=d,calibration={'intrinsic':k})
        observer=surface.DestinationObservation()
        observer.update(camera,dict(step=4,time_s=2.),np.eye(4),None,up_body=[0.,-1.,0.])
        result=observer.observe(2.)
        self.assertEqual(result['status'],'observed_destination_candidate')
        self.assertEqual(result['direction_status'],'unavailable')
        self.assertEqual(result['selection'],'unique_observed_level_color_plane')
    def test_mixed_time_up_cannot_select_a_surface(self):
        rgb,d,k,_=self.scene();camera=dict(step=4,time_s=2.,rgb=rgb,depth_m=d,calibration={'intrinsic':k})
        observer=surface.DestinationObservation()
        observer.update(camera,dict(step=5,time_s=2.02),np.eye(4),None,up_body=[0.,-1.,0.])
        self.assertEqual(observer.observe(2.02)['status'],'unavailable')
