import unittest
try:
    import numpy as np
except ImportError:
    np=None
if np is not None:
    from g1cap.visual_retention import VisualRetention

@unittest.skipIf(np is None,"NumPy runtime required")
class RetentionTests(unittest.TestCase):
    def sample(self,t):
        return dict(time_s=t,box=dict(status='accepted',observed_at_s=t,track_epoch=1,
            center_camera_m=[0,0,0],axes_camera=np.eye(3).tolist(),dimensions_m=[.2,.2,.2]),
            relative=dict(status='accepted',observed_at_s=t,track_epoch=1,
                box_center_pelvis_m=[0,0,0],box_axes_pelvis=np.eye(3).tolist(),
                wrist_positions_pelvis_m=dict(left=[.13,.2,0],right=[-.13,.2,0]),
                box_center_wrist_m=dict(left=[-.13,-.2,0],right=[.13,-.2,0])),
            motion=dict(status='tracked_local_segment',time_s=t,segment=1,body_in_segment=np.eye(4).tolist()),
            camera_transform=np.eye(4).tolist(),up_body=[0,0,1])
    def settled(self):
        observer=VisualRetention()
        for i in range(26):result=observer.update(**self.sample(i*.04))
        return observer,result
    def test_missing_height_does_not_mean_missing_retention_or_prove_pickup(self):
        _,result=self.settled()
        self.assertTrue(result['retained']);self.assertTrue(result['settled'])
        self.assertNotIn('raised',result);self.assertNotIn('gap_m',result)
        self.assertIs(result['pickup_proven'],False)
    def test_same_side_wrists_reject(self):
        x,_=self.settled();s=self.sample(1.04)
        s['relative']['wrist_positions_pelvis_m']['right']=[.12,.2,0]
        r=x.update(**s);self.assertFalse(r['retained']);self.assertFalse(r['settled'])
    def test_box_missing_clears_history(self):
        x,_=self.settled();s=self.sample(1.04);s['box']['status']='unavailable'
        self.assertEqual(x.update(**s)['status'],'unavailable')
        self.assertFalse(x.update(**self.sample(1.08))['settled'])
    def test_wrong_measurement_time_clears_history(self):
        x,_=self.settled();s=self.sample(1.04);s['relative']['observed_at_s']=1.
        self.assertEqual(x.update(**s)['status'],'unavailable')
        self.assertFalse(x.update(**self.sample(1.08))['settled'])
    def test_duplicate_or_dropout_does_not_preserve_settled(self):
        for stamp in (1.,.96,1.20):
            x,_=self.settled();r=x.update(**self.sample(stamp))
            self.assertFalse(r['settled'])
    def test_identity_change_resets_window(self):
        for key in ('box','motion'):
            x,_=self.settled();s=self.sample(1.04)
            if key=='box':s['box']['track_epoch']=s['relative']['track_epoch']=2
            else:s['motion']['segment']=2
            self.assertFalse(x.update(**s)['settled'])
    def test_wrist_relative_motion_rejects_settling(self):
        x=VisualRetention()
        for i in range(26):
            s=self.sample(i*.04);s['relative']['box_center_wrist_m']['left'][0]+=.06*i*.04
            r=x.update(**s)
        self.assertTrue(r['retained']);self.assertFalse(r['settled'])
    def test_scene_motion_rejects_settling_even_with_stationary_wrists(self):
        x=VisualRetention()
        for i in range(26):
            s=self.sample(i*.04);s['motion']['body_in_segment'][0][3]=.06*i*.04
            r=x.update(**s)
        self.assertTrue(r['retained']);self.assertFalse(r['settled'])
    def test_bad_geometry_is_unavailable(self):
        for value in (float('nan'),float('inf')):
            x,_=self.settled();s=self.sample(1.04);s['relative']['box_center_wrist_m']['left'][0]=value
            self.assertEqual(x.update(**s)['status'],'unavailable')

if __name__=='__main__':unittest.main()
