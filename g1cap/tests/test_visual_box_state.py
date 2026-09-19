import unittest
try:
    import numpy as np
except ImportError:
    np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class VisualBoxStateTests(unittest.TestCase):
    def make(self):
        from g1cap.visual_box_state import BoxEstimateStream
        x=BoxEstimateStream()
        x.initialize(time_s=1.,now_s=1.,center=[0,0,.7],rotation=np.eye(3),size=[.2,.2,.2])
        return x

    def quality(self,**changes):
        return dict(status='full_rank_candidate',constraint_rank=6,rms_m=.001,
                    face_rows=500,total_face_points=500,**changes)

    def submit(self,x,time=1.1,now=None,quality=None,center=None,size=None):
        return x.submit(time_s=time,now_s=time if now is None else now,
            center=[0,0,.7] if center is None else center,rotation=np.eye(3),
            size=[.2,.2,.2] if size is None else size,
            quality=self.quality() if quality is None else quality)

    def test_age_query_and_returned_data_are_not_stale_measurements(self):
        x=self.make();a=x.observe(1.05);self.assertEqual(a['status'],'accepted')
        a['center_camera_m'][0]=99
        self.assertEqual(x.observe(1.05)['center_camera_m'][0],0)
        old=x.observe(1.16);self.assertEqual(old['reason'],'stale')
        self.assertIsNone(old['center_camera_m'])

    def test_future_duplicate_and_out_of_order_frames_are_rejected(self):
        x=self.make()
        self.assertEqual(self.submit(x,time=1.1,now=1.05)['reason'],'future_frame')
        self.assertEqual(self.submit(x,time=1.)['reason'],'nonincreasing_frame_time')
        self.assertEqual(self.submit(x,time=.9,now=1.)['reason'],'nonincreasing_frame_time')
        self.assertEqual(self.submit(x)['status'],'accepted')

    def test_weak_fit_is_unavailable_but_short_recovery_can_use_prior(self):
        x=self.make();q=self.quality();q['constraint_rank']=3;q['status']='partially_constrained'
        result=self.submit(x,quality=q)
        self.assertEqual(result['reason'],'weak_geometry');self.assertIsNone(result['center_camera_m'])
        self.assertIsNotNone(x.prediction_seed(1.2))
        self.assertEqual(self.submit(x,time=1.2)['status'],'accepted')

    def test_large_residual_low_coverage_and_shape_change_are_rejected(self):
        for field,value,reason in [('rms_m',.02,'large_fit_residual'),('face_rows',100,'low_correspondence_coverage')]:
            x=self.make();q=self.quality();q[field]=value
            self.assertEqual(self.submit(x,quality=q)['reason'],reason)
        self.assertEqual(self.submit(self.make(),size=[.3,.2,.2])['reason'],'dimensions_changed')
        self.assertEqual(self.submit(self.make(),center=[.5,0,.7])['reason'],'implausible_motion')

    def test_long_gap_requires_explicit_new_initialization(self):
        x=self.make();r=self.submit(x,time=1.4)
        self.assertEqual(r['reason'],'reinitialization_required')
        self.assertIsNone(x.prediction_seed(1.4))
        x.initialize(time_s=1.5,now_s=1.5,center=[0,0,.7],rotation=np.eye(3),size=[.2,.2,.2])
        self.assertEqual(x.observe(1.5)['track_epoch'],2)

    def test_invalid_time_and_reflected_pose_cannot_initialize(self):
        from g1cap.visual_box_state import BoxEstimateStream
        for time,r in [(float('nan'),np.eye(3)),(1.,np.diag([-1,1,1]))]:
            with self.assertRaises(ValueError):
                BoxEstimateStream().initialize(time_s=time,now_s=1.,center=[0,0,.7],rotation=r,size=[.2,.2,.2])

if __name__=='__main__':unittest.main()
