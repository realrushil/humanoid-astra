import unittest
try:
    import numpy as np
except ImportError:
    np=None

@unittest.skipIf(np is None,'NumPy unavailable')
class VisualGraspTests(unittest.TestCase):
    def test_settled_retention_does_not_require_raised_clearance(self):
        from g1cap.visual_grasp import VisualGraspWindow
        x=VisualGraspWindow()
        for i in range(11):r=x.add(self.sample(i*.1,gap=.052))
        self.assertFalse(r['ready'])
        self.assertTrue(r.get('retention_stable',False))
        bad=self.sample(1.1,gap=.052);bad['wrist_positions_pelvis_m']['right']=[.12,.2,0]
        self.assertFalse(x.add(bad).get('retention_stable',True))
    def sample(self,t,gap=.08):
        return dict(time_s=t,gap_m=gap,dimensions_m=[.2,.2,.2],box_center_pelvis_m=[0,0,0],
            box_axes_pelvis=np.eye(3).tolist(),wrist_positions_pelvis_m={'left':[.13,.2,0],'right':[-.13,.2,0]},
            box_center_wrist_m={'left':[-.13,-.2,0],'right':[.13,-.2,0]},up_body=[0,0,1])
    def test_table_rest_is_not_pickup_and_stable_raised_window_is(self):
        from g1cap.visual_grasp import VisualGraspWindow
        x=VisualGraspWindow()
        for i in range(11):r=x.add(self.sample(i*.1,gap=0))
        self.assertFalse(r['raised']);self.assertFalse(r['ready'])
        for i in range(11,22):r=x.add(self.sample(i*.1))
        self.assertTrue(r['raised']);self.assertTrue(r['ready'])
    def test_drift_or_same_side_wrists_are_not_stable_grasp(self):
        from g1cap.visual_grasp import VisualGraspWindow
        x=VisualGraspWindow()
        for i in range(11):
            s=self.sample(i*.1);s['box_center_wrist_m']['left'][0]+=.01*i;r=x.add(s)
        self.assertFalse(r['ready'])
        s=self.sample(1.1);s['wrist_positions_pelvis_m']['right']=[.12,.2,0]
        self.assertFalse(x.add(s)['raised'])
    def test_missing_time_resets_stability_window(self):
        from g1cap.visual_grasp import VisualGraspWindow
        x=VisualGraspWindow()
        for i in range(11):r=x.add(self.sample(i*.1))
        self.assertTrue(r['ready'])
        self.assertFalse(x.add(self.sample(1.5))['ready'])
    def test_elapsed_time_at_camera_rates_and_mixed_cadence(self):
        from g1cap.visual_grasp import VisualGraspWindow
        for hz in (10,25,50):
            x=VisualGraspWindow()
            for i in range(hz):r=x.add(self.sample(i/hz))
            self.assertFalse(r['ready'],hz)
            self.assertTrue(x.add(self.sample(1.))['ready'],hz)
        x=VisualGraspWindow()
        for t in (0,.1,.2,.24,.28,.32,.36,.4,.44,.48,.52,.56,.6,.64,.68,.72,.76,.8,.84,.88,.92,.96,1.):
            r=x.add(self.sample(t))
        self.assertTrue(r['ready'])
        self.assertGreaterEqual(len(x.samples),2)
    def test_fast_camera_drift_and_missing_frame_reject(self):
        from g1cap.visual_grasp import VisualGraspWindow
        x=VisualGraspWindow()
        for i in range(51):
            s=self.sample(i*.02)
            s['box_center_wrist_m']['left'][0]+=.1*i*.02
            r=x.add(s)
        self.assertFalse(r['ready'])
        x=VisualGraspWindow()
        for i in range(51):r=x.add(self.sample(i*.02))
        self.assertTrue(r['ready'])
        self.assertFalse(x.add(self.sample(1.2))['ready'])
