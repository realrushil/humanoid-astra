import importlib
import unittest
try:import numpy as np
except ImportError:np=None

@unittest.skipIf(np is None,'NumPy required')
class ObservedHandTests(unittest.TestCase):
    def module(self):
        try:return importlib.import_module('g1cap.observed_hand')
        except ModuleNotFoundError:self.fail('sensor-derived hand clearance is missing')

    def shape(self,x=.2,z=.1):
        import itertools
        return dict(name='hand',points=np.array(list(itertools.product([x-.02,x+.02],[-.02,.02],[z-.01,z+.01]))))

    def sample(self,g,t,shape=None,top_offset=0.,front_offset=0.):
        g.advance(t,[0,0,0],[shape or self.shape()])
        g.observe(t,[0,0,1],top_offset,[1,0,0],front_offset)

    def test_clip_front_does_not_count_low_points_outside_table(self):
        m=self.module()
        points=np.array([[-.1,0,-.1],[-.1,.1,-.1],[.1,0,.1],[.1,.1,.1]])
        self.assertAlmostEqual(m.clipped_gap(points,[0,0,1],0.,[1,0,0],0.),-.03)
        self.assertIsNone(m.clipped_gap(self.shape(x=-.2)['points'],[0,0,1],0.,[1,0,0],0.))

    def test_current_finger_motion_updates_gap_between_images(self):
        g=self.module().HandClearanceEstimate()
        self.sample(g,0.);self.sample(g,.1)
        g.advance(.12,[0,0,0],[self.shape(z=.06)])
        f=g.feedback(.12)
        self.assertEqual(f['status'],'available');self.assertAlmostEqual(f['margin_m'],.05)
        self.assertAlmostEqual(f['age_s'],.02);self.assertEqual(f['observed_at_s'],.1)

    def test_plane_translation_prediction_only_credits_closing(self):
        g=self.module().HandClearanceEstimate()
        self.sample(g,0.,top_offset=.02);self.sample(g,.1,top_offset=0.)
        g.advance(.15,[0,0,0],[self.shape()]);self.assertAlmostEqual(g.feedback(.15)['margin_m'],.08)
        h=self.module().HandClearanceEstimate()
        self.sample(h,0.,top_offset=-.02);self.sample(h,.1,top_offset=0.)
        h.advance(.15,[0,0,0],[self.shape()]);self.assertAlmostEqual(h.feedback(.15)['margin_m'],.09)

    def test_loss_staleness_and_gap_require_fresh_history(self):
        g=self.module().HandClearanceEstimate();self.sample(g,0.)
        self.assertEqual(g.feedback(0.)['status'],'unavailable')
        self.sample(g,.1);g.invalidate('missing_support')
        self.sample(g,.2);self.assertEqual(g.feedback(.2)['status'],'unavailable')
        self.sample(g,.3);self.assertEqual(g.feedback(.3)['status'],'available')
        self.assertEqual(g.feedback(.31)['status'],'unavailable')
        g.advance(.4,[0,0,0],[self.shape()]);g.advance(.46,[0,0,0],[self.shape()])
        self.assertEqual(g.feedback(.46)['status'],'unavailable')
        g.advance(.6,[0,0,0],[self.shape()]);self.assertEqual(g.feedback(.6)['status'],'unavailable')

    def test_gyro_rotates_observed_normal_without_freshening_image(self):
        g=self.module().HandClearanceEstimate()
        for t in [0.,.1]:
            g.advance(t,[1,0,0],[self.shape()]);g.observe(t,[0,0,1],0.,[1,0,0],0.)
        g.advance(.15,[1,0,0],[self.shape()]);f=g.feedback(.15)
        np.testing.assert_allclose(f['up_body'],[0,np.sin(.05),np.cos(.05)],atol=1e-12)
        self.assertEqual(f['observed_at_s'],.1)
        with self.assertRaises(ValueError):g.observe(.14,[0,0,1],0.,[1,0,0],0.)
