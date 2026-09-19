"""Metric geometry must follow visible samples, not a nominal box definition."""
import importlib
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'numpy is required for depth geometry')
class RGBDGeometryTests(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('g1cap.rgbd_geometry')
        except ModuleNotFoundError:
            self.fail('RGB-D geometry implementation is missing')

    def test_depth_projection_keeps_invalid_pixels_unknown(self):
        m = self.module()
        z = np.array([[2., np.nan], [0., 4.]])
        p = m.depth_points(z, [[2,0,0],[0,2,0],[0,0,1]])
        np.testing.assert_allclose(p[0,0], [0,0,2])
        np.testing.assert_allclose(p[1,1], [2,2,4])
        self.assertTrue(np.isnan(p[0,1]).all())
        self.assertTrue(np.isnan(p[1,0]).all())
        with self.assertRaises(ValueError):
            m.depth_points(z, [[2,1,0],[0,2,0],[0,0,1]])

    def test_plane_fitting_is_metric_under_rotations_sizes_and_outliers(self):
        m = self.module()
        rng = np.random.default_rng(4)
        for width, depth, yaw in [(.14,.31,.4), (.29,.18,-.9)]:
            with self.subTest(width=width, depth=depth, yaw=yaw):
                n = np.array([np.sin(yaw), 0., np.cos(yaw)])
                x = np.array([np.cos(yaw), 0., -np.sin(yaw)])
                center = np.array([.12,-.25,.85])
                uv = rng.uniform(-.5,.5,(1500,2))*[width,depth]
                p = center + uv[:,0,None]*x + uv[:,1,None]*[0,1,0]
                p += rng.normal(0,.0003,p.shape)
                outliers = rng.uniform(-1,1,(150,3))
                planes = m.fit_planes(np.vstack((p,outliers)), max_planes=1)
                self.assertEqual(len(planes), 1)
                fit = planes[0]
                self.assertGreater(abs(np.dot(fit['normal'],n)), .999)
                self.assertLess(abs(np.dot(fit['normal'],center)+fit['offset_m']), .001)
                self.assertGreater(len(fit['points']), 1450)
                self.assertLess(fit['rms_m'], .001)

    def test_plane_fit_rejects_missing_and_collinear_geometry(self):
        m = self.module()
        self.assertEqual(m.fit_planes(np.empty((0,3))), [])
        line = np.arange(200)[:,None]*np.array([[.01,.02,.03]])
        self.assertEqual(m.fit_planes(line), [])
        self.assertEqual(m.fit_planes(np.full((200,3),np.nan)), [])
        with self.assertRaises(ValueError):
            m.fit_planes([[0,1]])

    def test_two_observed_planes_are_separate_and_replay_deterministic(self):
        m = self.module()
        a,b = np.meshgrid(np.linspace(-.1,.1,30),np.linspace(-.15,.15,30))
        front = np.column_stack((a.ravel(),b.ravel(),np.full(a.size,.7)))
        top = np.column_stack((a.ravel(),np.full(a.size,-.15),b.ravel()+.85))
        p = np.concatenate((front,top))
        fits = m.fit_planes(p,max_planes=2)
        self.assertEqual(len(fits),2)
        self.assertLess(abs(np.dot(fits[0]['normal'],fits[1]['normal'])),.01)
        for first, again in zip(fits,m.fit_planes(p,max_planes=2)):
            np.testing.assert_array_equal(first['normal'],again['normal'])
            np.testing.assert_array_equal(first['points'],again['points'])
