import importlib
import unittest

try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None, 'NumPy is required')
class ObservedApproachTests(unittest.TestCase):
    def guard(self):
        try:
            return importlib.import_module('g1cap.observed_approach').ApproachEstimate()
        except ModuleNotFoundError:
            self.fail('sensor approach estimate is missing')

    def sample(self, guard, t, clearance=.2, normal=(1.,0.,0.), gyro=(0.,0.,0.)):
        guard.advance_imu(t, gyro, [[0.,0.,0.]])
        guard.observe(t, -clearance, normal, [0.,0.,1.])

    def test_closing_speed_age_and_retreat(self):
        guard=self.guard()
        self.sample(guard,0.,.22)
        self.assertEqual(guard.feedback(0.)['status'],'unavailable')
        self.sample(guard,.1,.20)
        guard.advance_imu(.15,[0.,0.,0.],[[0,0,0]])
        action,info=guard.limit([.6,.1,.2],.15)
        # 0.2 m/s measured closing, 50 ms camera age plus 100 ms lookahead.
        self.assertAlmostEqual(action[0],(.2-.03-.15*.2)/.5)
        self.assertAlmostEqual(info['age_s'],.05)
        retreat,_=guard.limit([-.4,0.,0.],.15)
        np.testing.assert_allclose(retreat,[-.4,0.,0.])

    def test_normal_is_rotated_to_current_body_not_relabelled_fresh(self):
        guard=self.guard()
        self.sample(guard,0.,gyro=(0.,0.,2.))
        self.sample(guard,.1,gyro=(0.,0.,2.))
        guard.advance_imu(.15,[0.,0.,2.],[[0,0,0]])
        info=guard.feedback(.15)
        np.testing.assert_allclose(info['normal_navigation_xy'],[np.cos(.1),-np.sin(.1)],atol=1e-12)
        self.assertEqual(info['observed_at_s'],.1)

    def test_lost_edge_and_time_gap_require_two_new_measurements(self):
        guard=self.guard();self.sample(guard,0.);self.sample(guard,.1)
        guard.invalidate('ambiguous_front')
        with self.assertRaisesRegex(ValueError,'sensor_approach_unavailable'):
            guard.limit([.1,0.,0.],.1)
        self.sample(guard,.2)
        self.assertEqual(guard.feedback(.2)['status'],'unavailable')
        self.sample(guard,.3)
        self.assertEqual(guard.feedback(.3)['status'],'available')
        guard.advance_imu(.5,[0.,0.,0.],[[0,0,0]])
        self.assertEqual(guard.feedback(.5)['status'],'unavailable')

    def test_expired_future_and_unsynchronized_state_are_not_used(self):
        guard=self.guard();self.sample(guard,0.);self.sample(guard,.1)
        self.assertEqual(guard.feedback(.05)['status'],'unavailable')
        self.assertEqual(guard.feedback(.11)['status'],'unavailable')
        guard.advance_imu(.2,[0.,0.,0.],[[0,0,0]]);guard.advance_imu(.26,[0.,0.,0.],[[0,0,0]])
        self.assertEqual(guard.feedback(.26)['status'],'unavailable')
        with self.assertRaises(ValueError):guard.observe(.25,.1,[1,0,0],[0,0,1])

    def test_current_leg_extension_changes_clearance_before_next_camera(self):
        guard=self.guard();self.sample(guard,0.,.2);self.sample(guard,.1,.2)
        guard.advance_imu(.15,[0,0,0],[[.04,0,0],[0,0,0]])
        info=guard.feedback(.15)
        self.assertAlmostEqual(info['clearance_m'],.16)
        self.assertAlmostEqual(info['observed_clearance_m'],.2)
        self.assertAlmostEqual(info['origin_closing_speed_m_s'],0.)

    def test_tilted_body_uses_horizontal_navigation_axes(self):
        guard=self.guard();angle=.25
        up=np.array([-np.sin(angle),0.,np.cos(angle)])
        normal=np.array([np.cos(angle),0.,np.sin(angle)])
        for t in [0.,.1]:
            guard.advance_imu(t,[0,0,0],[[0,0,0]]);guard.observe(t,-.08,normal,up)
        np.testing.assert_allclose(guard.feedback(.1)['normal_navigation_xy'],[1,0],atol=1e-12)
        with self.assertRaises(ValueError):guard.observe(.1,float('nan'),normal,up)


@unittest.skipIf(np is None, 'NumPy is required')
class TableBoundaryTests(unittest.TestCase):
    def test_depth_front_varies_with_geometry_and_abstains_without_support(self):
        try:module=importlib.import_module('g1cap.rgbd_table')
        except ModuleNotFoundError:self.fail('observed table boundary is missing')
        v,u=np.indices((120,160));k=np.array([[120,0,80],[0,120,20],[0,0,1]])
        x=(u-80)/120;y=(v-20)/120
        for height,front in [(.4,1.),(.3,.8),(.5,1.2)]:
            with self.subTest(height=height,front=front):
                z=np.divide(height,y,out=np.full(y.shape,np.nan),where=y>0)
                top=(z>=front)&(z<=front+.6)&(abs(x*z)<.45)
                floor=np.divide(1.,y,out=np.full(y.shape,np.nan),where=y>0)
                depth=np.where(top,z,floor)
                support=dict(status='observed_candidate',normal_camera=[0,-1,0],offset_m=height)
                exclude=lambda points:np.zeros(len(points),bool)
                result=module.front_boundary(depth,k,support,np.ones(z.shape,bool),exclude)
                self.assertEqual(result['status'],'observed_candidate',result)
                line=result['line']
                self.assertLess(abs(-line['offset_m']-front),.025)
                self.assertGreater(line['observed_span_m'],.5)
                missing=module.front_boundary(depth,k,None,np.ones(z.shape,bool),exclude)
                self.assertEqual(missing['status'],'unavailable')
                masked=module.front_boundary(depth,k,support,np.ones(z.shape,bool),lambda p:np.ones(len(p),bool))
                self.assertEqual(masked['status'],'unavailable')
