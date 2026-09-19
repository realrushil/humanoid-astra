"""Behavioral checks for fresh partial-front association; no simulator inputs."""
import unittest
try:
    import numpy as np
    from g1cap.observed_approach import ApproachEstimate
except ImportError:
    np=None


def associate(candidates,estimate,time_s):
    return estimate.select_front(candidates,time_s)


def edge(span=.2, offset=-.5, angle=0.):
    return dict(observed_span_m=span, normal_body=[np.cos(angle),np.sin(angle),0.],
                offset_body_m=offset)


@unittest.skipIf(np is None, 'NumPy is required')
class FrontTrackingTests(unittest.TestCase):
    def prior(self):
        estimate=ApproachEstimate()
        for t,offset in [(0.,-.52),(.1,-.5)]:
            estimate.advance_imu(t,[0,0,1],[[0,0,0]])
            estimate.observe(t,offset,[1,0,0],[0,0,1])
        estimate.advance_imu(.2,[0,0,1],[[0,0,0]])
        return estimate

    def test_initialization_requires_one_long_segment(self):
        e=ApproachEstimate();e.advance_imu(0.,[0,0,0],[[0,0,0]])
        self.assertEqual(associate([edge(.11)],e,0.)['status'],'unavailable')
        self.assertEqual(associate([edge(),edge(offset=-.7)],e,0.)['status'],'unavailable')
        result=associate([edge(),edge(.1,offset=-.7)],e,0.)
        self.assertEqual(result['mode'],'initialize')
        self.assertEqual(result['candidate']['observed_span_m'],.2)

    def test_short_fresh_segment_follows_gyro_and_measured_offset_speed(self):
        result=associate([edge(.11,-.48,-.1)],self.prior(),.2)
        self.assertEqual(result['mode'],'track')
        self.assertEqual(result['candidate'],edge(.11,-.48,-.1))

    def test_wrong_heading_offset_and_too_short_are_rejected(self):
        for candidate in [edge(.11,-.48,.1),edge(.11,-.44,-.1),edge(.06,-.48,-.1),edge(.3,-.8,-.1)]:
            with self.subTest(candidate=candidate):
                self.assertEqual(associate([candidate],self.prior(),.2)['status'],'unavailable')

    def test_multiple_matching_fresh_edges_are_ambiguous(self):
        result=associate([edge(.11,-.48,-.1),edge(.2,-.49,-.1)],self.prior(),.2)
        self.assertEqual(result['reason'],'ambiguous_tracked_front')

    def test_no_measurement_cannot_be_replaced_by_prior(self):
        self.assertEqual(associate([],self.prior(),.2)['status'],'unavailable')

    def test_expired_or_unsynchronized_prior_cannot_admit_short_line(self):
        e=self.prior();e.advance_imu(.3,[0,0,1],[[0,0,0]])
        self.assertEqual(associate([edge(.11,-.46,-.2)],e,.3)['status'],'unavailable')
        self.assertEqual(associate([edge(.11,-.48,-.1)],self.prior(),.21)['status'],'unavailable')

    def test_invalidation_requires_strict_reinitialization(self):
        e=self.prior();e.invalidate('front_not_visible')
        self.assertEqual(associate([edge(.11,-.48,-.1)],e,.2)['status'],'unavailable')
        self.assertEqual(associate([edge(.2,-.48,-.1)],e,.2)['mode'],'initialize')

    def test_production_loss_or_ambiguity_requires_two_strict_recovery_samples(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from g1cap.observed_approach import observe_front
        # Isolate the association boundary with fresh extracted segments. The
        # real fitter is covered separately by synthetic depth and full replays.
        geometry=SimpleNamespace(exclude=lambda *args:np.zeros(0,bool),
            front_clearance=lambda line,*args:(np.array(line['normal_camera']),line['offset_m'],.5))
        def segment(span=.2):
            return dict(observed_span_m=span,normal_camera=[1,0,0],offset_m=-.5)
        for loss in ([],[segment(.11),segment(.12)]):
            with self.subTest(candidates=loss):
                e=ApproachEstimate()
                def sample(step,lines):
                    t=step*.1;e.advance_imu(t,[0,0,0],[[0,0,0]])
                    packet=dict(step=step,time_s=t)
                    camera=dict(packet,rgb=np.zeros((1,1,3)),depth_m=np.ones((1,1)),calibration=dict(intrinsic=np.eye(3)))
                    with patch('g1cap.rgbd_table.front_candidates',return_value=dict(status='observed_candidates',lines=lines,planar_points=30)):
                        return observe_front(geometry,e,packet,camera,None,[0,0,1])
                sample(0,[segment()]);sample(1,[segment(.11)])
                self.assertEqual(e.feedback(.1)['status'],'available')
                self.assertEqual(sample(2,loss)['status'],'unavailable')
                self.assertIsNone(e.previous)
                self.assertEqual(e.feedback(.2)['status'],'unavailable')
                self.assertEqual(sample(3,[segment(.11)])['status'],'unavailable')
                self.assertEqual(sample(4,[segment()])['association'],'initialize')
                self.assertEqual(e.feedback(.4)['status'],'unavailable')
                sample(5,[segment()])
                self.assertEqual(e.feedback(.5)['status'],'available')
                self.assertEqual(e.feedback(.5)['observed_at_s'],.5)


if __name__=='__main__':unittest.main()
