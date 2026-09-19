"""Stationary command lifecycle regressions; no robot/controller is started."""
import unittest
from g1cap.toolkit.sonic_backend import LeasedPlanner


class StationaryLeaseTests(unittest.TestCase):
    def test_stationary_reference_expires_without_leaving_arm_override(self):
        planner = LeasedPlanner(0., 0.)
        self.assertTrue(hasattr(planner, 'stationary'), 'stationary command path is missing')
        planner.stationary(4, .7, [0.] * 17, [0.] * 17, .01)
        active = planner.fields(0., .1)
        self.assertEqual(active['mode'], 4)
        self.assertEqual(active['upper_body_position'], [0.] * 17)
        expired = planner.fields(0., .3)
        self.assertEqual(expired['mode'], 0)
        self.assertNotIn('upper_body_position', expired)
        self.assertEqual(expired['height'], -1.)

    def test_new_velocity_or_explicit_idle_clears_stationary_reference(self):
        planner = LeasedPlanner(0., 0.)
        self.assertTrue(hasattr(planner, 'stationary'))
        planner.stationary(4, .7, [0.] * 17, None, .01)
        planner.command(.1, 0., 0., .02)
        moving = planner.fields(0., .03)
        self.assertEqual(moving['mode'], 1)
        self.assertNotIn('upper_body_position', moving)
        planner.stationary(0, -1., [0.] * 17, None, .04)
        planner.idle(.05)
        self.assertNotIn('upper_body_position', planner.fields(0., .06))

    def test_invalid_reference_does_not_replace_valid_command(self):
        planner = LeasedPlanner(0., 0.)
        self.assertTrue(hasattr(planner, 'stationary'))
        planner.stationary(4, .7, None, None, .01)
        for mode, height, q in [(1, .7, None), (4, .1, None),
                                 (0, -1., [0.] * 14), (0, -1., [float('nan')] * 17)]:
            with self.subTest(mode=mode, height=height, q=q):
                with self.assertRaises(ValueError):
                    planner.stationary(mode, height, q, None, .02)
        self.assertEqual(planner.fields(0., .03)['mode'], 4)

class ContactEvidenceTests(unittest.TestCase):
    def test_nonfoot_support_is_not_counted_as_supported_standing(self):
        from g1cap import sonic_sim
        self.assertTrue(hasattr(sonic_sim, 'summarize_contacts'))
        rows = [dict(body1='world', body2='left_ankle_roll_link', distance=-.001, normal_force=80.),
                dict(body1='right_wrist_yaw_link', body2='world', distance=-.005, normal_force=25.)]
        result = sonic_sim.summarize_contacts(rows)
        self.assertEqual(result['foot_normal_forces'], {'left': 80., 'right': 0.})
        self.assertEqual(len(result['forbidden_contacts']), 1)

    def test_light_touch_does_not_establish_load_bearing_support(self):
        from g1cap import sonic_sim
        self.assertTrue(hasattr(sonic_sim, 'summarize_contacts'))
        result = sonic_sim.summarize_contacts([
            dict(body1='world', body2='right_ankle_roll_link', distance=0., normal_force=.1)])
        self.assertLess(result['foot_normal_forces']['right'], 5.)

class SelfContactTests(unittest.TestCase):
    def test_shallow_self_touch_is_recorded_separately_from_external_support(self):
        from g1cap.sonic_sim import summarize_contacts
        row = dict(body1='right_hip_roll_link', body2='right_hand_thumb_2_link',
                   distance=-.0003, normal_force=6., self_contact=True)
        result = summarize_contacts([row])
        self.assertEqual(result['forbidden_contacts'], [])
        self.assertEqual(result['self_contacts'], [row])
        row = dict(row, distance=-.003)
        self.assertEqual(summarize_contacts([row])['forbidden_contacts'], [row])
