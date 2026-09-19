import unittest
from g1cap.arena_placement_geometry import placement_retraction


def fixture(palm_gap=.0075):
    table = dict(min=[0., -.5, -.1], max=[1., .5, 0.])
    shape = dict(min=[-.01, -.01, 0.], max=[.01, .01, .03])
    geometry = {'robot': {'right_hand_palm_link': [shape], 'right_wrist_link': [shape]}}
    observation = dict(root_pos=[-.5, 0., .6], box_pos=[.2, 0., .2],
        box_bounds=dict(min=[.1, -.1, .1], max=[.3, .1, .3]),
        surfaces={'destination': dict(bounds=table, clearance_m=.1)},
        body_poses={
            'right_hand_palm_link': dict(pos=[.2, 0., .102+palm_gap], xyzw=[0.,0.,0.,1.]),
            'right_wrist_link': dict(pos=[.008, 0., .1], xyzw=[0.,0.,0.,1.])})
    return observation, geometry, {'destination': [table]}


class PreferredRetractionTests(unittest.TestCase):
    def test_unattainable_preference_uses_best_hard_feasible_margin(self):
        args = fixture()
        self.assertIsNone(placement_retraction(*args, arm_margin_m=.008))
        result = placement_retraction(*args, preferred_arm_margin_m=.008)
        self.assertAlmostEqual(result['predicted_arm_clearance_m'], .0075)
        self.assertAlmostEqual(result['distance_m'], .0255)
        self.assertGreaterEqual(result['predicted_box_margin_m'], .01)

    def test_never_falls_back_below_the_hard_margin(self):
        self.assertIsNone(placement_retraction(*fixture(.004), preferred_arm_margin_m=.008))

    def test_attainable_preference_matches_the_old_strict_plan(self):
        args = fixture(.012)
        self.assertEqual(placement_retraction(*args, preferred_arm_margin_m=.008),
                         placement_retraction(*args, arm_margin_m=.008))

    def test_preference_does_not_override_box_containment(self):
        row, geometry, supports = fixture()
        row['box_bounds']['min'][0] = .011
        self.assertIsNone(placement_retraction(row, geometry, supports, preferred_arm_margin_m=.008))

    def test_invalid_preference_is_rejected(self):
        for margin in (.004, .03, float('nan')):
            with self.assertRaises(ValueError):
                placement_retraction(*fixture(), preferred_arm_margin_m=margin)


if __name__ == '__main__':
    unittest.main()
