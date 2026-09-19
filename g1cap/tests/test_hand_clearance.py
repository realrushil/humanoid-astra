"""Portable regression checks for geometry/rate behavior, without run fixtures."""
import copy
import unittest
from g1cap.toolkit.hand_clearance import ClearanceBias

GEOMETRY = {'robot': {'left_hand_link': [dict(min=[-.02]*3, max=[.02]*3)]}}


def observation(time, gap, x=0.):
    return dict(time=time,
        body_poses={'left_hand_link': dict(pos=[x, 0., gap+.02], xyzw=[0., 0., 0., 1.])},
        surfaces={'support': {'bounds': dict(min=[-.5, -.5, -.1], max=[.5, .5, 0.])}})


class HandClearanceTests(unittest.TestCase):
    def test_sensor_gap_drives_same_bounded_bias_without_scene_geometry(self):
        legacy, measured = ClearanceBias(GEOMETRY), ClearanceBias(None)
        self.assertTrue(hasattr(measured, 'update_gap'))
        for i in range(20):
            gap = .12-i*.016 if i < 10 else None
            expected = legacy.update(observation(i*.02, gap or 0., x=0. if gap is not None else 2.))
            actual = measured.update_gap(i*.02, gap)
            for key in ('bias_m', 'requested_bias_m', 'closing_speed_m_s'):
                self.assertAlmostEqual(actual[key], expected[key])

    def test_safe_stationary_hand_is_unchanged(self):
        controller = ClearanceBias(GEOMETRY)
        for i in range(6):
            self.assertEqual(controller.update(observation(i*.02, .08))['bias_m'], 0.)

    def test_closing_hand_is_corrected_before_contact(self):
        controller = ClearanceBias(GEOMETRY)
        results = [controller.update(observation(i*.02, .12-i*.016)) for i in range(6)]
        first = next(r for r in results if r['bias_m'] > 0)
        self.assertGreater(first['margin_m'], 0.)
        self.assertGreater(first['closing_speed_m_s'], 0.)

    def test_bias_and_rate_are_bounded_and_decay_away_from_support(self):
        controller = ClearanceBias(GEOMETRY)
        previous = 0.
        for i in range(50):
            result = controller.update(observation(i*.02, -.03-i*.01))
            self.assertLessEqual(result['bias_m'], .060001)
            self.assertLessEqual(result['bias_m']-previous, .004001)
            previous = result['bias_m']
        result = controller.update(observation(1., 0., x=2.))
        self.assertIsNone(result['margin_m'])
        self.assertAlmostEqual(previous-result['bias_m'], .0004)

    def test_support_world_translation_does_not_change_bias(self):
        a, b = ClearanceBias(GEOMETRY), ClearanceBias(GEOMETRY)
        for i in range(6):
            row = observation(i*.02, .12-i*.016)
            moved = copy.deepcopy(row)
            delta = [3., -2., 1.]
            pose = moved['body_poses']['left_hand_link']
            pose['pos'] = [x+d for x,d in zip(pose['pos'], delta)]
            for key in ('min', 'max'):
                bounds = moved['surfaces']['support']['bounds']
                bounds[key] = [x+d for x,d in zip(bounds[key], delta)]
            self.assertAlmostEqual(a.update(row)['bias_m'], b.update(moved)['bias_m'])

    def test_missing_duplicate_or_reordered_samples_are_rejected(self):
        for time in (0., -.02, .04, float('nan')):
            controller = ClearanceBias(GEOMETRY)
            controller.update(observation(0., .04))
            with self.assertRaises(ValueError):
                controller.update(observation(time, .04))

    def test_new_acquisition_has_no_previous_correction_or_velocity_history(self):
        old = ClearanceBias(GEOMETRY)
        for i in range(6):
            old.update(observation(i*.02, .12-i*.016))
        self.assertGreater(old.bias, 0.)
        new = ClearanceBias(GEOMETRY)
        result = new.update(observation(40., .08))
        self.assertEqual(result['bias_m'], 0.)
        self.assertEqual(result['closing_speed_m_s'], 0.)


if __name__ == '__main__':
    unittest.main()
