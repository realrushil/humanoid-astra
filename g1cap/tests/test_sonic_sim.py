"""Coordinate-contract checks without loading MuJoCo or the controller."""
import math
import unittest

from g1cap.sonic_sim import quaternion_state, world_to_body


class PhysicsStateTransformTests(unittest.TestCase):
    def test_wxyz_yaw_rotates_world_velocity_into_body(self):
        q = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
        attitude = quaternion_state(q)
        self.assertAlmostEqual(attitude['yaw'], math.pi / 2)
        self.assertAlmostEqual(attitude['tilt'], 0.)
        result = world_to_body([0., .2, 0.], q)
        for actual, expected in zip(result, [.2, 0., 0.]):
            self.assertAlmostEqual(actual, expected)

    def test_roll_is_not_misreported_as_yaw_or_zero_tilt(self):
        attitude = quaternion_state([math.sqrt(.5), math.sqrt(.5), 0., 0.])
        self.assertAlmostEqual(attitude['roll'], math.pi / 2)
        self.assertAlmostEqual(attitude['yaw'], 0.)
        self.assertAlmostEqual(attitude['tilt'], math.pi / 2)
        result = world_to_body([0., 0., 1.], [math.sqrt(.5), math.sqrt(.5), 0., 0.])
        for actual, expected in zip(result, [0., 1., 0.]):
            self.assertAlmostEqual(actual, expected)

    def test_quaternion_sign_and_scale_do_not_change_orientation(self):
        original = quaternion_state([1., 0., 0., 1.])
        self.assertEqual(original, quaternion_state([-2., 0., 0., -2.]))

    def test_invalid_physics_values_fail_instead_of_becoming_ready_state(self):
        for q in ([0., 0., 0., 0.], [1., 0., 0.], [1., math.nan, 0., 0.]):
            with self.subTest(q=q), self.assertRaises(ValueError):
                quaternion_state(q)
        with self.assertRaises(ValueError):
            world_to_body([math.inf, 0., 0.], [1., 0., 0., 0.])


if __name__ == '__main__':
    unittest.main()
