import unittest
import numpy as np
from g1cap.source_stop import SourceSides
from g1cap.loaded_stop import TopPlane
from g1cap.remembered_source_plane import RememberedSourcePlane
from g1cap._prepared_source_sides import prepare
from g1cap.departure_sweep import DepartureSweep
from g1cap._source_turn_probe import TurnPlane


def source(top=True):
    motion = dict(status='tracked_local_segment', segment=1, time_s=1.,
                  body_in_segment=np.eye(4).tolist())
    memory = RememberedSourcePlane(1., motion, [-1, 0, 0], .5, [.5, 0, 0],
        plane_error_m=.003, plane_angle_rad=.05, max_age_s=10.)
    plane = TopPlane([0, 0, 1], 0, [.4, 0, 0], .004, .01, 1., np.eye(3)) if top else None
    return SourceSides(plane, memory, motion, np.eye(3), 1.)


class BatchedDepartureTests(unittest.TestCase):
    def test_many_parts_match_independent_scalar_choices(self):
        rng = np.random.default_rng(42)
        for top in (False, True):
            p = prepare(source(top), 1., np.eye(3))
            points = rng.uniform(-.8, .8, (40, 8, 3))
            front, height, travel = rng.uniform(0., .2, (3, 40))
            expected = np.array([p.bounds(part, 1., np.eye(3), front_closing_m=f,
                top_closing_m=h, point_displacement_m=d)
                for part, f, h, d in zip(points, front, height, travel)])
            actual = p.bounds_many(points, 1., np.eye(3), front_closing_m=front,
                top_closing_m=height, point_displacement_m=travel)
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)

    def test_whole_part_cannot_choose_a_different_side_per_vertex(self):
        p = prepare(source(), 1., np.eye(3))
        points = np.array([[[.1, 0, -.2], [.9, 0, .2]],
                           [[.1, 0, .2], [.2, 0, .3]]])
        actual = p.bounds_many(points, 1., np.eye(3))
        for part, values in zip(points, actual):
            np.testing.assert_allclose(values, p.bounds(part, 1., np.eye(3)), atol=1e-12)
        self.assertLess(actual[0].min(), 0)

    def test_invalid_time_frame_points_and_allowances_reject(self):
        p = prepare(source(), 1., np.eye(3))
        for points, now, rotation, kw in [
            (np.ones((2, 8, 3)), 1.02, np.eye(3), {}),
            (np.ones((2, 8, 3)), 1., -np.eye(3), {}),
            (np.full((2, 8, 3), np.nan), 1., np.eye(3), {}),
            (np.ones((2, 0, 3)), 1., np.eye(3), {}),
            (np.ones((2, 8, 3)), 1., np.eye(3), {'front_closing_m': -.1}),
            (np.ones((2, 8, 3)), 1., np.eye(3), {'point_displacement_m': [0., np.inf]}),
        ]:
            with self.subTest(now=now, kw=kw), self.assertRaises(ValueError):
                p.bounds_many(points, now, rotation, **kw)

    def test_invalid_memory_preserves_only_top_fallback(self):
        for top in (False, True):
            p = prepare(source(top), 1., np.eye(3))
            p.memory.invalid = True
            points = np.ones((2, 8, 3)) * .2
            if top:
                np.testing.assert_allclose(p.bounds_many(points, 1., np.eye(3)),
                    np.array([p.bounds(part, 1., np.eye(3)) for part in points]))
            else:
                with self.assertRaises(ValueError):p.bounds_many(points, 1., np.eye(3))

    def test_complete_yaw_sweep_matches_scalar_reference(self):
        rng = np.random.default_rng(7)
        for top in (False, True):
            for yaw in (-np.pi/2, -.13, .13, np.pi/2):
                p = prepare(source(top), 1., np.eye(3))
                up = np.array([.01, -.02, 1.]); up /= np.linalg.norm(up)
                sweep = DepartureSweep(p, np.eye(3), up, yaw)
                for _ in range(8):
                    points = rng.uniform(-.8, .8, (8, 3))
                    expected = np.minimum.reduce([TurnPlane.bounds(sweep, points @ R.T,
                        1., np.eye(3)) for R in sweep.rotations])
                    np.testing.assert_allclose(sweep.bounds(points, 1., np.eye(3)),
                                               expected, atol=2e-12, rtol=0)

    def test_nonidentity_motion_and_query_rotation_match(self):
        def yaw(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        s = source()
        pose = np.eye(4); pose[:3, :3] = yaw(.2); pose[:3, 3] = [.03, -.02, .01]
        s.motion = dict(s.motion, time_s=1.02, body_in_segment=pose.tolist())
        s.gyro = yaw(.1)
        rotation = yaw(.3)
        p = prepare(s, 1.02, rotation)
        points = np.random.default_rng(3).normal(size=(20, 8, 3))
        expected = np.array([p.bounds(part, 1.02, rotation, point_displacement_m=.1)
                             for part in points])
        np.testing.assert_allclose(p.bounds_many(points, 1.02, rotation,
            point_displacement_m=.1), expected, atol=2e-12, rtol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
