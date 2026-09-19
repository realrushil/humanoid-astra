import unittest

try:
    import numpy as np
except ImportError:
    np = None
if np is not None:
    from g1cap.level_source_plane import fit_level_plane, box_corners, _core, _hull


@unittest.skipIf(np is None, 'NumPy runtime required')
class LevelPlaneTests(unittest.TestCase):
    def patch(self, width, depth, z):
        x, y = np.meshgrid(np.linspace(-width/2, width/2, 40), np.linspace(-depth/2, depth/2, 40))
        return np.c_[x.ravel(), y.ravel(), np.full(x.size, z)]

    def test_partial_table_uses_floor_orientation_and_fresh_table_height(self):
        source = self.patch(.3, .1, .7)
        a = fit_level_plane(self.patch(2, 2, 0), source, [[0, .3, .8]])
        b = fit_level_plane(self.patch(2, 2, .2), source, [[0, .3, .8]])
        self.assertEqual(a['status'], 'measured')
        self.assertLess(a['height_perturbation_m'], .01)
        self.assertLess(a['source_residual_p90_m'], 1e-10)
        sign = np.sign(np.dot(a['normal'], b['normal']))
        self.assertAlmostEqual(a['offset_m'], sign*b['offset_m'], places=10)
        self.assertAlmostEqual(a['height_perturbation_m'], b['height_perturbation_m'], places=10)

    def test_repeated_pixels_do_not_invent_independent_noise_precision(self):
        floor, source = self.patch(2, 2, 0), self.patch(.3, .1, .7)
        a = fit_level_plane(floor, source, [[0, .3, .8]])
        b = fit_level_plane(np.repeat(floor, 3, axis=0), np.repeat(source, 3, axis=0), [[0, .3, .8]])
        self.assertAlmostEqual(a['height_perturbation_m'], b['height_perturbation_m'], places=10)

    def test_correlated_floor_error_is_covered_without_noise_averaging(self):
        floor, source = self.patch(2, 2, 0), self.patch(.3, .1, .7)
        queries = np.array([[.5, .6, .9], [-.5, .6, .9]])
        baseline = fit_level_plane(floor, source, queries)
        n = np.array(baseline['normal']); d = baseline['offset_m']
        for direction in ([1, 0], [0, 1], [1, 1], [-1, 1]):
            moved = floor.copy()
            moved[:, 2] += .003*np.sign(moved[:, :2]@direction)
            result = fit_level_plane(moved, source, queries)
            sign = np.sign(n@result['normal'])
            difference = abs(queries@(sign*np.array(result['normal'])-n) + sign*result['offset_m']-d)
            self.assertLessEqual(max(difference), baseline['height_perturbation_m'])

    def test_tilt_is_visible_as_a_model_residual(self):
        source = self.patch(.4, .3, .7); source[:, 2] += .05*source[:, 0]
        result = fit_level_plane(self.patch(2, 2, 0), source, [[0, .3, .8]])
        self.assertGreater(result['source_residual_p90_m'], .003)

    def test_concentrated_floor_bounds_svd_not_only_graph_regression(self):
        # Freeze membership/tangents, and perturb both floor and source along
        # the original normal. OLS-only sensitivity falls below the actual TLS
        # distance change for this concentrated (not uniform) footprint.
        xx = .9*np.r_[np.linspace(-.01, .01, 80), np.linspace(-.1, .1, 20)]
        x, y = np.meshgrid(xx, np.linspace(-.5, .5, 30))
        floor = np.c_[x.ravel(), y.ravel(), np.zeros(x.size)]
        sx, sy = np.meshgrid(np.linspace(-1e-6, 1e-6, 25), np.linspace(-.05, .05, 25))
        source = np.c_[sx.ravel(), sy.ravel(), np.full(sx.size, .7)]
        query = np.array([[.1087, 0, .7]])
        baseline = fit_level_plane(floor, source, query)
        f, s = _core(floor), _core(source)
        inverse = np.linalg.pinv(np.c_[f[:, :2], np.ones(len(f))])
        f[:, 2] += .003*np.sign(inverse[0])
        s[:, 2] += .003
        _, _, axes = np.linalg.svd(f-f.mean(axis=0), full_matrices=False)
        normal = axes[-1]*np.sign(axes[-1, -1])
        actual = abs((query@normal-np.median(s@normal))[0])
        self.assertGreater(actual, .01)
        self.assertLessEqual(actual, baseline['height_perturbation_m'])

    def test_sparse_remote_points_do_not_rescue_degenerate_floor(self):
        line = self.patch(2, 0, 0)
        result = fit_level_plane(np.r_[line, [[0, 2, 0], [0, -2, 0]]], self.patch(.3, .1, .7), [[0, 0, .8]])
        self.assertEqual(result['status'], 'unavailable')

    def test_hull_preserves_joint_influence_and_distance_extrema(self):
        rng = np.random.default_rng(17)
        for points in (rng.normal(size=(100, 2)), np.c_[np.arange(20), np.zeros(20)],
                       np.zeros((20, 2))):
            vertices = _hull(np.repeat(points, 2, axis=0))
            influence = rng.normal(size=(2, 73))
            for query in rng.normal(size=(4, 2)):
                all_deltas, hull_deltas = query-points, query-vertices
                self.assertAlmostEqual(np.abs(all_deltas@influence).sum(axis=1).max(),
                                       np.abs(hull_deltas@influence).sum(axis=1).max())
                self.assertAlmostEqual(np.linalg.norm(all_deltas, axis=1).max(),
                                       np.linalg.norm(hull_deltas, axis=1).max())

    def test_missing_or_invalid_clouds_do_not_publish_a_plane(self):
        for floor in (None, [], np.empty((0, 3)), [[np.nan, 0, 0]]*500):
            result = fit_level_plane(floor, self.patch(.3, .1, .7), [[0, 0, .8]])
            self.assertEqual(result['status'], 'unavailable')

    def test_corners_include_measured_rotation_and_size(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        corners = box_corners(dict(center_camera_m=[1, 2, 3], axes_camera=rotation,
                                  dimensions_m=[.2, .4, .6]))
        np.testing.assert_allclose(corners.min(axis=0), [.8, 1.9, 2.7])
        np.testing.assert_allclose(corners.max(axis=0), [1.2, 2.1, 3.3])


if __name__ == '__main__':
    unittest.main()
