import unittest
import numpy as np

from g1cap.destination_surface import (DestinationSurfaceStream, destination_direction_segment,
                                       green_destination_mask, observe_destination_surface)


class DestinationSurfaceTests(unittest.TestCase):
    def test_green_mask_is_strict_and_shape_preserving(self):
        rgb = np.zeros((2, 3, 3), dtype=np.uint8)
        rgb[0, 0] = [10, 80, 20]
        rgb[0, 1] = [60, 70, 60]
        mask = green_destination_mask(rgb)
        self.assertEqual(mask.shape, (2, 3))
        self.assertTrue(mask[0, 0])
        self.assertFalse(mask[0, 1])

    def test_direction_adapter_uses_only_calibrated_rotations(self):
        candidate = {'status': 'observed_destination_candidate', 'centroid_camera_m': [1., 0., 1.]}
        direction = destination_direction_segment(candidate, np.eye(3), np.eye(3))
        self.assertAlmostEqual(direction[0], 1.)
        self.assertAlmostEqual(direction[1], 0.)
        with self.assertRaisesRegex(ValueError, 'near_vertical'):
            destination_direction_segment({'status': candidate['status'], 'centroid_camera_m': [0., 0., 1.]},
                                          np.eye(3), np.eye(3))

    def plane_case(self):
        h, w = 40, 50
        yy, xx = np.indices((h, w))
        depth = np.full((h, w), np.nan)
        mask = np.zeros((h, w), dtype=bool)
        mask[10:30, 10:40] = True
        depth[mask] = 1.0
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[mask] = [70, 180, 80]
        return rgb, mask, depth

    def test_accepts_one_finite_surface_with_extent_and_provenance(self):
        rgb, mask, depth = self.plane_case()
        result = observe_destination_surface(
            time_s=2.0, rgb=rgb, candidate_mask=mask, depth_m=depth,
            intrinsic=[[40., 0., 25.], [0., 40., 20.], [0., 0., 1.]])
        self.assertEqual(result['status'], 'observed_destination_candidate')
        self.assertEqual(len(result['centroid_camera_m']), 3)
        self.assertGreater(result['range_camera_m'], 0)
        self.assertTrue(result['finite_support'])
        self.assertEqual(result['observed_at_s'], 2.0)
        self.assertGreater(result['valid_points'], 500)

    def test_missing_depth_or_multiple_components_is_unavailable(self):
        rgb, mask, depth = self.plane_case()
        ys, xs = np.where(mask)
        depth[ys[::2], xs[::2]] = np.nan
        result = observe_destination_surface(
            time_s=2.0, rgb=rgb, candidate_mask=mask, depth_m=depth,
            intrinsic=[[40., 0., 25.], [0., 40., 20.], [0., 0., 1.]])
        self.assertEqual(result['status'], 'unavailable')
        mask[2:8, 2:8] = True
        depth[2:8, 2:8] = 1.0
        result = observe_destination_surface(
            time_s=2.0, rgb=rgb, candidate_mask=mask, depth_m=depth,
            intrinsic=[[40., 0., 25.], [0., 40., 20.], [0., 0., 1.]])
        self.assertEqual(result['status'], 'unavailable')

    def test_rejects_bad_inputs_and_small_extent(self):
        rgb, mask, depth = self.plane_case()
        mask[:, 14:] = False
        with self.assertRaises(ValueError):
            observe_destination_surface(
                time_s=-1.0, rgb=rgb, candidate_mask=mask, depth_m=depth,
                intrinsic=[[40., 0., 25.], [0., 40., 20.], [0., 0., 1.]])
        result = observe_destination_surface(
            time_s=2.0, rgb=rgb, candidate_mask=mask, depth_m=depth,
            intrinsic=[[40., 0., 25.], [0., 40., 20.], [0., 0., 1.]])
        self.assertEqual(result['status'], 'unavailable')

    def test_stream_requires_explicit_epoch_initialization_and_rejects_gaps(self):
        rgb, mask, depth = self.plane_case()
        sample = observe_destination_surface(
            time_s=1.0, rgb=rgb, candidate_mask=mask, depth_m=depth,
            intrinsic=[[40., 0., 25.], [0., 40., 20.], [0., 0., 1.]])
        stream = DestinationSurfaceStream(max_gap_s=.15)
        with self.assertRaises(ValueError):
            stream.submit(sample, view_epoch=0)
        self.assertEqual(stream.initialize(sample, view_epoch=0)['stream_status'], 'initialized')
        next_sample = dict(sample, observed_at_s=1.1)
        self.assertEqual(stream.submit(next_sample, view_epoch=0)['stream_status'], 'tracked')
        with self.assertRaises(ValueError):
            stream.submit(dict(sample, observed_at_s=1.2), view_epoch=1)
        with self.assertRaises(ValueError):
            stream.submit(dict(sample, observed_at_s=1.4), view_epoch=0)


if __name__ == '__main__':
    unittest.main()
