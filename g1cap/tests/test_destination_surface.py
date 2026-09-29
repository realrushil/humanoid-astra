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
        self.assertNotIn('finite_support',result)
        self.assertFalse(result['complete_footprint_observed'])
        self.assertEqual(result['observed_at_s'], 2.0)
        self.assertGreater(result['valid_points'], 500)

    def test_missing_depth_is_unavailable(self):
        rgb, mask, depth = self.plane_case()
        ys, xs = np.where(mask)
        depth[ys[::2], xs[::2]] = np.nan
        result = observe_destination_surface(
            time_s=2.0, rgb=rgb, candidate_mask=mask, depth_m=depth,
            intrinsic=[[40., 0., 25.], [0., 40., 20.], [0., 0., 1.]])
        self.assertEqual(result['status'], 'unavailable')

    def test_direction_rejects_scaling_and_reflection(self):
        candidate={'status':'observed_destination_candidate','centroid_camera_m':[1.,0.,1.]}
        for rotation in (np.diag([2.,1.,1.]),np.diag([-1.,1.,1.])):
            with self.subTest(rotation=rotation.tolist()),self.assertRaises(ValueError):
                destination_direction_segment(candidate,rotation,np.eye(3))
            with self.assertRaises(ValueError):
                destination_direction_segment(candidate,np.eye(3),rotation)

    def test_direction_is_from_pelvis_not_camera_origin(self):
        candidate={'status':'observed_destination_candidate','centroid_camera_m':[1.,0.,1.]}
        result=destination_direction_segment(candidate,np.eye(3),np.eye(3),camera_position_body_m=[0.,1.,0.])
        self.assertAlmostEqual(result[0],2**-.5)
        self.assertAlmostEqual(result[1],2**-.5)

    def test_outliers_do_not_expand_the_observed_plane_patch(self):
        rgb,mask,depth=self.plane_case()
        depth[10,10:15]=10.
        result=observe_destination_surface(2.,rgb,mask,depth,[[40.,0.,25.],[0.,40.,20.],[0.,0.,1.]])
        self.assertEqual(result['status'],'observed_destination_candidate')
        self.assertLess(max(result['observed_spans_m']),.75)
        self.assertGreater(result['plane_fraction'],.9)

    def test_multiple_components_are_unavailable(self):
        rgb,mask,depth=self.plane_case()
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

class DestinationPublicationTests(unittest.TestCase):
    def camera(self):
        rgb,mask,depth=DestinationSurfaceTests().plane_case()
        return dict(step=50,time_s=1.,rgb=rgb,depth_m=depth,
                    calibration={'intrinsic':[[40.,0.,25.],[0.,40.,20.],[0.,0.,1.]]})

    def pose(self):
        pose=np.eye(4);pose[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]];pose[:3,3]=[.1,.2,.4]
        return pose

    def test_missing_motion_keeps_camera_candidate_without_direction(self):
        from g1cap.destination_surface import DestinationObservation
        stream=DestinationObservation()
        stream.update(self.camera(),{'step':50,'time_s':1.},self.pose(),None)
        result=stream.observe(1.02)
        self.assertEqual(result['status'],'observed_destination_candidate')
        self.assertNotIn('direction_segment_xy',result)
        self.assertEqual(result['direction_status'],'unavailable')
        self.assertEqual(stream.observe(1.2)['status'],'unavailable')

    def test_camera_time_and_segment_are_preserved_without_refitting(self):
        from g1cap.destination_surface import DestinationObservation
        stream=DestinationObservation();camera=self.camera()
        motion=dict(status='tracked_local_segment',time_s=1.,segment=3,body_in_segment=np.eye(4).tolist())
        stream.update(camera,{'step':50,'time_s':1.},self.pose(),motion)
        result=stream.observe(1.02)
        self.assertEqual(result['segment_id'],3)
        self.assertEqual(result['observed_at_s'],1.)
        self.assertGreater(result['direction_segment_xy'][1],.1)
        camera['rgb'][:]=0
        stream.update(camera,{'step':50,'time_s':1.},self.pose(),motion)
        self.assertEqual(stream.observe(1.02),result)
        camera=self.camera();camera.update(step=52,time_s=1.04)
        motion.update(time_s=1.04,segment=4)
        stream.update(camera,{'step':52,'time_s':1.04},self.pose(),motion)
        self.assertGreater(stream.observe(1.04)['view_epoch'],result['view_epoch'])

    def test_mismatched_encoder_or_motion_time_never_produces_direction(self):
        from g1cap.destination_surface import DestinationObservation
        for body_time,motion_time in [(1.02,1.),(1.,1.02)]:
            stream=DestinationObservation()
            motion=dict(status='tracked_local_segment',time_s=motion_time,segment=3,body_in_segment=np.eye(4).tolist())
            stream.update(self.camera(),{'step':50,'time_s':body_time},self.pose(),motion)
            self.assertNotIn('direction_segment_xy',stream.observe(1.02))

    def test_loss_then_reacquisition_changes_epoch(self):
        from g1cap.destination_surface import DestinationObservation
        stream=DestinationObservation();camera=self.camera()
        stream.update(camera,{'step':50,'time_s':1.},self.pose(),None)
        epoch=stream.observe(1.)['view_epoch']
        camera=self.camera();camera.update(step=52,time_s=1.04);camera['rgb'][:]=0
        stream.update(camera,{'step':52,'time_s':1.04},self.pose(),None)
        self.assertEqual(stream.observe(1.04)['status'],'unavailable')
        camera=self.camera();camera.update(step=54,time_s=1.08)
        stream.update(camera,{'step':54,'time_s':1.08},self.pose(),None)
        self.assertGreater(stream.observe(1.08)['view_epoch'],epoch)
