import unittest
import numpy as np

from g1cap.remembered_source_plane import RememberedSourcePlane


def motion(time_s=0.0, segment=1, status='tracked_local_segment', pose=None):
    transform = np.eye(4) if pose is None else np.asarray(pose, float)
    return dict(time_s=time_s, segment=segment, status=status,
                body_in_segment=transform.tolist())


class RememberedSourcePlaneTests(unittest.TestCase):
    def make(self):
        return RememberedSourcePlane(
            time_s=0.0, motion=motion(), normal_body=[-1, 0, 0],
            offset_body=1.0, anchor_body=[1, 0, 0],
            plane_error_m=0.003, plane_angle_rad=0.01, max_age_s=10.0)

    def test_query_uses_current_segment_transform_and_retains_provenance(self):
        plane = self.make()
        pose = np.eye(4)
        pose[0, 3] = 0.2
        result = plane.query(1.0, motion(1.0, pose=pose), [[0.3, 0, 0]],
                             translation_error_m=0.0, rotation_error_rad=0.0)
        self.assertAlmostEqual(result['nominal_m'][0], 0.5)
        self.assertEqual(result['observed_at_s'], 0.0)
        self.assertEqual(result['transform_at_s'], 1.0)
        self.assertFalse(result['fresh_obstacle_observation'])

    def test_segment_loss_latches_unavailable_and_cannot_be_bridged(self):
        plane = self.make()
        with self.assertRaises(ValueError):
            plane.query(.04, motion(.04, status='unavailable'), [[0, 0, 0]],
                        translation_error_m=0, rotation_error_rad=0)
        with self.assertRaises(ValueError):
            plane.query(.08, motion(.08), [[0, 0, 0]],
                        translation_error_m=0, rotation_error_rad=0)

    def test_invalid_age_segment_and_points_are_rejected(self):
        plane = self.make()
        for now, frame in [(11.0, motion(11.0)), (1.0, motion(1.0, segment=2)),
                           (1.0, motion(.9))]:
            with self.assertRaises(ValueError):
                plane.query(now, frame, [[0, 0, 0]],
                            translation_error_m=.03, rotation_error_rad=.02)
        with self.assertRaises(ValueError):
            plane.query(1.0, motion(1.0), [], translation_error_m=0,
                        rotation_error_rad=0)

    def test_conditional_bound_includes_declared_pose_and_plane_uncertainty(self):
        plane = self.make()
        result = plane.query(1.0, motion(1.0), [[0.3, 0.1, -0.2]],
                             translation_error_m=0.03, rotation_error_rad=0.02)
        self.assertEqual(result['status'], 'remembered_conditional_plane')
        self.assertGreater(result['uncertainty_m'][0], 0.03)
        self.assertLessEqual(result['lower_m'][0], result['nominal_m'][0])


if __name__ == '__main__':
    unittest.main()
