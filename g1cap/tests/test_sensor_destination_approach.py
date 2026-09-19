import unittest

from g1cap.sensor_destination_approach import SensorDestinationApproach


def sample(t, x=0.0, *, candidate=True, segment=1, ready=True):
    return {
        'destination': ({'status': 'observed_destination_candidate', 'observed_at_s': t}
                        if candidate else {'status': 'unavailable', 'observed_at_s': t}),
        'grasp': {'status': 'available', 'raised': True, 'ready': ready},
        'frame': {'status': 'tracked_local_segment', 'time_s': t, 'segment': segment,
                  'translation_segment_m': [x, 0.0, 0.0]},
    }


class SensorDestinationApproachTests(unittest.TestCase):
    def test_requires_fresh_candidate_and_preserves_bounded_command(self):
        current = sample(0.0)
        current['destination']['direction_segment_xy'] = [1.0, 0.0]
        controller = SensorDestinationApproach(0.0, .20, lambda _: current, lambda: True)
        current = sample(.02, .01)
        current['destination']['direction_segment_xy'] = [1.0, 0.0]
        action = controller.command(.02, [0.0] * 50)
        self.assertGreater(action[43], 0.0)
        self.assertLessEqual(action[43], .08)

    def test_candidate_loss_and_segment_change_stop_without_reuse(self):
        current = sample(0.0)
        current['destination']['direction_segment_xy'] = [1.0, 0.0]
        controller = SensorDestinationApproach(0.0, .20, lambda _: current, lambda: True)
        current = sample(.02, .01, candidate=False)
        current['destination']['direction_segment_xy'] = [1.0, 0.0]
        with self.assertRaisesRegex(ValueError, 'destination_observation_unavailable'):
            controller.command(.02, [0.0] * 50)

    def test_completion_requires_observed_progress_and_time(self):
        current = sample(0.0)
        current['destination']['direction_segment_xy'] = [1.0, 0.0]
        controller = SensorDestinationApproach(0.0, .20, lambda _: current, lambda: True)
        current = sample(1.01, .17)
        current['destination']['direction_segment_xy'] = [1.0, 0.0]
        self.assertIsNone(controller.update(1.01))
        current = sample(1.02, .20)
        current['destination']['direction_segment_xy'] = [1.0, 0.0]
        self.assertEqual(controller.update(1.02), ('completed', 'destination_approach_and_hold'))


if __name__ == '__main__':
    unittest.main()
