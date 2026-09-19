import math
import unittest

from g1cap.models import State
from g1cap.observations import ObservationConfig, ObservationStream


def state(t, seq, x=0.0, yaw=0.0, ok=True, age=0.0):
    return State(sim_time=t, sequence=seq, pelvis_position=(x, 0.0, .75),
                 right_wrist_position=(x + .25, -.20, .90), pelvis_yaw=yaw,
                 backend_ok=ok, state_age_s=age)


class ObservationTests(unittest.TestCase):
    def test_delay_is_causal_and_not_ready_until_sample_exists(self):
        stream = ObservationStream(ObservationConfig(delay_s=.2)); stream.push(state(1., 1, x=1.))
        self.assertFalse(stream.read(state(1.1, 2)).ready)
        observed = stream.read(state(1.2, 3))
        self.assertTrue(observed.ready); self.assertEqual((observed.sim_time, observed.sequence), (1., 1))

    def test_capture_noise_stable_and_truth_unchanged(self):
        truth = state(1., 1, x=1., yaw=3.); stream = ObservationStream(ObservationConfig(
            position_bias_xy=(.1, -.2), yaw_bias=.5, position_noise_std=.01, yaw_noise_std=.02, seed=4))
        stream.push(truth); first = stream.read(state(2., 2)); second = stream.read(state(2.1, 3))
        self.assertEqual(truth, state(1., 1, x=1., yaw=3.))
        self.assertEqual(first.pelvis_position, second.pelvis_position)
        self.assertEqual(first.right_wrist_position, second.right_wrist_position)
        self.assertEqual(first.pelvis_yaw, second.pelvis_yaw)
        self.assertNotEqual(first.pelvis_position[0], 1.)

    def test_same_seed_reproduces_noise(self):
        config = ObservationConfig(position_noise_std=.05, yaw_noise_std=.1, seed=9); results = []
        for _ in range(2):
            stream = ObservationStream(config); stream.push(state(1., 1, x=1., yaw=.2)); results.append(stream.read(state(2., 2)))
        self.assertEqual(results[0], results[1])

    def test_age_and_health_propagate(self):
        stream = ObservationStream(); stream.push(state(1., 1, age=.1))
        self.assertAlmostEqual(stream.read(state(1.5, 2, age=.2)).state_age_s, .7)
        stream.push(state(2., 3, ok=False)); self.assertFalse(stream.read(state(2., 4)).backend_ok)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({'delay_s': True}, {'delay_s': .6}, {'yaw_bias': math.nan},
                        {'position_bias_xy': (0.,)}, {'position_noise_std': -.1},
                        {'yaw_noise_std': .3}, {'seed': True}, {'unknown': 1}):
            with self.assertRaises((TypeError, ValueError)):
                ObservationConfig(**kwargs)


if __name__ == '__main__':
    unittest.main()
