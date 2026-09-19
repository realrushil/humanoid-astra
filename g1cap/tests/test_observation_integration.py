from dataclasses import replace
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from g1cap.models import State, Task
from g1cap.observations import ObservationConfig
from g1cap.robot import Robot
from g1cap.runner import run_episode


class ObservationIntegrationTests(unittest.TestCase):
    def task(self, x=.4):
        return Task('waypoint', 'observation-test', 0, 4., (x,0.,0.), 0., 0.)

    def test_estimated_success_does_not_become_true_success(self):
        robot = Robot(self.task(), observation_config=ObservationConfig(position_bias_xy=(.4,0.)))
        result = robot.hold(1.)
        self.assertEqual(result['reason'], 'success')
        self.assertAlmostEqual(result['observation']['pelvis_position'][0], .4)
        self.assertAlmostEqual(result['measurements']['position_error'], 0.)
        summary = robot.summary()
        self.assertFalse(summary['success'])
        self.assertTrue(summary['estimated_success'])
        self.assertEqual(summary['terminal_reason'], 'estimated_success_only')
        self.assertAlmostEqual(summary['metrics']['position_error'], .4)

    def test_ground_truth_success_does_not_cancel_online_tools(self):
        robot = Robot(self.task(0.), observation_config=ObservationConfig(position_bias_xy=(.4,0.)))
        result = robot.hold(1.)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['observation']['episode_status'], 'running')
        self.assertIsNone(robot.terminal_reason)
        self.assertTrue(robot.summary()['success'])
        self.assertFalse(robot.summary()['estimated_success'])

    def test_delay_warms_up_without_exposing_future_pose(self):
        robot = Robot(self.task(), observation_config=ObservationConfig(delay_s=.1))
        observed = robot.observe()
        self.assertTrue(observed['ready'])
        self.assertGreaterEqual(robot.backend.state.sim_time, .1)
        self.assertGreaterEqual(robot.backend.state.sim_time-observed['sim_time'], .1-1e-9)
        self.assertFalse(any(e['type']=='state' and e['observation'].get('ready') is False for e in robot.trace))

    def test_same_delayed_sample_does_not_earn_dwell_twice(self):
        robot = Robot(self.task(0.), observation_config=ObservationConfig(delay_s=.1))
        before = robot.evaluator.last_time
        for _ in range(20):
            robot.observe()
        self.assertEqual(robot.evaluator.last_time, before)
        self.assertIsNone(robot.terminal_reason)

    def test_truth_fall_guard_retains_transient_failure(self):
        robot = Robot(self.task(), observation_config=ObservationConfig(delay_s=.2))
        origin = robot.backend.state
        for i, dt in enumerate((.02, .14), 1):
            state = replace(origin, sequence=origin.sequence+i, sim_time=origin.sim_time+dt, tilt=1.)
            robot.backend.state = state
            robot._score_sample(state)
        self.assertEqual(robot.terminal_reason, 'external_stop')
        self.assertEqual(robot.summary()['terminal_reason'], 'fell')

    def test_truth_guard_continues_after_true_goal_attainment(self):
        robot = Robot(self.task(0.), observation_config=ObservationConfig(position_bias_xy=(.4,0.)))
        robot.hold(1.)
        self.assertTrue(robot.summary()['success'])
        origin = robot.backend.state
        for i, dt in enumerate((.02, .14), 1):
            state = replace(origin, sequence=origin.sequence+i, sim_time=origin.sim_time+dt, tilt=1.)
            robot.backend.state = state
            robot._score_sample(state)
        self.assertEqual(robot.summary()['terminal_reason'], 'fell')
        self.assertFalse(robot.summary()['success'])

    def test_identity_observed_mode_matches_nominal_goal(self):
        nominal = Robot(self.task())
        estimated = Robot(self.task(), observation_config=ObservationConfig())
        nominal.walk_to((.4,0.))
        estimated.walk_to((.4,0.))
        self.assertEqual(estimated.summary()['success'], nominal.summary()['success'])
        self.assertAlmostEqual(estimated.control_state.pelvis_position[0], nominal.backend.state.pelvis_position[0])

    def test_saved_true_initial_state_and_estimate_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'episode'
            with patch('g1cap.runner.execute_policy', return_value={'status':'completed', 'stderr':'', 'error':''}):
                result = run_episode('def run(robot, task): pass', self.task(), output,
                                     observation_config=ObservationConfig(position_bias_xy=(.4,0.)))
            self.assertEqual(json.loads((output/'initial_state.json').read_text())['pelvis_position'][0], 0.)
            self.assertEqual(json.loads((output/'initial_observation.json').read_text())['pelvis_position'][0], .4)
            self.assertFalse(result['clean_completion'])
