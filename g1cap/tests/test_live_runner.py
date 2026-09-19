import dataclasses
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from g1cap.models import State
from g1cap.robot import MockBackend, Robot
from g1cap.runner import run_episode
from g1cap.remote_sonic import RemoteSonicRunner
from g1cap.tasks import make_task


class MeasuredFixture(MockBackend):
    """Interface fixture only; never used in saved physics evidence."""
    name = 'sonic'
    supported_operations = {'observe', 'move_base', 'stop', 'hold', 'walk_to', 'turn_to'}
    metadata = {'physics': True, 'robot_model': 'fixture', 'controller': 'fixture'}


class LiveRunnerContractTests(unittest.TestCase):
    def test_initial_artifact_uses_frozen_baseline(self):
        backend = MeasuredFixture(make_task())
        backend.initial_state = backend.state
        backend.state = dataclasses.replace(backend.state, sim_time=.1, sequence=1)
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)/'episode'
            with patch('g1cap.runner.execute_policy', return_value={
                'status': 'completed', 'stderr': '', 'error': ''}):
                run_episode('def run(robot, task): pass', make_task(), output, backend=backend)
            self.assertEqual(json.loads((output/'initial_state.json').read_text())['sim_time'], 0.)

    def test_remote_failure_cannot_be_clean_after_task_success(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)
            (directory/'candidate.py').write_text('saved policy')
            def transport(command, **kwargs):
                if command[:3] == ['scp', '-q', '-r']:
                    destination = Path(command[-1])/'episode'
                    destination.mkdir(parents=True)
                    (destination/'summary.json').write_text(json.dumps(dict(
                        success=True, clean_completion=True, execution_status='completed')))
                return subprocess.CompletedProcess(command, 1 if command[0] == 'ssh' else 0, '', '')
            with patch('g1cap.remote_sonic.subprocess.run', side_effect=transport):
                result = RemoteSonicRunner('test-host', '/test-root', 0)(
                    'saved policy', make_task(), directory/'episode')
            self.assertTrue(result['success'])
            self.assertFalse(result['clean_completion'])
            self.assertEqual(result['execution_status'], 'remote_runtime_error')

    def test_observe_scores_queued_physics_including_a_transient_fall(self):
        backend = MeasuredFixture(make_task())
        samples = [dataclasses.replace(backend.state, sim_time=t, sequence=i,
                                      tilt=1. if t < .3 else 0.)
                   for i, t in enumerate((.1, .21, .3), 1)]
        pending = list(samples)
        def consume():
            result = list(pending)
            pending.clear()
            return result
        backend.consume_samples = consume
        robot = Robot(make_task(), backend)
        backend.state = samples[-1]
        result = robot.observe()
        self.assertEqual(result['episode_status'], 'fell')
        scored = [e['observation']['sim_time'] for e in robot.trace if e['type'] == 'state']
        self.assertIn(.1, scored)
        self.assertIn(.21, scored)

    def test_unsupported_reach_is_rejected_without_moving(self):
        backend = MeasuredFixture(make_task())
        robot = Robot(make_task(), backend)
        before = backend.state
        result = robot.reach_right((.30, -.20, .90))
        self.assertEqual(result['reason'], 'unsupported_operation')
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(backend.state, before)
        self.assertEqual(robot.observe()['backend'], 'sonic')

    def test_measured_initial_height_sets_fall_threshold(self):
        backend = MeasuredFixture(make_task())
        backend.state = dataclasses.replace(State(), pelvis_position=(0., 0., 1.))
        robot = Robot(make_task(), backend)
        self.assertEqual(robot.evaluator.start_height, 1.)

    def test_runner_records_selected_backend(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('g1cap.runner.execute_policy', return_value={
                'status': 'completed', 'stderr': '', 'error': ''}):
                result = run_episode('def run(robot, task): pass', make_task(),
                                     Path(root)/'episode', backend=MeasuredFixture(make_task()))
            self.assertEqual(result['backend'], 'sonic')
            self.assertTrue(result['physics'])
