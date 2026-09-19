import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from g1cap.tasks import make_task


class TrialTests(unittest.TestCase):
    def run_trial(self, *args, **kwargs):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.trial'))
        from g1cap.trial import run_trial
        return run_trial(*args, **kwargs)

    def test_single_generation_and_episode_with_frozen_submission(self):
        calls = []
        source = 'def run(robot, task):\n robot.hold(1)\n'
        class Agent:
            metadata = {'kind': 'fixture'}
            def generate(self, request, directory):
                calls.append(('generate', request))
                return source
        def episode(code, task, directory):
            calls.append(('episode', code))
            self.assertEqual((directory.parent/'candidate.py').read_text(), source)
            return {'success': False, 'clean_completion': False, 'terminal_reason': 'deadline',
                    'execution_status': 'completed', 'physics': False}
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)/'trial'
            report = self.run_trial(Agent(), make_task(), output, episode_runner=episode)
            self.assertEqual([c[0] for c in calls], ['generate', 'episode'])
            self.assertNotIn('feedback', calls[0][1])
            self.assertEqual(report['protocol'], 'single_turn')
            self.assertEqual(report['generation_sessions'], 1)
            self.assertEqual(report['episode_attempts'], 1)
            self.assertFalse(report['physics'])
            self.assertEqual(json.loads((output/'evidence/report.json').read_text()), report)
            with self.assertRaises(FileExistsError):
                self.run_trial(Agent(), make_task(), output, episode_runner=episode)

    def test_generation_failure_never_starts_episode(self):
        class Agent:
            metadata = {'kind': 'fixture'}
            def generate(self, request, directory):
                raise RuntimeError('model unavailable')
        def episode(*args):
            self.fail('generation failure must not execute a policy')
        with tempfile.TemporaryDirectory() as root:
            report = self.run_trial(Agent(), make_task(), Path(root)/'trial', episode_runner=episode)
            self.assertEqual(report['episode_attempts'], 0)
            self.assertEqual(report['status'], 'generation_error')
            self.assertIn('model unavailable', report['error'])

    def test_episode_exception_is_preserved_without_retry(self):
        class Agent:
            metadata = {'kind': 'fixture'}
            def generate(self, request, directory):
                return 'def run(robot, task): pass'
        def episode(*args):
            raise RuntimeError('startup failed')
        with tempfile.TemporaryDirectory() as root:
            report = self.run_trial(Agent(), make_task(), Path(root)/'trial', backend='sonic', episode_runner=episode)
            self.assertEqual(report['episode_attempts'], 1)
            self.assertEqual(report['status'], 'episode_error')
            self.assertIn('startup failed', report['error'])
