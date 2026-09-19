import json
from pathlib import Path
import tempfile
import unittest
from g1cap.execution import execute_policy


class WorkerTests(unittest.TestCase):
    def run_policy(self, source, **options):
        calls = []
        def dispatch(method, args, kwargs, episode_id):
            calls.append((method, args, kwargs, episode_id))
            return {'pelvis_position': [0, 0, 0.8], 'status': 'completed'}
        result = execute_policy(source, {'target_position': [1, 0, 0]}, 'episode-test', dispatch, **options)
        return result, calls

    def test_worker_executes_tools_and_captures_print(self):
        result, calls = self.run_policy('def run(robot, task):\n print("policy log")\n assert robot.observe()["pelvis_position"][2] == 0.8\n robot.move_base(0.1, 0, 0, duration=0.5)')
        self.assertEqual(result['status'], 'completed', result)
        self.assertEqual(calls, [('observe', [], {}, 'episode-test'), ('move_base', [0.1, 0, 0], {'duration': 0.5}, 'episode-test')])
        self.assertIn('policy log', result['stderr'])

    def test_paired_methods_cross_the_real_worker_boundary(self):
        result,calls=self.run_policy('def run(robot, task):\n robot.check_hands({})\n robot.reach_hands({}, timeout=10)')
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual([c[0] for c in calls],['check_hands','reach_hands'])

    def test_exception_is_preserved(self):
        result, _ = self.run_policy('def run(robot, task):\n raise ValueError("deliberate")')
        self.assertEqual(result['status'], 'policy_error', result)
        self.assertIn('ValueError: deliberate', result['error'])

    def test_infinite_python_loop_is_killed(self):
        result, _ = self.run_policy('def run(robot, task):\n while True: pass', wall_timeout=0.4)
        self.assertEqual(result['status'], 'wall_timeout', result)

    def test_observation_spam_hits_request_budget(self):
        result, calls = self.run_policy('def run(robot, task):\n while True: robot.observe()', max_requests=10)
        self.assertEqual(result['status'], 'request_limit', result)
        self.assertEqual(len(calls), 10)

    def test_files_and_network_are_denied(self):
        with tempfile.TemporaryDirectory() as root:
            secret = Path(root) / 'private.txt'
            secret.write_text('test fixture only')
            source = '''def run(robot, task):
 import socket
 denied = 0
 for action in [lambda: open(%r).read(), lambda: open(%r, 'w'), lambda: socket.create_connection(('1.1.1.1', 443), timeout=.1)]:
  try:
   action()
  except OSError:
   denied += 1
 assert denied == 3, denied
''' % (str(secret), str(Path(root) / 'out.txt'))
            result, _ = self.run_policy(source)
            self.assertEqual(result['status'], 'completed', result)
            self.assertFalse((Path(root) / 'out.txt').exists())

    def test_output_flood_is_bounded(self):
        result, _ = self.run_policy('def run(robot, task):\n while True: print("x"*10000)', output_limit=16000)
        self.assertEqual(result['status'], 'output_limit', result)
        self.assertLessEqual(len(result['stderr'].encode()), 16000)

    def test_unrecognized_protocol_cannot_become_success(self):
        result, _ = self.run_policy('def run(robot, task):\n import os\n os.write(1, b\'not-json\\n\')')
        self.assertEqual(result['status'], 'protocol_error', result)

    @unittest.skipUnless(__import__('sys').platform == 'darwin', 'macOS RSS watchdog regression')
    def test_memory_watchdog_survives_closed_output_pipes(self):
        source = 'def run(robot, task):\n import os, time\n os.close(1)\n os.close(2)\n time.sleep(.15)\n data=bytearray(128*1024*1024)\n while True: pass'
        result, _ = self.run_policy(source, memory_limit_mb=64, wall_timeout=2)
        self.assertEqual(result['status'], 'memory_limit', result)

    def test_memory_budget_stops_large_allocation(self):
        result, _ = self.run_policy('def run(robot, task):\n data = bytearray(128*1024*1024)\n while True: pass', memory_limit_mb=64, wall_timeout=2)
        self.assertEqual(result['status'], 'memory_limit', result)

    def test_worker_does_not_inherit_secrets(self):
        import os
        from unittest.mock import patch
        with patch.dict(os.environ, {'G1CAP_TEST_SECRET':'fake secret'}):
            result, _ = self.run_policy('def run(robot, task):\n import os\n assert "G1CAP_TEST_SECRET" not in os.environ')
        self.assertEqual(result['status'], 'completed', result)

if __name__ == '__main__':
    unittest.main()
