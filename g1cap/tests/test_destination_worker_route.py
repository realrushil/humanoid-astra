"""Qualification-only worker route; no simulation or model calls."""
import tempfile
import time
import unittest
from pathlib import Path

from g1cap.arena_session import ArenaSession
from test_arena_session import FakeControl


class DestinationWorkerRouteTests(unittest.TestCase):
    def call(self, method):
        control = FakeControl()
        calls = []
        original = control.start

        def start(name, arguments, observation):
            calls.append((name, arguments))
            return original(name, arguments, observation)

        control.start = start

        def executor(source, task, round_id, dispatch, **options):
            result = dispatch(method, [], {'distance_m': .2}, round_id)
            return dict(status='completed', tool_result=result, task=task)

        with tempfile.TemporaryDirectory() as folder:
            session = ArenaSession(control, {}, Path(folder), executor=executor,
                sensor_observation=lambda t: dict(time=t, observation_mode='sensor_estimates_v1'))
            try:
                session.tick(dict(time=0., step=0))
                session.submit('qualification route')
                deadline = time.monotonic() + 2.
                step = 0
                while not session.rounds and time.monotonic() < deadline:
                    step += 1
                    session.tick(dict(time=step * .02, step=step))
                    time.sleep(.001)
                self.assertEqual(len(session.rounds), 1)
                return session.rounds[0]['execution'], calls
            finally:
                session.close()

    def test_approach_reaches_existing_single_owner_queue(self):
        result, calls = self.call('move_with_box')
        self.assertEqual(result['tool_result']['status'], 'completed')
        self.assertEqual(calls, [('move_with_box', {'distance_m': .2})])
        self.assertIn('move_with_box', result['task']['available_tools'])

    def test_privileged_placement_remains_unavailable(self):
        result, calls = self.call('place_box')
        self.assertEqual(result['tool_result']['reason'], 'sensor_tool_unavailable')
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
