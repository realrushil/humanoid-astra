import json
from pathlib import Path
import tempfile
import unittest
from g1cap.agent import ReplayAgent, run_iterations
from g1cap.runner import run_episode
from g1cap.tasks import make_task

BAD = 'def run(robot, task):\n robot.hold(0.1)\n'
GOOD = 'def run(robot, task):\n robot.walk_to(task["target_position"][:2])\n robot.turn_to(task["target_yaw"])\n robot.hold(1.0)\n'

class PipelineTests(unittest.TestCase):
    def test_feedback_reaches_second_candidate_and_resets_episode(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_iterations(ReplayAgent([BAD, GOOD]), make_task('waypoint', 0), Path(root)/'run', attempts=2)
            self.assertEqual(len(result['episodes']), 2)
            self.assertFalse(result['episodes'][0]['success'])
            self.assertTrue(result['episodes'][1]['success'], result)
            first = json.loads((Path(root)/'run/iteration-00/request.json').read_text())
            second = json.loads((Path(root)/'run/iteration-01/request.json').read_text())
            self.assertIsNone(first['feedback'])
            self.assertEqual(second['previous_source'], BAD)
            self.assertEqual(second['feedback']['summary']['terminal_reason'], 'policy_exited_early')
            for i in range(2):
                start = json.loads((Path(root)/f'run/iteration-{i:02d}/episode/initial_state.json').read_text())
                self.assertEqual(start['sim_time'], 0)
            self.assertNotEqual(result['episodes'][0]['episode_id'], result['episodes'][1]['episode_id'])
            self.assertEqual(result['agent']['kind'], 'replay')
            self.assertEqual(result['backend'], 'mock')

    def test_python_error_is_not_robot_success(self):
        with tempfile.TemporaryDirectory() as root:
            result = run_episode('def run(robot, task):\n raise ValueError("bad candidate")', make_task('waypoint',0), Path(root)/'episode')
            self.assertFalse(result['success'])
            self.assertEqual(result['terminal_reason'], 'policy_error')
            self.assertTrue((Path(root)/'episode/manifest.json').exists())
            self.assertIn('bad candidate', (Path(root)/'episode/execution.json').read_text())

    def test_existing_evidence_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'episode';p.mkdir();(p/'sentinel').write_text('keep')
            with self.assertRaises(FileExistsError):
                run_episode(BAD, make_task('waypoint',0), p)
            self.assertEqual((p/'sentinel').read_text(), 'keep')

    def test_terminal_observation_keeps_public_schema_in_worker(self):
        with tempfile.TemporaryDirectory() as root:
            source = GOOD + ' assert robot.observe()["episode_status"] == "success"\n'
            result = run_episode(source, make_task('waypoint',0), Path(root)/'episode')
            self.assertTrue(result['clean_completion'],result)

    def test_reach_reference_is_scored(self):
        with tempfile.TemporaryDirectory() as root:
            source='def run(robot, task):\n robot.reach_right(task["target_position"])\n robot.hold(1)'
            result=run_episode(source,make_task('reach',0),Path(root)/'episode')
            self.assertTrue(result['success'],result)
