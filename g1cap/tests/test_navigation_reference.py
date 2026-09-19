"""Software checks for the example policy; the mock does not qualify physics."""
import math
from pathlib import Path
import runpy
import unittest

from g1cap.models import Task
from g1cap.robot import Robot


class NavigationReferenceTests(unittest.TestCase):
    def policy(self):
        path = Path(__file__).resolve().parents[1]/'examples/turn_forward_reference.py'
        self.assertTrue(path.exists(), 'turn-and-forward reference policy is missing')
        return runpy.run_path(str(path))['run']

    def test_reaches_offset_goal_without_lateral_requests(self):
        run = self.policy()
        task = Task('waypoint', 'offset', 0, 35., (.3,.2,0.), .3, 0.)
        robot = Robot(task)
        run(robot, task.to_dict())
        self.assertTrue(robot.summary()['success'], robot.summary())
        moves = [e['arguments'] for e in robot.trace
                 if e['type']=='request' and e['operation']=='move_base']
        self.assertTrue(moves)
        self.assertTrue(all(m['vx']>0 and m['vy']==0 and m['yaw_rate']==0 for m in moves))

    def test_goal_behind_uses_bounded_turns(self):
        run = self.policy()
        task = Task('waypoint', 'behind', 0, 35., (-.3,0.,0.), math.pi, 0.)
        robot = Robot(task)
        run(robot, task.to_dict())
        self.assertTrue(robot.summary()['success'], robot.summary())
        self.assertFalse(any(e['type']=='result' and e['result']['status']=='rejected'
                             for e in robot.trace))

    def test_timeout_returns_without_more_motion_requests(self):
        run = self.policy()
        task = Task('waypoint', 'short-budget', 0, .2, (.4,0.,0.), .3, 0.)
        robot = Robot(task)
        run(robot, task.to_dict())
        self.assertEqual(robot.terminal_reason, 'timeout')
        terminal_index = next(i for i,e in enumerate(robot.trace) if e['type']=='terminal')
        self.assertFalse(any(e['type']=='request' for e in robot.trace[terminal_index+1:]))
