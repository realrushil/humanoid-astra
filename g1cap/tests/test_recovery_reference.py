"""Controlled tool outcomes test policy decisions; physics is checked separately."""
from dataclasses import replace
from pathlib import Path
import runpy
import unittest

from g1cap.models import Task
from g1cap.robot import Robot


class InterruptedWalkRobot(Robot):
    """Inject a native-tool failure; retain real observation/stop/turn/evaluation."""
    def __init__(self, task, after_walk=(.1,0.,.75), persistent=False, failure='no_progress'):
        super().__init__(task)
        self.walk_calls = 0
        self.after_walk = after_walk
        self.persistent = persistent
        self.failure = failure

    def walk_to(self, goal, timeout=None):
        self.walk_calls += 1
        if self.walk_calls==1 or self.persistent:
            self.backend.state = replace(self.backend.state,pelvis_position=self.after_walk)
            status = 'timed_out' if self.failure=='no_progress' else 'backend_error'
            return self._result('walk_to',status,self.failure,self.control_state.sim_time)
        return super().walk_to(goal,timeout)


class RecoveryReferenceTests(unittest.TestCase):
    def policy(self):
        path = Path(__file__).resolve().parents[1]/'examples/recovery_reference.py'
        self.assertTrue(path.exists(),'bounded recovery policy is missing')
        return runpy.run_path(str(path))['run']

    def task(self):
        return Task('waypoint','recovery-test',0,35.,(.4,0.,0.),.3,0.)

    def test_normal_goal_still_completes(self):
        run,task = self.policy(),self.task()
        robot = Robot(task)
        run(robot,task.to_dict())
        self.assertTrue(robot.summary()['success'])

    def test_near_goal_after_failure_sets_heading_without_second_walk(self):
        run,task = self.policy(),self.task()
        robot = InterruptedWalkRobot(task,after_walk=(.36,0.,.75))
        run(robot,task.to_dict())
        self.assertEqual(robot.walk_calls,1)
        self.assertTrue(robot.summary()['success'])
        operations = [e['operation'] for e in robot.trace if e['type']=='request']
        self.assertEqual(operations[:2],['stop','turn_to'])

    def test_far_goal_after_failure_retries_once_and_reaches_goal(self):
        run,task = self.policy(),self.task()
        robot = InterruptedWalkRobot(task)
        run(robot,task.to_dict())
        self.assertEqual(robot.walk_calls,2)
        self.assertTrue(robot.summary()['success'])

    def test_persistent_no_progress_does_not_retry_indefinitely(self):
        run,task = self.policy(),self.task()
        robot = InterruptedWalkRobot(task,persistent=True)
        run(robot,task.to_dict())
        self.assertEqual(robot.walk_calls,2)
        self.assertFalse(robot.summary()['success'])
        self.assertFalse(any(e['type']=='request' and e['operation']=='turn_to' for e in robot.trace))

    def test_backend_error_does_not_trigger_recovery_motion(self):
        run,task = self.policy(),self.task()
        robot = InterruptedWalkRobot(task,failure='backend_failed')
        run(robot,task.to_dict())
        self.assertEqual(robot.walk_calls,1)
        self.assertFalse(any(e['type']=='request' for e in robot.trace))
