"""Bounded API behavior against synthetic kinematics, not physics qualification."""
import dataclasses
import json
import math
import unittest
from g1cap.models import State
from g1cap.tasks import make_task
from g1cap.robot import MockBackend, Robot, StateUnavailable


class RobotTests(unittest.TestCase):
    def test_observe_is_detached_and_does_not_advance(self):
        robot = Robot(make_task())
        before = robot.observe()
        snapshot = robot.dispatch('observe', [], {}, robot.episode_id)
        snapshot['pelvis_position'][0] = 50
        self.assertEqual(robot.observe(), before)
        self.assertEqual(robot.trace[0]['type'], 'state')

    def test_invalid_arguments_rejected_without_state_change(self):
        robot = Robot(make_task())
        requests = [('move_base', [float('nan'),0,0,1], {}),
                    ('move_base', [10**1000,0,0,1], {}),
                    ('move_base', [.3,.2,0,1], {}),
                    ('move_base', [0,0,.41,1], {}),
                    ('hold', [0], {}), ('hold', [True], {}),
                    ('reach_right', [[.3,.2]], {}),
                    ('reach_right', [[90,0,0]], {}),
                    ('walk_to', [[1.3,0]], {}),
                    ('turn_to', [math.pi], {}),
                    ('stop', [], {'timeout': 21}),
                    ('hold', [], {'bogus': 1}), ('__init__', [], {})]
        before = robot.observe()
        for method,args,kwargs in requests:
            with self.subTest(method=method,args=args):
                result = robot.dispatch(method,args,kwargs,robot.episode_id)
                self.assertEqual(result['status'], 'rejected')
                self.assertTrue(result['errors'])
                self.assertEqual(robot.observe(),before)
        json.dumps(robot.trace, allow_nan=False)

    def test_terminal_dispatch_observe_keeps_snapshot_schema(self):
        task = make_task('reach',2)
        robot = Robot(task)
        initial_keys = set(robot.dispatch('observe',[],{},robot.episode_id))
        robot.reach_right(task.target_position)
        robot.policy_exit()
        snapshot = robot.dispatch('observe',[],{},robot.episode_id)
        self.assertEqual(snapshot['episode_status'], 'success')
        self.assertEqual(set(snapshot), initial_keys)
        self.assertEqual(robot.hold(1)['status'], 'cancelled')
        self.assertEqual(robot.dispatch('observe',[],{},robot.episode_id), snapshot)

    def test_bounded_trace_preserves_initial_and_counts_dropped_records(self):
        robot = Robot(make_task())
        robot.TRACE_LIMIT = 4
        initial = robot.trace[0]
        # One request, five state samples and one result, plus initial state.
        robot.hold(.1)
        self.assertEqual(len(robot.trace), 4)
        self.assertEqual(robot.trace[0], initial)
        self.assertEqual(robot.summary()['trace_truncated'], 4)
        robot.finish('policy_exited_early')
        self.assertEqual(robot.summary()['trace_truncated'], 5)
        self.assertEqual(robot.trace[-1]['type'], 'terminal')

    def test_rotated_body_command_and_expiry_idle(self):
        backend = MockBackend(make_task())
        backend.state = dataclasses.replace(backend.state,pelvis_yaw=math.pi/2)
        robot = Robot(make_task(), backend)
        result = robot.move_base(.2,0,0,1)
        self.assertEqual(result['status'], 'completed')
        self.assertAlmostEqual(result['observation']['pelvis_position'][0],0)
        self.assertAlmostEqual(result['observation']['pelvis_position'][1],.2)
        robot.hold(.2)
        self.assertAlmostEqual(robot.observe()['pelvis_position'][1], .2)
        self.assertEqual(robot.observe()['planar_velocity'], [0.,0.])

    def test_stop_requires_settled_dwell(self):
        robot = Robot(make_task())
        robot.move_base(.2,0,0,.2)
        result = robot.stop(timeout=.2)
        self.assertEqual(result['status'], 'timed_out')
        self.assertEqual(result['observation']['planar_velocity'], [0.,0.])
        result = robot.stop()
        self.assertEqual(result['status'], 'completed')
        self.assertGreaterEqual(result['end_time']-result['start_time'], .5)

    def test_stale_episode_and_terminal_requests_cannot_move(self):
        robot = Robot(make_task())
        before = robot.observe()
        result = robot.dispatch('hold',[1],{},'old-episode')
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(robot.observe(),before)
        robot.finish('policy_error')
        terminal = robot.observe()
        self.assertEqual(robot.hold(1)['status'],'cancelled')
        self.assertEqual(robot.observe(),terminal)
        self.assertEqual(robot.terminal_reason,'policy_error')

    def test_goal_tools_compose_and_evaluator_owns_success(self):
        for seed in range(6):
            with self.subTest(seed=seed):
                task = make_task(seed=seed)
                robot = Robot(task)
                walk = robot.walk_to(task.target_position[:2])
                self.assertIn(walk['status'], ('completed','cancelled'))
                if not robot.terminal_reason:
                    robot.turn_to(task.target_yaw)
                if not robot.terminal_reason:
                    robot.policy_exit()
                self.assertTrue(robot.summary()['success'])
                self.assertEqual(robot.summary()['backend'],'mock')
                self.assertLessEqual(robot.summary()['metrics']['position_error'], .10)

    def test_turn_then_walk_handles_rotated_motion(self):
        task = dataclasses.replace(make_task(), target_position=(.6,.3,0.),target_yaw=1.)
        robot = Robot(task)
        self.assertEqual(robot.turn_to(1.)['status'], 'completed')
        robot.walk_to([.6,.3])
        robot.policy_exit()
        self.assertTrue(robot.summary()['success'])

    def test_timeout_stops_pursuit_and_reach_returns_before_task_dwell(self):
        robot = Robot(make_task('reach'))
        result = robot.reach_right([.45,-.1,1.], timeout=.02)
        self.assertEqual(result['status'],'timed_out')
        wrist = robot.observe()['right_wrist_position']
        robot.hold(.2)
        self.assertEqual(robot.observe()['right_wrist_position'],wrist)
        task = make_task('reach',2)
        robot = Robot(task)
        result = robot.reach_right(task.target_position)
        self.assertEqual(result['status'],'completed')
        self.assertFalse(robot.summary()['success'])
        robot.policy_exit()
        self.assertTrue(robot.summary()['success'])

    def test_early_exit_and_deadline_cancellation_are_distinct(self):
        robot = Robot(make_task())
        robot.policy_exit()
        self.assertEqual(robot.terminal_reason,'policy_exited_early')
        self.assertAlmostEqual(robot.observe()['sim_time'],1.)
        task = dataclasses.replace(make_task(),deadline=.1)
        robot = Robot(task)
        result = robot.hold(.1)
        self.assertEqual(result['status'],'cancelled')
        self.assertEqual(result['reason'],'timeout')
        self.assertAlmostEqual(result['end_time'],.1)

    def test_state_loss_is_infrastructure_failure(self):
        robot = Robot(make_task())
        robot.backend.state = dataclasses.replace(robot.backend.state,state_age_s=.3)
        with self.assertRaises(StateUnavailable):
            robot.observe()
        self.assertEqual(robot.terminal_reason,'state_stale')
        robot.finish('policy_error')
        self.assertEqual(robot.terminal_reason,'state_stale')

    def test_goal_measurements_refer_to_requested_goal(self):
        robot = Robot(make_task())
        result = robot.walk_to([.3,0])
        self.assertEqual(result['status'], 'completed')
        self.assertLessEqual(result['measurements']['goal_position_error'], .10)
        self.assertGreater(result['measurements']['position_error'], .3)
        result = robot.turn_to(.5)
        self.assertLessEqual(result['measurements']['goal_heading_error'], math.radians(10))
        self.assertAlmostEqual(result['measurements']['command_base_displacement'], 0.)

    def test_missing_state_during_reach_cancels_without_cleanup_crash(self):
        class LostStateBackend(MockBackend):
            def step(self, dt):
                self.state = None
                return None
        robot = Robot(make_task('reach'),LostStateBackend(make_task('reach')))
        result = robot.reach_right([.4,-.2,.9])
        self.assertIn(result['status'], ('cancelled','backend_error'))
        self.assertEqual(result['reason'],'backend_failed')
        self.assertFalse(robot.summary()['success'])
        json.dumps(result, allow_nan=False)
        result = robot.dispatch('hold',[1],{},robot.episode_id)
        self.assertEqual(result['status'],'cancelled')

    def test_no_progress_is_reported_before_goal_timeout(self):
        class StuckBackend(MockBackend):
            def step(self, dt):
                self.state = dataclasses.replace(self.state,sim_time=self.state.sim_time+dt,
                                                 sequence=self.state.sequence+1)
                return self.state
        robot = Robot(make_task(),StuckBackend(make_task()))
        result = robot.walk_to([.8,0])
        self.assertEqual(result['status'],'timed_out')
        self.assertEqual(result['reason'],'no_progress')
        self.assertLess(result['end_time'],5.)


if __name__ == '__main__':
    unittest.main()
