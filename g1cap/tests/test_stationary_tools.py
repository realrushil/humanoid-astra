"""Contract tests for the trusted stationary executor, without a motor process."""
import importlib
import unittest
from test_stationary_task import state


class Publisher:
    def __init__(self, raw):
        self.raw = raw
        self.commands = []
    def latest_state(self):
        return dict(self.raw)
    def set_stationary(self, **fields):
        self.commands.append(fields)
    def idle(self):
        self.commands.append({'idle': True})


class StationaryToolsTests(unittest.TestCase):
    def tools(self, raw):
        module = importlib.import_module('g1cap.toolkit.stationary')
        publisher = Publisher(raw)
        return module.StationaryTools(publisher, None), publisher

    def test_stale_observation_rejects_before_command(self):
        tools,publisher = self.tools(state(0.))
        publisher.raw['state_age_s'] = .5
        result = tools.hold(.5)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'state_stale')
        self.assertFalse(any('positions' in c for c in publisher.commands))

    def test_invalid_posture_does_not_change_active_command(self):
        tools,publisher = self.tools(state(0.))
        for height in [.1, float('nan'), True]:
            with self.assertRaises(ValueError):
                tools.set_posture(height)
        self.assertEqual(publisher.commands, [])

    def test_planner_required_without_synthetic_reach_fallback(self):
        tools,publisher = self.tools(state(0.))
        result = tools.reach_right([.4,-.2,.9])
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['reason'], 'planner_unavailable')
        self.assertEqual(publisher.commands, [])

class Clock:
    def __init__(self): self.now=0.
    def monotonic(self): return self.now
    def sleep(self,duration): self.now += duration


class ReplayPublisher(Publisher):
    def __init__(self, clock):
        super().__init__(state(0.))
        self.clock=clock
    def latest_state(self):
        from g1cap.toolkit.arm_planner import UPPER_BODY
        raw=state(round(self.clock.now,8),body_joint_names=list(UPPER_BODY),
                  body_joint_positions=[0.]*17,pelvis_yaw=0.)
        return raw


class ExecutorCompletionTests(unittest.TestCase):
    def test_valid_plan_cannot_claim_success_when_measured_wrist_never_moves(self):
        from unittest.mock import patch
        from g1cap.toolkit.stationary import StationaryTools
        class Planner:
            def plan(self,sample,target):
                return dict(status='planned',reason='local_ik',joint_path=[[0.]*7],position_error=.001)
        clock=Clock();publisher=ReplayPublisher(clock)
        with patch('g1cap.toolkit.stationary.time',clock):
            tools=StationaryTools(publisher,Planner())
            result=tools.reach_right([.4,-.2,.9],timeout=1.)
        self.assertEqual(result['status'],'timed_out')
        self.assertEqual(result['reason'],'command_timeout')
        self.assertAlmostEqual(result['wrist_error'],.1)
        self.assertEqual(publisher.commands[-1],{'idle':True})

    def test_stalled_physics_has_bounded_wall_timeout_and_releases_reference(self):
        from unittest.mock import patch
        from g1cap.toolkit.stationary import StationaryTools
        clock=Clock();publisher=Publisher(state(0.))
        with patch('g1cap.toolkit.stationary.time',clock):
            tools=StationaryTools(publisher,None)
            result=tools.hold(.5)
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['reason'],'wall_timeout')
        self.assertLess(clock.now,4.)
        self.assertEqual(publisher.commands[-1],{'idle':True})

    def test_paused_caller_cannot_resurrect_expired_posture_with_hold(self):
        from unittest.mock import patch
        from g1cap.toolkit.stationary import StationaryTools
        clock=Clock();publisher=ReplayPublisher(clock)
        with patch('g1cap.toolkit.stationary.time',clock):
            tools=StationaryTools(publisher,None)
            self.assertEqual(tools.set_posture(.7)['status'],'completed')
            clock.sleep(1.)
            result=tools.hold(.5)
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(result['reason'],'reference_expired')
        self.assertEqual(publisher.commands[-1],{'idle':True})
