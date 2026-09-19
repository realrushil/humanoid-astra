"""Synthetic scoring contracts; these are not physical walking witnesses."""
import importlib.util
import math
import unittest
from test_session import sample


class MobilityTests(unittest.TestCase):
    def make(self, waypoints=((3., 0.),), final_yaw=None):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.mobility_task'))
        from g1cap.mobility_task import MobilityTask, MobilityEvaluator
        return MobilityEvaluator(MobilityTask(waypoints, final_yaw=final_yaw), sample(0.))

    def dwell(self, e, xy, yaw=0., speed=0.):
        for _ in range(7):
            t=e.previous['sim_time']+.05
            e.update(sample(t,pelvis_position=[*xy,.7],pelvis_yaw=yaw,
                            planar_velocity=[speed,0.]))

    def test_distance_is_not_success_and_long_workspace_is_valid(self):
        e=self.make()
        for xy in ((2.,0.),(0.,0.),(2.,0.)):
            self.dwell(e,xy)
        self.assertGreater(e.path_length,3.)
        self.assertIsNone(e.terminal_reason)
        self.assertEqual(e.waypoint_index,0)

    def test_order_progress_persists_and_final_heading_must_be_current(self):
        e=self.make(((2.,0.),(0.,0.)),math.pi/2)
        self.dwell(e,(0.,0.))
        self.assertEqual(e.waypoint_index,0)
        for _ in range(2): self.dwell(e,(2.,0.))
        self.assertEqual(e.waypoint_index,1)
        self.dwell(e,(1.,0.))  # Consequence between revisions is retained.
        for _ in range(2): self.dwell(e,(0.,0.))
        self.assertIsNone(e.terminal_reason)
        for _ in range(2): self.dwell(e,(0.,0.),math.pi/2)
        self.assertEqual(e.terminal_reason,'success')

    def test_corridor_exit_is_terminal(self):
        e=self.make()
        self.dwell(e,(1.,.7))
        self.assertEqual(e.terminal_reason,'corridor_exceeded')

    def test_missing_heading_fails_closed(self):
        e=self.make(final_yaw=0.)
        raw=sample(.05);raw.pop('pelvis_yaw')
        self.assertEqual(e.update(raw),'invalid_state')

    def test_navigation_uses_task_corridor_and_keeps_legacy_limit(self):
        from types import SimpleNamespace
        from g1cap.session_tools import SessionTools
        from g1cap.session_task import SessionTask
        e=self.make()
        tools=SessionTools(SimpleNamespace(task=e.task,initial=sample(0.)),None)
        # No dynamics here: exercise request validation before the motor boundary.
        tools._navigate=lambda *args: args
        self.assertEqual(tools.walk_to([3.,0.])[0],'walk_to')
        with self.assertRaises(ValueError): tools.walk_to([2.,1.])
        tools.session.task=SessionTask([.2,0.],[.5,0.,.8])
        with self.assertRaises(ValueError): tools.walk_to([3.,0.])

    def test_arrival_requires_stop_and_fault_wins(self):
        e=self.make()
        for _ in range(2): self.dwell(e,(3.,0.),speed=.1)
        self.assertIsNone(e.terminal_reason)
        e.update(sample(e.previous['sim_time']+.05,pelvis_position=[3.,0.,.3]))
        self.assertNotEqual(e.terminal_reason,'success')

    def test_recipe_and_api_describe_route(self):
        from g1cap.session_runtime import task_from_recipe
        from g1cap.session_tools import session_api
        task=task_from_recipe({'task':'mobility','waypoints':[[6.,0.]]})
        self.assertEqual(task.to_dict()['waypoints'],[[6.,0.]])
        self.assertIn('waypoint_index',session_api(task.to_dict()))
