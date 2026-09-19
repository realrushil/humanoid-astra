"""Paired planning is read-only; execution uses the ordinary session tool boundary."""
import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from test_paired_execution import GOALS,Clock,Planner,PairedPublisher
from g1cap.session_tools import SessionTools

class PairedApiTests(unittest.TestCase):
    def test_planning_and_unsettled_execution_do_not_change_commands(self):
        planner=Planner();publisher=PairedPublisher(Clock(),planner)
        raw=publisher.latest_state();raw.update(model_path='unused',settled_for=0.)
        session=SimpleNamespace(observe=lambda:copy.deepcopy(raw),idle=publisher.idle)
        tools=SessionTools(session,None)
        with patch('g1cap.toolkit.bimanual.BimanualPlanner',return_value=planner):
            result=tools.check_hands(copy.deepcopy(GOALS))
            self.assertEqual(result['status'],'planned')
            self.assertEqual(result['sequence'],raw['sequence'])
            result=tools.reach_hands(copy.deepcopy(GOALS))
            self.assertEqual(result['reason'],'base_not_settled')
        self.assertEqual(publisher.commands,[])

    def test_new_methods_are_exposed_and_documented(self):
        from g1cap.session_tools import session_api
        self.assertTrue({'check_hands','reach_hands'}<=SessionTools.METHODS)
        self.assertIn('reach_hands',session_api({}))

    def test_independent_stationary_evaluator_requires_both_orientations(self):
        from g1cap.stationary_task import StationaryTask,StationaryEvaluator
        planner=Planner();planner.calls=2;clock=Clock();publisher=PairedPublisher(clock,planner)
        initial=publisher.latest_state()
        task=StationaryTask(stages=({'target_world':None,'height_band':[.55,.9],'hand_goals':copy.deepcopy(GOALS)},),deadline=5.)
        evaluator=StationaryEvaluator(task,initial)
        for i in range(1,8):
            clock.now=i*.1;raw=publisher.latest_state();raw['left_wrist_quaternion_wxyz']=[0,1,0,0]
            self.assertIsNone(evaluator.update(raw))
        for i in range(8,14):
            clock.now=i*.1;result=evaluator.update(publisher.latest_state())
        self.assertEqual(result,'success')
