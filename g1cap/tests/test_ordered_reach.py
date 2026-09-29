"""Ordered marker and measured-neutral contracts; synthetic states are not physics."""
import importlib.util
import tempfile
import time
import unittest

from test_session import Publisher
from test_stationary_task import state
from g1cap.toolkit.arm_planner import UPPER_BODY


LOWER = [.78, -.24, .70]
UPPER = [.78, -.24, .86]


def measured(t, *, wrist=None, joints=None, height=.70, speeds=None, **changes):
    raw = state(t, pelvis_position=[0., 0., height],
                right_wrist_position=list(wrist or [.3, -.2, .9]),
                body_joint_names=list(UPPER_BODY),
                body_joint_positions=list(joints or [0.] * len(UPPER_BODY)),
                body_joint_velocities=list(speeds or [0.] * len(UPPER_BODY)))
    raw.update(changes)
    return raw


class OrderedPublisher(Publisher):
    """Existing timed publisher fake plus the actual ordered-task observation fields."""
    def __init__(self):
        super().__init__()
        self.wrist = [.3, -.2, .9]

    def latest_state(self):
        raw = super().latest_state()
        raw.update(body_joint_names=list(UPPER_BODY),
                   body_joint_positions=[0.] * len(UPPER_BODY),
                   body_joint_velocities=[0.] * len(UPPER_BODY),
                   right_wrist_position=list(self.wrist))
        return raw


class OrderedReachTests(unittest.TestCase):
    def evaluator(self, initial=None, **task_fields):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.ordered_reach_task'),
                             'ordered reach task is missing')
        from g1cap.ordered_reach_task import OrderedReachTask, OrderedReachEvaluator
        from g1cap.scene import workstation_scene
        task = OrderedReachTask(workstation_scene(), **task_fields)
        return OrderedReachEvaluator(task, initial or measured(0.))

    def test_wrong_marker_does_not_complete_first_stage(self):
        evaluator = self.evaluator()
        for i in range(1, 9):
            evaluator.update(measured(i * .1, wrist=UPPER))
        self.assertEqual(evaluator.metrics['stage_index'], 0)
        self.assertIsNone(evaluator.terminal_reason)
        for i in range(9, 15):
            evaluator.update(measured(i * .1, wrist=LOWER))
        self.assertEqual(evaluator.metrics['stage_index'], 1)
        self.assertIsNone(evaluator.terminal_reason)

    def test_both_wrist_stages_then_current_neutral_dwell_are_required(self):
        evaluator = self.evaluator()
        for i in range(1, 7):
            evaluator.update(measured(i * .1, wrist=LOWER))
        self.assertEqual(evaluator.metrics['stage_index'], 1)
        for i in range(7, 13):
            evaluator.update(measured(i * .1, wrist=UPPER))
        self.assertEqual(evaluator.metrics['stage_index'], 2)
        for i in range(13, 21):
            evaluator.update(measured(i * .1, wrist=UPPER, joints=[.3] * len(UPPER_BODY)))
        self.assertIsNone(evaluator.terminal_reason)
        for i in range(21, 27):
            evaluator.update(measured(i * .1))
        self.assertEqual(evaluator.terminal_reason, 'success')
        self.assertEqual(evaluator.metrics['stage_index'], 3)

    def test_neutral_is_measured_startup_posture_not_zero_or_tool_return(self):
        baseline = [.2] * len(UPPER_BODY)
        evaluator = self.evaluator(measured(0., joints=baseline, height=.73))
        for i in range(1, 7): evaluator.update(measured(i * .1, wrist=LOWER, joints=baseline, height=.73))
        for i in range(7, 13): evaluator.update(measured(i * .1, wrist=UPPER, joints=baseline, height=.73))
        for i in range(13, 20): evaluator.update(measured(i * .1, height=.70))
        self.assertIsNone(evaluator.terminal_reason)
        for i in range(20, 26): evaluator.update(measured(i * .1, joints=baseline, height=.73))
        self.assertEqual(evaluator.terminal_reason, 'success')

    def test_interrupted_wrist_and_neutral_dwell_reset(self):
        evaluator = self.evaluator()
        for i in range(1, 5): evaluator.update(measured(i * .1, wrist=LOWER))
        evaluator.update(measured(.5, wrist=LOWER, right_wrist_velocity_world=[.1, 0., 0.]))
        for i in range(6, 11): evaluator.update(measured(i * .1, wrist=LOWER))
        self.assertEqual(evaluator.metrics['stage_index'], 0)
        evaluator.update(measured(1.1, wrist=LOWER))
        self.assertEqual(evaluator.metrics['stage_index'], 1)
        for i in range(12, 18): evaluator.update(measured(i * .1, wrist=UPPER))
        self.assertEqual(evaluator.metrics['stage_index'], 2)
        for i in range(18, 22): evaluator.update(measured(i * .1))
        evaluator.update(measured(2.2, speeds=[.2] + [0.] * (len(UPPER_BODY) - 1)))
        for i in range(23, 28): evaluator.update(measured(i * .1))
        self.assertIsNone(evaluator.terminal_reason)
        evaluator.update(measured(2.8))
        self.assertEqual(evaluator.terminal_reason, 'success')

    def test_existing_fault_and_deadline_rules_remain_terminal(self):
        for bad, reason in (({'forbidden_contacts': [{}]}, 'forbidden_contact'),
                            ({'pelvis_position': [0., 0., .3]}, 'unsafe_posture')):
            evaluator = self.evaluator()
            self.assertEqual(evaluator.update(measured(.1, **bad)), reason)
            self.assertEqual(evaluator.update(measured(.2, wrist=LOWER)), reason)
        evaluator = self.evaluator()
        self.assertEqual(evaluator.update(measured(.3)), 'state_gap')
        evaluator = self.evaluator(deadline=.6)
        for i in range(1, 7): evaluator.update(measured(i * .1, wrist=LOWER))
        self.assertEqual(evaluator.terminal_reason, 'timeout')

    def test_missed_between_sample_contact_cannot_advance_ordered_stage(self):
        evaluator = self.evaluator()
        contact = dict(sim_time=.075, contact=dict(body1='left_ankle_roll_link',
                         body2='blue_workstation', normal_force=140.8))
        self.assertEqual(evaluator.update(measured(.1, wrist=LOWER,
                         forbidden_contacts=[], first_forbidden_contact=contact)),
                         'forbidden_contact')
        self.assertEqual(evaluator.metrics['stage_index'], 0)
        self.assertEqual(evaluator.update(measured(.2, wrist=LOWER)), 'forbidden_contact')

    def test_recipe_and_public_goal_do_not_disclose_a_stance(self):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.ordered_reach_task'),
                             'ordered reach task is missing')
        from g1cap.session_runtime import task_from_recipe
        task = task_from_recipe({'task': 'ordered_reach', 'target_ids': ['blue_lower', 'blue_upper']})
        goal = task.to_dict()
        self.assertEqual(goal['target_ids'], ['blue_lower', 'blue_upper'])
        self.assertEqual(goal['name'], 'ordered_reach')
        self.assertEqual(goal['observation'], 'privileged_simulator_state')
        for key in ('approach_world_xy', 'ideal_base_pose', 'reference_code'):
            self.assertNotIn(key, goal)
        for ids in (['blue_lower'], ['blue_lower', 'blue_lower'], ['missing', 'blue_upper']):
            with self.assertRaises(ValueError):
                task_from_recipe({'task': 'ordered_reach', 'target_ids': ids})

    def test_agent_receives_ordered_stage_and_neutral_contract(self):
        from g1cap.session_tools import session_api
        goal = self.evaluator().task.to_dict()
        text = session_api(goal)
        self.assertIn("task['target_ids']", text)
        self.assertIn("task_progress']['stage_index'", text)
        self.assertIn("task_progress']['neutral_reference'", text)
        self.assertIn('No grasping is implemented', text)

    def test_session_supervisor_preserves_ordered_progress_and_clock(self):
        from g1cap.session import Session
        from g1cap.session_runtime import task_from_recipe
        task = task_from_recipe({'task': 'ordered_reach'})
        publisher = OrderedPublisher()
        with tempfile.TemporaryDirectory() as root:
            session = Session(publisher, None, task, root).start()
            try:
                first = session.observe()
                self.assertEqual(first['task_progress']['stage_index'], 0)
                publisher.wrist = LOWER
                end = time.monotonic() + 2.
                while time.monotonic() < end and session.status()['metrics']['stage_index'] == 0:
                    time.sleep(.02)
                second = session.observe()
                self.assertEqual(second['task_progress']['stage_index'], 1)
                self.assertEqual(second['task_progress']['target_id'], 'blue_upper')
                self.assertEqual(first['session_id'], second['session_id'])
                self.assertGreater(second['elapsed'], first['elapsed'])
            finally:
                session.close()


if __name__ == '__main__':
    unittest.main()
