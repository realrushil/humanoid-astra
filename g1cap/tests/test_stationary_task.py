import importlib
import unittest


def state(t, **changes):
    raw = dict(sim_time=t, sequence=round(t*200), state_age_s=0., ready=True,
               backend_ok=True, dds_domain=1, dds_interface='lo', no_support=True,
               pelvis_position=[0.,0.,.7], planar_velocity=[0.,0.], yaw_rate=0., tilt=0.,
               right_wrist_position=[.3,-.2,.9], right_wrist_velocity_world=[0.,0.,0.],
               foot_normal_forces={'left':100.,'right':100.}, forbidden_contacts=[])
    raw.update(changes)
    return raw


class StationaryTaskTests(unittest.TestCase):
    def evaluator(self):
        module = importlib.import_module('g1cap.stationary_task')
        task = module.StationaryTask(stages=({'target_world':[.3,-.2,.9], 'height_band':[.67,.73]},), deadline=5.)
        return module.StationaryEvaluator(task, state(0.))

    def test_interrupted_goal_dwell_does_not_accumulate(self):
        evaluator = self.evaluator()
        for i in range(1, 5):
            evaluator.update(state(i*.1))
        evaluator.update(state(.5, right_wrist_position=[.5,-.2,.9]))
        for i in range(6, 11):
            self.assertIsNone(evaluator.update(state(i*.1)))
        self.assertEqual(evaluator.update(state(1.1)), 'success')

    def test_height_and_speed_must_hold_with_wrist_target(self):
        for bad in [dict(pelvis_position=[0.,0.,.8]), dict(right_wrist_velocity_world=[.1,0.,0.])]:
            evaluator = self.evaluator()
            for i in range(1, 12):
                self.assertIsNone(evaluator.update(state(i*.1, **bad)))
            self.assertEqual(evaluator.stage_index, 0)

    def test_collapse_contact_and_missing_evidence_cannot_succeed(self):
        for bad, reason in [(dict(pelvis_position=[0.,0.,.3]), 'unsafe_posture'),
                            (dict(forbidden_contacts=[{'body':'hand'}]), 'forbidden_contact'),
                            (dict(state_age_s=.5), 'state_stale')]:
            evaluator = self.evaluator()
            self.assertEqual(evaluator.update(state(.1, **bad)), reason)
        evaluator = self.evaluator()
        self.assertEqual(evaluator.update(state(.3)), 'state_gap')

    def test_support_loss_never_earns_success(self):
        evaluator = self.evaluator()
        for i in range(1,4):
            evaluator.update(state(i*.1, foot_normal_forces={'left':0.,'right':100.}))
        self.assertEqual(evaluator.update(state(.4, foot_normal_forces={'left':0.,'right':100.})), 'support_lost')

    def test_fault_after_goal_is_not_hidden_by_earlier_success(self):
        evaluator = self.evaluator()
        for i in range(1,6):
            evaluator.update(state(i*.1))
        self.assertEqual(evaluator.terminal_reason, 'success')
        self.assertEqual(evaluator.update(state(.6,pelvis_position=[0.,0.,.3])), 'unsafe_posture')
