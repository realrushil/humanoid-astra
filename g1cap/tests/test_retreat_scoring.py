"""Task-level counterexamples, built from synthetic observations, not physics."""
import unittest
from g1cap.arena_task import ArenaRetreatTask
from test_arena_retreat import state


def lifted(step,travel=0.,lift=0.):
    o=state(step,travel)
    o['box_pos'][2]=.12+lift  # source top−.03 + cube half-height .10 + initial clearance .05
    o['clearance']=.05+lift
    return o


class RetreatScoringTests(unittest.TestCase):
    def episode(self,fault=None):
        score=ArenaRetreatTask(.5)
        for i in range(51):score.update(lifted(i))
        for i in range(51,301):
            o=lifted(i,lift=(i-50)*.0002)  # deliberate1 cm/s rise by5 cm
            if i==175 and fault:fault(o)
            score.update(o)
        for i in range(301,551):score.update(lifted(i,travel=(i-300)*.5/250,lift=.05))
        for i in range(551,602):score.update(lifted(i,travel=.5,lift=.05))
        return score

    def test_intentional_lift_preserves_original_progress_anchor_and_can_finish(self):
        score=self.episode()
        self.assertTrue(score.metrics()['success'])
        self.assertEqual(score.anchor['time'],1.)
        self.assertAlmostEqual(score.metrics()['box_retreat_m'],.5)
        self.assertGreaterEqual(score.metrics()['maximum_box_pelvis_change_m'],.049999)
        self.assertEqual(score.metrics()['scoring_version'],'retreat_outcome_v2')

    def test_real_faults_during_lift_remain_failed_after_later_valid_transport(self):
        faults=[lambda o:o['loaded_contacts'].update(minimum_hand_N=0.),
                lambda o:o['loaded_contacts'].update(minimum_total_foot_N=0.),
                lambda o:o.update(clearance=.02),
                lambda o:o.update(robot_source_peak_N=6.),
                lambda o:o.update(tilt=.3)]
        for fault in faults:
            with self.subTest(fault=fault):
                score=self.episode(fault)
                self.assertFalse(score.metrics()['success'])
                self.assertIsNotNone(score.failure)

    def test_task_does_not_award_success_for_lift_without_transport(self):
        score=ArenaRetreatTask(.5)
        for i in range(400):score.update(lifted(i,lift=min(.05,max(0.,(i-50)*.0002))))
        self.assertFalse(score.metrics()['success'])
