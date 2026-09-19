import copy
import unittest

from g1cap.arena_control import BoxControl
from g1cap.arena_observation import loaded_contact_summary
from g1cap.arena_retreat import LoadedRetreat
from test_arena_retreat import state


class WristMotion:
    def command(self, measured, root_pos, root_xyzw, previous, elapsed):
        result=list(previous);result[0]+=.001
        return result


def sample(step):
    o=state(step);o['loaded_contacts']['maximum_hand_N']=11.
    return o


class CarryPreparationTests(unittest.TestCase):
    def control(self):
        c=BoxControl([0.]*46+[.75,0.,0.,0.],lambda o:None,
                     lambda *args,**kw:WristMotion())
        for i in range(51):c.update(sample(i))
        return c

    def test_preparation_keeps_navigation_zero_until_stable_then_moves(self):
        c=self.control();c.start('retreat_with_box',{'distance_m':.5},sample(50))
        self.assertEqual(c.phase,'prepare_carry')
        for i in range(50,200):
            self.assertEqual(c.command(sample(i))[43:46],[0.,0.,0.])
            c.update(sample(i+1))
        self.assertEqual(c.phase,'loaded_retreat')
        self.assertLess(c.command(sample(200))[43],0.)

    def test_transient_force_peak_is_not_hidden_by_last_substep(self):
        c=self.control();c.start('retreat_with_box',{'distance_m':.5},sample(50))
        sent=c.command(sample(50))
        rows=[dict(hand_forces_N={'left':10.,'right':10.},foot_upward_N={'left':150.,'right':150.}) for _ in range(4)]
        rows[1]['hand_forces_N']['left']=25.1
        o=sample(51);o['loaded_contacts']=loaded_contact_summary(rows)
        result=c.update(o)
        self.assertIsNotNone(result)
        self.assertEqual(result['reason'],'preparation_force_limit')
        self.assertEqual(c.command(o),sent)

    def test_missing_invalid_or_excessive_force_peak_rejects_before_motion(self):
        for value in (None,float('nan'),float('inf'),True,-1.,25.1):
            c=self.control();o=sample(50);o['loaded_contacts']['maximum_hand_N']=value
            before=copy.deepcopy(c.last_action)
            self.assertEqual(c.start('retreat_with_box',{'distance_m':.5},o)['status'],'rejected')
            self.assertEqual(c.last_action,before)

    def test_preparation_timeout_and_cancel_keep_loaded_targets(self):
        c=self.control();c.start('retreat_with_box',{'distance_m':.5},sample(50))
        for i in range(50,300):
            c.command(sample(i));o=sample(i+1);o['clearance']=.04;c.update(o)
        self.assertIsNotNone(c.result)
        self.assertEqual(c.result['reason'],'preparation_did_not_settle')
        c=self.control();c.start('retreat_with_box',{'distance_m':.5},sample(50));sent=c.command(sample(50))
        c.cancel('worker_exit',sample(50));self.assertEqual(c.command(sample(50)),sent)

    def test_prepared_grasp_is_reused_after_motion_stop(self):
        c=self.control();c.start('retreat_with_box',{'distance_m':.5},sample(50))
        for i in range(50,200):c.command(sample(i));c.update(sample(i+1))
        c.cancel('worker_exit',sample(200))
        for i in range(201,252):c.update(sample(i))
        result=c.start('retreat_with_box',{'distance_m':.3},sample(251))
        self.assertEqual(result['status'],'running')
        self.assertEqual(c.phase,'loaded_retreat')
        self.assertEqual(c.command(sample(251))[:43],c.preparation.last_action[:43])

    def test_incomplete_preparation_cannot_grant_another_squeeze(self):
        c=self.control();c.start('retreat_with_box',{'distance_m':.5},sample(50))
        c.command(sample(50));c.update(sample(51));c.cancel('worker_exit',sample(51))
        for i in range(52,103):c.update(sample(i))
        self.assertEqual(c.start('retreat_with_box',{'distance_m':.3},sample(102))['reason'],
                         'grasp_preparation_not_verified')

    def test_reacquisition_starts_a_new_grasp(self):
        c=self.control();c.start('retreat_with_box',{'distance_m':.5},sample(50))
        for i in range(50,200):c.command(sample(i));c.update(sample(i+1))
        c.cancel('worker_exit',sample(200))
        c.start('pickup_box',{'object_id':'brown_box'},sample(200))
        self.assertFalse(c.grasp_prepared);self.assertIsNone(c.preparation)

    def test_stop_at_requested_center_not_lower_acceptance_edge(self):
        c=LoadedRetreat(state(0),[0.]*50,.5)
        c.update(state(1,.48));self.assertEqual(c.phase,'loaded_retreat')
        c.update(state(2,.5));self.assertEqual(c.phase,'settle_loaded')
        for i in range(3,53):self.assertIsNone(c.update(state(i,.5)))
        self.assertEqual(c.update(state(53,.5)),('completed','loaded_retreat_and_hold'))
