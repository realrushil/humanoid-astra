import copy
import math
import unittest
from types import SimpleNamespace

from g1cap.arena_control import BoxControl
from g1cap.arena_observation import loaded_contact_summary
from g1cap.arena_retreat import LoadedRetreat
from g1cap.arena_task import ArenaRetreatTask
from test_arena_control import observation


def state(step=0,travel=0):
    o=observation(step*.02)
    o['root_pos'][0]-=travel;o['box_pos'][0]-=travel
    o.update(approach_clearance_m=.10,loaded_contacts=dict(samples=4,minimum_hand_N=10.,maximum_hand_N=10.,minimum_total_foot_N=300.))
    return o


class RetreatTests(unittest.TestCase):
    def test_retained_box_pose_change_is_diagnostic_but_clearance_loss_fails(self):
        anchor=state(50)
        controller=LoadedRetreat(anchor,[0.]*46+[.75,0.,0.,0.],.5)
        changed=state(51)
        # Arm/body motion can change this proxy while contact and height remain.
        changed['box_pos'][1]+=.05
        self.assertIsNone(controller.update(changed))
        self.assertAlmostEqual(controller.measurements(changed)['box_pelvis_change_m'],.05)
        changed=copy.deepcopy(changed);changed['time']+=.02;changed['clearance']=.02
        self.assertEqual(controller.update(changed),('failed','grasp_or_height_lost'))

    def setup_control(self):
        motion=SimpleNamespace(command=lambda measured,root,quat,previous,elapsed:list(previous))
        c=BoxControl([0.]*46+[.75,0.,0.,0.],lambda o:None,lambda *a,**kw:motion)
        for i in range(51):c.update(state(i))
        return c

    def test_admission_and_invalid_requests_preserve_references(self):
        c=self.setup_control();before=copy.deepcopy(c.last_action)
        for value in [None,True,.1,.6,float('nan')]:
            self.assertEqual(c.start('retreat_with_box',{'distance_m':value},state(50))['status'],'rejected')
            self.assertEqual(c.last_action,before)
        o=state(50);o['root_quat']=[math.cos(.1),0,0,math.sin(.1)]
        self.assertEqual(c.start('retreat_with_box',{'distance_m':.5},o)['reason'],'unsupported_heading')

    def test_transient_substep_loss_is_seen_and_cancel_holds_references(self):
        c=self.setup_control();c.start('retreat_with_box',{'distance_m':.5},state(50))
        for i in range(50,200):c.command(state(i));c.update(state(i+1))
        original=c.command(state(200));self.assertLess(original[43],0.)
        contacts=[dict(hand_forces_N={'left':10.,'right':10.},foot_upward_N={'left':150.,'right':150.}) for _ in range(4)]
        contacts[1]['hand_forces_N']['left']=0.
        o=state(201);o['loaded_contacts']=loaded_contact_summary(contacts)
        result=c.update(o)
        self.assertEqual(result['reason'],'substep_hand_contact_lost')
        stopped=c.command(o);self.assertEqual(stopped[43:46],[0.,0.,0.])
        self.assertEqual(stopped[:43],original[:43]);self.assertEqual(stopped[46:],original[46:])

    def test_missing_substep_data_and_cancelled_operation(self):
        c=self.setup_control();o=state(50);del o['loaded_contacts']
        self.assertEqual(c.start('retreat_with_box',{'distance_m':.5},o)['reason'],'missing_loaded_contacts')
        c.start('retreat_with_box',{'distance_m':.5},state(50));c.command(state(50))
        self.assertEqual(c.cancel('worker_exit',state(50))['status'],'cancelled')
        self.assertEqual(c.command(state(50))[43:46],[0.,0.,0.])

    def test_task_requires_box_motion_and_full_hold_and_keeps_late_fault(self):
        score=ArenaRetreatTask(.5)
        for i in range(51):score.update(state(i))
        for i in range(51,80):score.update(state(i,(i-50)*.49/29))
        self.assertFalse(score.metrics()['success'])
        for i in range(80,131):score.update(state(i,.49))
        self.assertTrue(score.metrics()['success'])
        o=state(131,.49);o['loaded_contacts']['minimum_hand_N']=0.;score.update(o)
        self.assertFalse(score.metrics()['success'])
        for i in range(132,200):score.update(state(i,.49))
        self.assertFalse(score.metrics()['success'])
        other=ArenaRetreatTask(.5)
        for i in range(51):other.update(state(i))
        for i in range(51,151):
            o=state(i,.49);o['box_pos']=state(i)['box_pos'];other.update(o)
        self.assertFalse(other.metrics()['success'])
