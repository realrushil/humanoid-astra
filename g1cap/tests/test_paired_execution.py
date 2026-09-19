"""Measured bilateral execution, holding, feedback and failure contracts."""
import copy
import unittest
from concurrent.futures import Future
from unittest.mock import patch
from test_stationary_task import state
from test_stationary_tools import Clock, Publisher
from g1cap.toolkit.arm_planner import UPPER_BODY
from g1cap.toolkit.stationary import StationaryTools

GOALS={s:dict(position=[.4,y,.9],quaternion=[1.,0.,0.,0.]) for s,y in (('left',.2),('right',-.2))}

class Planner:
    def __init__(self,reject_second=False):self.calls=0;self.reject_second=reject_second
    def plan(self,sample,goals):
        self.calls+=1
        if self.reject_second and self.calls==2:return dict(status='rejected',reason='path_model_collision',joint_path=[])
        return dict(status='planned',reason='paired_local_ik',joint_path=[[0.]*14],errors={})

class ImmediatePool:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def submit(self,fn,*args):
        future=Future();future.set_result(fn(*args));return future

class PairedPublisher(Publisher):
    def __init__(self,clock,planner,*,moves=True,bad_orientation=False):
        super().__init__({});self.clock=clock;self.planner=planner;self.moves=moves;self.bad_orientation=bad_orientation;self.changes={}
    def latest_state(self):
        moved=self.moves and self.planner.calls>=2
        raw=state(round(self.clock.now,8),pelvis_yaw=0.,body_joint_names=list(UPPER_BODY),body_joint_positions=[0.]*17,
                  joint_names=list(UPPER_BODY),joint_positions=[0.]*17,objects={})
        for side,goal in GOALS.items():
            raw[side+'_wrist_position']=[.4 if moved else .3,goal['position'][1],.9]
            raw[side+'_wrist_quaternion_wxyz']=[0.,1.,0.,0.] if self.bad_orientation else [1.,0.,0.,0.]
            raw[side+'_wrist_velocity_world']=[0.,0.,0.]
        raw.update(self.changes)
        if hasattr(self,'transform'):self.transform(raw)
        return raw

class PairedExecutionTests(unittest.TestCase):
    def exercise(self,*,moves=True,bad_orientation=False,reject_second=False,changes=None):
        clock=Clock();planner=Planner(reject_second);publisher=PairedPublisher(clock,planner,moves=moves,bad_orientation=bad_orientation)
        if changes:publisher.changes.update(changes)
        with patch('g1cap.toolkit.stationary.time',clock),patch('g1cap.toolkit.stationary.ThreadPoolExecutor',return_value=ImmediatePool()):
            tools=StationaryTools(publisher,None,paired_planner=planner)
            result=tools.reach_hands(copy.deepcopy(GOALS),timeout=10.)
        return result,tools,publisher,planner,clock

    def test_feedback_replans_and_requires_both_measured_poses(self):
        result,tools,publisher,planner,clock=self.exercise()
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(result['replans'],1)
        self.assertGreaterEqual(result['elapsed'],3.5)
        self.assertEqual(planner.calls,2)
        self.assertIsNotNone(tools.positions)
        self.assertTrue(all(e['position_m']<=.02 for e in result['hand_errors'].values()))
        # A held target remains a pose promise; moving one hand out of tolerance
        # cannot be reported as a successful hold merely because it is stationary.
        publisher.changes['left_wrist_position']=[.5,.2,.9]
        with patch('g1cap.toolkit.stationary.time',clock):
            held=tools.hold(.5)
        self.assertEqual(held['status'],'timed_out')
        self.assertEqual(publisher.commands[-1],{'idle':True})

    def test_planning_success_cannot_substitute_for_position_or_orientation(self):
        for kwargs in (dict(moves=False),dict(bad_orientation=True)):
            result,tools,publisher,planner,_=self.exercise(**kwargs)
            self.assertEqual(result['status'],'timed_out',result)
            self.assertEqual(planner.calls,3)
            self.assertIsNone(tools.positions)
            self.assertEqual(publisher.commands[-1],{'idle':True})

    def test_replan_collision_stops_partial_execution(self):
        result,_,publisher,_,_=self.exercise(reject_second=True)
        self.assertEqual(result['status'],'failed',result)
        self.assertEqual(result['reason'],'replan_path_model_collision')
        self.assertTrue(any(c.get('positions') for c in publisher.commands))
        self.assertEqual(publisher.commands[-1],{'idle':True})

    def test_stale_or_missing_left_pose_cannot_issue_commands(self):
        for bad in ({'state_age_s':.5},{'left_wrist_quaternion_wxyz':None}):
            result,_,publisher,_,_=self.exercise(changes=bad)
            self.assertEqual(result['status'],'failed',result)
            self.assertFalse(any(c.get('positions') for c in publisher.commands))

    def test_invalid_goal_preserves_existing_reference(self):
        clock=Clock();planner=Planner();publisher=PairedPublisher(clock,planner)
        tools=StationaryTools(publisher,None,paired_planner=planner)
        for bad in ({},dict(left=GOALS['left']),dict(GOALS,right=dict(position=[.4,0,.9],quaternion=[0,0,0,0]))):
            with self.assertRaises(ValueError):tools.reach_hands(bad)
        self.assertEqual(publisher.commands,[])

    def test_gap_support_loss_and_moved_object_fail_during_execution(self):
        def gap(raw):
            if raw['sim_time']>=.2:
                raw['sim_time']+=.2;raw['sequence']+=40
        def support(raw):
            if raw['sim_time']>=.2:raw['foot_normal_forces']['left']=0.
        def rotate_object(raw):
            raw['objects']={'box':dict(position_world=[1,0,1],quaternion_wxyz=[1,0,0,0])}
            if raw['sim_time']>=.2:raw['objects']['box']['quaternion_wxyz']=[0,0,0,1]
        for transform,reason in ((gap,'state_gap'),(support,'support_lost'),(rotate_object,'plan_object_changed')):
            clock=Clock();planner=Planner();publisher=PairedPublisher(clock,planner);publisher.transform=transform
            with patch('g1cap.toolkit.stationary.time',clock),patch('g1cap.toolkit.stationary.ThreadPoolExecutor',return_value=ImmediatePool()):
                tools=StationaryTools(publisher,None,paired_planner=planner)
                result=tools.reach_hands(copy.deepcopy(GOALS),timeout=10.)
            self.assertEqual(result['status'],'failed',result)
            self.assertEqual(result['reason'],reason)
            self.assertEqual(publisher.commands[-1],{'idle':True})

    def test_episode_termination_cancels_in_flight_pose_request(self):
        clock=Clock();planner=Planner();publisher=PairedPublisher(clock,planner)
        publisher.transform=lambda raw:raw.update(episode_status='timeout' if raw['sim_time']>=.2 else 'running')
        with patch('g1cap.toolkit.stationary.time',clock),patch('g1cap.toolkit.stationary.ThreadPoolExecutor',return_value=ImmediatePool()):
            tools=StationaryTools(publisher,None,paired_planner=planner)
            result=tools.reach_hands(copy.deepcopy(GOALS),timeout=10.)
        self.assertEqual(result['status'],'cancelled',result)
        self.assertEqual(result['reason'],'timeout')
        self.assertEqual(result['holding'],'nominal_idle')
        self.assertLess(result['elapsed'],.3)

    def test_failed_new_request_cannot_leave_an_old_pose_available_to_hold(self):
        result,tools,publisher,_,clock=self.exercise()
        self.assertEqual(result['status'],'completed')
        publisher.changes['state_age_s']=.5
        with patch('g1cap.toolkit.stationary.time',clock):
            result=tools.reach_hands(copy.deepcopy(GOALS))
        self.assertEqual(result['status'],'failed')
        self.assertIsNone(tools.positions)
        self.assertIsNone(tools.pose_goals)
