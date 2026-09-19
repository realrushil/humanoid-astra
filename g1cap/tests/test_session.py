"""Same-world session contracts; synthetic publisher is not a physics test."""
import copy
from contextlib import ExitStack
import importlib.util
import tempfile
import threading
import time
import unittest
from pathlib import Path
from test_stationary_task import state


def sample(t, **changes):
    raw = state(t, pelvis_yaw=0., left_wrist_position=[.3,.2,.9],
                right_wrist_quaternion_wxyz=[1.,0.,0.,0.])
    raw.update(changes)
    return raw


class SessionTaskTests(unittest.TestCase):
    def evaluator(self, deadline=5.):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.session_task'))
        from g1cap.session_task import SessionTask, SessionEvaluator
        task=SessionTask([.2,0.], [.5,-.2,.9], height=.7, deadline=deadline)
        return SessionEvaluator(task, sample(0.))

    def test_progress_and_path_survive_round_boundary_but_final_pose_is_current(self):
        e=self.evaluator()
        for i in range(1,9):
            e.update(sample(i*.1,pelvis_position=[.2,0.,.7]))
        self.assertTrue(e.approached)
        self.assertIsNone(e.terminal_reason)
        for i in range(9,17):
            e.update(sample(i*.1,pelvis_position=[.4,0.,.7],right_wrist_position=[.5,-.2,.9]))
        self.assertIsNone(e.terminal_reason)
        self.assertAlmostEqual(e.path_length,.4)
        for i in range(17,24):
            e.update(sample(i*.1,pelvis_position=[.2,0.,.7],right_wrist_position=[.5,-.2,.9]))
        self.assertEqual(e.terminal_reason,'success')
        self.assertEqual(e.start_time,0.)

    def test_deadline_and_fall_are_irreversible(self):
        e=self.evaluator(deadline=1.)
        for i in range(1,12): e.update(sample(i*.1))
        self.assertEqual(e.terminal_reason,'timeout')
        e.update(sample(1.2,pelvis_position=[.2,0.,.7],right_wrist_position=[.5,-.2,.9]))
        self.assertEqual(e.terminal_reason,'timeout')
        e=self.evaluator()
        e.update(sample(.1,pelvis_position=[0.,0.,.3]))
        e.update(sample(.2))
        self.assertEqual(e.terminal_reason,'unsafe_posture')

    def test_missing_history_fails_and_walking_does_not_require_two_feet(self):
        e=self.evaluator()
        for i in range(1,6):
            e.update(sample(i*.1,foot_normal_forces={'left':0.,'right':100.}),stationary=False)
        self.assertIsNone(e.terminal_reason)
        e.update(sample(.8))
        self.assertEqual(e.terminal_reason,'state_gap')


class Publisher:
    """Time advances independently of worker calls; no MuJoCo/controller claim."""
    lease_s=.25
    def __init__(self):
        self.start=time.monotonic()
        self.last_t=0.
        self.x=0.
        self.velocity=0.
        self.commands=[]
        self.bad=False
        self.raw=sample(0.)
        self.lock=threading.Lock()
    def latest_state(self):
        with self.lock:
            t=round((time.monotonic()-self.start)*50)/50
            self.x+=self.velocity*(t-self.last_t)
            self.last_t=t
            self.raw=sample(t,pelvis_position=[self.x,0.,.3 if self.bad else .7])
            return copy.deepcopy(self.raw)
    def samples_after(self,sequence):
        raw=self.latest_state()
        return [raw] if raw['sequence']>sequence else []
    def set_velocity(self,vx,vy,yaw_rate):
        self.velocity=vx
        self.commands.append((threading.current_thread().name,'velocity',vx))
    def set_stationary(self,**fields):
        self.velocity=0.
        self.commands.append((threading.current_thread().name,'stationary',fields))
    def idle(self):
        self.velocity=0.
        self.commands.append((threading.current_thread().name,'idle',None))


class LiveSessionTests(unittest.TestCase):
    def session(self,root,deadline=10.):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.session'))
        from g1cap.session import Session
        from g1cap.session_task import SessionTask
        publisher=Publisher()
        s=Session(publisher,None,SessionTask([.5,0.],[.8,-.2,.9],height=.7,deadline=deadline),root)
        s.start()
        return s,publisher

    def test_real_workers_continue_after_python_error_without_reset(self):
        with ExitStack() as stack:
            root=stack.enter_context(tempfile.TemporaryDirectory())
            s,p=self.session(root)
            stack.callback(s.close)
            first=s.execute('def run(robot, task):\n robot.move_base(.2,0,0,.2)\n raise ValueError("repair me")\n')
            self.assertEqual(first['execution']['status'],'policy_error')
            self.assertEqual(first['tools'][0]['method'],'move_base')
            self.assertEqual(first['tools'][0]['reason'],'duration_elapsed')
            x=s.observe()['pelvis_position'][0]
            elapsed=s.observe()['elapsed']
            self.assertGreater(x,.015)
            time.sleep(.12)
            second=s.execute('def run(robot, task):\n s=robot.observe()\n assert s["pelvis_position"][0] > .015\n robot.hold(.5)\n')
            self.assertEqual(second['execution']['status'],'completed')
            self.assertEqual(first['session_id'],second['session_id'])
            self.assertGreater(second['start_observation']['elapsed'],elapsed)
            self.assertIsNone(s.status()['terminal_reason'])
            self.assertEqual(len(s.rounds),2)
            self.assertTrue(all(c[0]=='g1-session-supervisor' for c in p.commands))

    def test_fall_while_editing_prevents_next_round(self):
        with ExitStack() as stack:
            root=stack.enter_context(tempfile.TemporaryDirectory())
            s,p=self.session(root)
            stack.callback(s.close)
            p.bad=True
            time.sleep(.1)
            self.assertEqual(s.status()['terminal_reason'],'unsafe_posture')
            with self.assertRaisesRegex(RuntimeError,'terminal'):
                s.execute('def run(robot, task): pass')

    def test_stale_worker_cannot_command_and_velocity_expires_to_idle(self):
        with ExitStack() as stack:
            root=stack.enter_context(tempfile.TemporaryDirectory())
            s,p=self.session(root)
            stack.callback(s.close)
            self.assertEqual(s.dispatch('move_base',[.2,0,0,.2],{},'old')['reason'],'stale_round')
            s.command_velocity(.2,0,0)
            time.sleep(.35)
            self.assertEqual(p.velocity,0.)
            self.assertIsNone(s.status()['terminal_reason'])

    def test_posture_hold_refreshes_during_editing_but_fault_stops_it(self):
        with ExitStack() as stack:
            root=stack.enter_context(tempfile.TemporaryDirectory())
            s,p=self.session(root)
            stack.callback(s.close)
            s.command_stationary(mode=4,height=.7)
            time.sleep(.35)
            count=sum(c[1]=='stationary' for c in p.commands)
            self.assertGreater(count,8)
            p.bad=True
            time.sleep(.1)
            self.assertEqual(s.status()['terminal_reason'],'unsafe_posture')
            self.assertEqual(p.commands[-1][1],'idle')

    def test_supervised_posture_can_be_used_again_after_long_caller_pause(self):
        with ExitStack() as stack:
            root=stack.enter_context(tempfile.TemporaryDirectory())
            s,p=self.session(root)
            stack.callback(s.close)
            time.sleep(.6)
            first=s.execute('def run(robot,task):\n assert robot.set_posture(.7)["status"] == "completed"\n')
            self.assertEqual(first['execution']['status'],'completed')
            time.sleep(.4)
            second=s.execute('def run(robot,task):\n assert robot.hold(.5)["status"] == "completed"\n')
            self.assertEqual(second['execution']['status'],'completed')
            self.assertGreater(second['start_observation']['elapsed'],first['end_observation']['elapsed'])

    def test_stationary_entry_requires_measured_settling_before_commands(self):
        with ExitStack() as stack:
            root=stack.enter_context(tempfile.TemporaryDirectory())
            s,p=self.session(root)
            stack.callback(s.close)
            result=s.tools.set_posture(.7)
            self.assertEqual(result['status'],'rejected')
            self.assertEqual(result['reason'],'base_not_settled')
            self.assertFalse(any(c[1]=='stationary' for c in p.commands))
            time.sleep(.6)
            self.assertGreaterEqual(s.observe()['settled_for'],.5)
            self.assertEqual(s.tools.set_posture(.7)['status'],'completed')

    def test_round_limit_is_total_and_cannot_be_reset_by_resubmission(self):
        with ExitStack() as stack:
            root=stack.enter_context(tempfile.TemporaryDirectory())
            s,p=self.session(root)
            stack.callback(s.close)
            s.max_rounds=2
            for _ in range(2): s.execute('def run(robot,task):\n robot.observe()\n')
            self.assertEqual(s.status()['terminal_reason'],'revision_budget')
            with self.assertRaisesRegex(RuntimeError,'terminal'):
                s.execute('def run(robot,task): pass')
