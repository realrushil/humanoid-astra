"""Control-contract tests; synthetic state is not physical qualification."""
import copy
import importlib
import math
import unittest
from types import SimpleNamespace
from g1cap.toolkit.arm_planner import RIGHT_ARM,UPPER_BODY
from g1cap.toolkit.sonic_backend import LeasedPlanner


def initial():
    return dict(pelvis_yaw=.2,body_joint_names=list(UPPER_BODY),
                body_joint_positions=[0.]*17)


class MotionSequenceTests(unittest.TestCase):
    def prepare(self,segments):
        module=importlib.import_module('g1cap.toolkit.motion_sequence')
        return module.prepare_motion(segments,initial(),SimpleNamespace(lower=[-1.]*7,upper=[1.]*7))

    def test_world_facing_and_arm_interpolate_and_persist(self):
        module=importlib.import_module('g1cap.toolkit.motion_sequence')
        segments=[dict(mode='walk',duration=1.,velocity_world=[.1,0],facing_world=.4,
                       right_arm={'right_elbow_joint':.15}),dict(mode='idle',duration=1.)]
        original=copy.deepcopy(segments)
        prepared=self.prepare(segments)
        fields=module.motion_fields(prepared[0],.5)
        self.assertAlmostEqual(fields['facing_world'],.3)
        self.assertAlmostEqual(fields['positions'][UPPER_BODY.index('right_elbow_joint')],.075)
        self.assertEqual(fields['velocity_world'],[.1,0])
        later=module.motion_fields(prepared[1],.5)
        self.assertAlmostEqual(later['positions'][UPPER_BODY.index('right_elbow_joint')],.15)
        self.assertEqual(segments,original)

    def test_reject_invalid_later_segment_and_unsupported_fields(self):
        valid=dict(mode='walk',duration=1.,velocity_world=[.1,0])
        for bad in [dict(mode='run',duration=1.),dict(mode='idle',duration=float('nan')),
                    dict(mode='walk',duration=1.,velocity_world=[.3,0]),
                    dict(mode='idle',duration=1.,right_arm={'left_elbow_joint':.1}),
                    dict(mode='idle',duration=1.,right_arm={'right_elbow_joint':.21}),
                    dict(mode='idle',duration=.2,right_arm={'right_elbow_joint':.15}),
                    dict(mode='idle',duration=1.,height=.7),
                    dict(mode='idle',duration=1.,facing_world=1.),
                    dict(mode='idle',duration=1.,teleport=True)]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):self.prepare([valid,bad])
        with self.assertRaises(ValueError):self.prepare([dict(mode='idle',duration=2.)]*5)

    def test_joint_limits_checked_separately_from_relative_bound(self):
        module=importlib.import_module('g1cap.toolkit.motion_sequence')
        with self.assertRaises(ValueError):
            module.prepare_motion([dict(mode='idle',duration=1.,right_arm={'right_elbow_joint':.15})],
                                  initial(),SimpleNamespace(lower=[-.1]*7,upper=[.1]*7))

    def test_motion_transport_uses_world_velocity_and_expires_all_references(self):
        p=LeasedPlanner(0.,0.)
        p.motion(mode=1,velocity_world=[0.,.1],facing_world=.3,height=-1.,positions=[0.]*17,now=0.)
        fields=p.fields(measured_yaw=1.2,now=.1)
        self.assertEqual(fields['movement'],[0.,1.,0.])
        self.assertAlmostEqual(fields['facing'][0],math.cos(.3))
        self.assertEqual(fields['speed'],.1)
        expired=p.fields(measured_yaw=1.2,now=.3)
        self.assertEqual(expired['mode'],0)
        self.assertNotIn('upper_body_position',expired)

class SessionMotionTests(unittest.TestCase):
    def test_worker_routes_motion_and_invalid_sequence_preserves_hold(self):
        import tempfile,time,threading
        from test_session import LiveSessionTests
        with tempfile.TemporaryDirectory() as root:
            session,publisher=LiveSessionTests().session(root)
            def publish(**fields):
                publisher.velocity=fields['velocity_world'][0]
                publisher.commands.append((threading.current_thread().name,'motion',fields))
            publisher.set_motion=publish
            try:
                time.sleep(.6)
                result=session.execute("def run(robot,task):\n r=robot.sonic_motion([{'mode':'walk','duration':.3,'velocity_world':[.1,0]}])\n assert r['status']=='completed', r\n")
                self.assertEqual(result['execution']['status'],'completed')
                self.assertEqual(result['tools'][0]['reason'],'duration_elapsed')
                self.assertTrue(any(c[1]=='motion' for c in publisher.commands))
                self.assertEqual(session.desired[0],'idle')
                session.command_stationary(mode=4,height=.7)
                before=copy.deepcopy(session.desired)
                with self.assertRaises(ValueError):
                    session.tools.sonic_motion([dict(mode='idle',duration=1.),dict(mode='run',duration=1.)])
                self.assertEqual(session.desired,before)
                session.command_motion(mode=1,velocity_world=[.1,0],facing_world=0.,height=-1.,positions=None)
                time.sleep(.3)
                self.assertEqual(session.desired[0],'idle')
                self.assertEqual(publisher.velocity,0.)
                self.assertTrue(all(c[0]=='g1-session-supervisor' for c in publisher.commands))
            finally:session.close()

    def test_terminal_fault_interrupts_timed_motion(self):
        import tempfile,time,threading
        from test_session import LiveSessionTests
        with tempfile.TemporaryDirectory() as root:
            session,publisher=LiveSessionTests().session(root)
            publisher.set_motion=lambda **fields: None
            try:
                time.sleep(.6)
                timer=threading.Timer(.12,lambda:setattr(publisher,'bad',True));timer.start()
                result=session.tools.sonic_motion([dict(mode='idle',duration=1.)])
                timer.join()
                self.assertEqual(result['status'],'cancelled')
                self.assertEqual(result['reason'],'unsafe_posture')
                self.assertLess(result['elapsed'],.5)
                self.assertEqual(session.desired[0],'idle')
            finally:session.close()

    def test_expired_reference_cannot_be_reported_complete_after_executor_stall(self):
        import tempfile,time
        from unittest.mock import patch
        from test_session import LiveSessionTests
        real_sleep=time.sleep
        with tempfile.TemporaryDirectory() as root:
            session,publisher=LiveSessionTests().session(root)
            publisher.set_motion=lambda **fields: None
            try:
                real_sleep(.6)
                with patch('g1cap.session_tools.time.sleep',side_effect=lambda _:real_sleep(.35)):
                    result=session.tools.sonic_motion([dict(mode='idle',duration=.2)])
                self.assertEqual(result['status'],'failed')
                self.assertEqual(result['reason'],'motion_reference_lost')
            finally:session.close()
