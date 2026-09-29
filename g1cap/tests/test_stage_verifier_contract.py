"""Shared motion guards remain authoritative over arbitrary program decisions.

These are synthetic sensor/worker tests, not new physics or generated verifiers.
"""
from copy import deepcopy
import unittest
from g1cap.arena_control import BoxControl
from g1cap.execution import execute_policy
from g1cap.sensor_turn import SensorTurn


class StageGuardTests(unittest.TestCase):
    def setUp(self):
        self.sample=dict(frame=dict(status='gyro_propagated_floor_frame',time_s=0.,segment=1,
            body_in_control_frame=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),up=[0,0,1],
            grasp=dict(status='available',time_s=0.,raised=True,ready=True,
                opposing_near_wrists=True,attitude_ok=True,gap_m=.1))
        self.action=[.2]*43+[0.,0.,0.,.75,0.,0.,0.]
        self.control=BoxControl(self.action,lambda obs:self.action,None,
            visual_grasp=lambda t:self.sample['grasp'],scene_hold=lambda phase,t,a:a,
            approach_feedback=lambda t:dict(status='available'),
            hand_clearance_feedback=lambda t:dict(status='available'),sensor_fault=lambda t:None,
            sensor_turn_factory=lambda t,a,yaw:SensorTurn(t,yaw,lambda now:deepcopy(self.sample)))
        self.control.update({'time':0.})

    def test_always_pass_program_cannot_turn_with_empty_hands(self):
        self.sample['grasp'].update(raised=False,ready=False,opposing_near_wrists=False)
        replies=[]
        def dispatch(method,args,kwargs,round_id):
            self.assertEqual(method,'turn_with_box')
            reply=self.control.start(method,kwargs,{'time':0.});replies.append(reply)
            return reply
        source="""def verify_stage(stage, observations):
    return {'status':'pass'}
def run(robot, task):
    if verify_stage('pickup', [])['status']=='pass':
        robot.turn_with_box(yaw_rad=.5)
"""
        result=execute_policy(source,{},'unconditional-pass',dispatch,
            tools={'turn_with_box'},wall_timeout=5.)
        self.assertEqual(result['status'],'completed',result)
        self.assertEqual(replies[0]['status'],'rejected')
        self.assertEqual(replies[0]['reason'],'settled_bilateral_grasp_required')
        self.assertEqual(self.control.last_action[43:46],[0.,0.,0.])
        self.assertIsNone(self.control.method)

    def test_grasp_loss_during_accepted_turn_stops_before_next_navigation(self):
        c=self.control
        self.assertEqual(c.start('turn_with_box',{'yaw_rad':.5},{'time':0.})['status'],'running')
        self.assertGreater(c.command({'time':0.})[45],0.)
        self.sample['frame']['time_s']=.02
        self.sample['grasp'].update(time_s=.02,opposing_near_wrists=False)
        self.assertEqual(c.command({'time':.02})[43:46],[0.,0.,0.])
        self.assertEqual(c.result['status'],'failed')
        self.assertEqual(c.result['reason'],'hold_visual_retention_lost')
        # A verifier cannot erase this failed operation. Even a contradictory
        # ready flag cannot bypass the controller's independent opposed check.
        self.assertEqual(c.start('turn_with_box',{'yaw_rad':.5},{'time':.02})['status'],'rejected')
        self.assertEqual(c.result['status'],'failed')

    def test_stale_or_future_grasp_cannot_authorize_navigation(self):
        for observed_at in (-1.,1.):
            with self.subTest(observed_at=observed_at):
                self.setUp();self.sample['grasp']['time_s']=observed_at
                reply=self.control.start('turn_with_box',{'yaw_rad':.5},{'time':0.})
                self.assertEqual(reply['status'],'rejected')
                self.assertEqual(reply['reason'],'turn_ready_camera_time_invalid')
                self.assertEqual(self.control.last_action[43:46],[0.,0.,0.])
