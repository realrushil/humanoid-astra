from copy import deepcopy
import math
import unittest
try:
    import numpy as np
except ImportError:
    np=None
from g1cap.arena_control import BoxControl


@unittest.skipIf(np is None,'numpy runtime required')
class SensorTurnTests(unittest.TestCase):
    def setUp(self):
        from g1cap.sensor_turn import SensorTurn
        self.Controller=SensorTurn
        self.sample=dict(frame=dict(status='gyro_propagated_floor_frame',time_s=0.,segment=1,
            body_in_control_frame=np.eye(4).tolist()),up=[0.,0.,1.],
            grasp=dict(status='available',time_s=0.,raised=True,ready=True,
                opposing_near_wrists=True,attitude_ok=True,gap_m=.1))
        self.front=dict(status='available')
        self.action=[.2]*43+[0.,0.,0.,.75,0.,0.,0.]

    def measure(self,t,angle):
        c,s=math.cos(angle),math.sin(angle)
        self.sample['frame'].update(time_s=t,
            body_in_control_frame=[[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]])
        self.sample['grasp']['time_s']=round(math.floor((t+1e-8)/.1)*.1,8)

    def factory(self,t,a,angle):
        return self.Controller(t,angle,lambda now:deepcopy(self.sample))

    def control(self):
        return BoxControl(self.action,lambda obs:self.action,None,
            visual_grasp=lambda t:self.sample['grasp'],scene_hold=lambda phase,t,a:a,
            approach_feedback=lambda t:self.front,hand_clearance_feedback=lambda t:dict(status='available'),
            sensor_fault=lambda t:None,sensor_turn_factory=self.factory)

    def test_signed_turn_changes_only_navigation_and_needs_fresh_ready_window(self):
        for sign in (-1,1):
            self.setUp();c=self.factory(0,self.action,sign*.5)
            a=c.command(0,self.action)
            self.assertAlmostEqual(a[45],sign*.18)
            self.assertEqual(a[:43],self.action[:43]);self.assertEqual(a[46:],self.action[46:])
            for i in range(1,56):
                t=i*.02;self.measure(t,sign*.49);a=c.command(t,a)
                result=c.update(t)
                if i<55:self.assertIsNone(result)
            self.assertEqual(result,('completed','measured_heading_and_hold'))
            self.assertEqual(a[43:46],[0,0,0])
            self.assertEqual(c.measurements(t)['ready_window']['fresh_samples'],11)

    def test_unavailable_or_old_frame_and_retention_loss_stop(self):
        for fault in ('stale','segment','grasp','opposed','gap'):
            self.setUp();c=self.factory(0,self.action,.5);c.command(0,self.action)
            self.measure(.02,0)
            if fault=='stale':self.sample['frame']['time_s']=0.
            if fault=='segment':self.sample['frame']['segment']=2
            if fault=='grasp':self.sample['grasp']['status']='unavailable'
            if fault=='opposed':self.sample['grasp']['opposing_near_wrists']=False
            if fault=='gap':self.sample['grasp']['gap_m']=.01
            with self.subTest(fault=fault),self.assertRaises(ValueError):c.command(.02,self.action)

    def test_public_control_turn_then_hold_uses_time_only_without_privileged_space_guard(self):
        c=self.control();c.update({'time':0.})
        self.assertEqual(c.start('turn_with_box',{'yaw_rad':.5},{'time':0.})['status'],'running')
        for i in range(61):
            t=i*.02;self.measure(t,.49 if i else 0.)
            c.command({'time':t});result=c.update({'time':t})
            if result:break
        self.assertEqual(result['status'],'completed')
        self.assertEqual(c.last_action[43:46],[0.,0.,0.])
        self.assertEqual(c.start('hold_box',{'duration':1.},{'time':t})['status'],'running')
        c.cancel('worker_request_cancelled',{'time':t})
        t+=.02;self.measure(t,.49);c.update({'time':t})
        self.assertEqual(c.start('turn_with_box',{'yaw_rad':-.5},{'time':t})['status'],'running')
        c.cancel('worker_request_cancelled',{'time':t})
        self.assertEqual(c.last_action[43:46],[0.,0.,0.])
        np.testing.assert_allclose(c.last_action[:43],self.action[:43],atol=1e-7)

    def test_geometry_loss_fails_turn_at_both_command_and_update(self):
        for boundary in ('command','update'):
            self.setUp();c=self.control();c.update({'time':0.})
            self.assertEqual(c.start('turn_with_box',{'yaw_rad':.5},{'time':0.})['status'],'running')
            c.command({'time':0.});self.front['status']='unavailable'
            getattr(c,boundary)({'time':.02})
            self.assertEqual(c.result['reason'],'sensor_approach_unavailable')
            self.assertEqual(c.last_action[43:46],[0.,0.,0.])

    def test_publication_allows_turn_and_sensor_reasons(self):
        from g1cap.arena_public import METHODS,public_reason
        self.assertIn('turn_with_box',METHODS)
        for reason in ('measured_heading_and_hold','measured_turn_timeout','turn_ready_camera_time_invalid'):
            self.assertEqual(public_reason(reason),reason)

    def test_post_physics_completion_requires_current_heading_and_readiness(self):
        for change in ('heading','ready'):
            self.setUp();c=self.factory(0,self.action,.5);c.command(0,self.action)
            for i in range(1,56):
                self.measure(i*.02,.49);c.command(i*.02,self.action)
            self.assertEqual(c.latest['outcome'],'completed')
            self.measure(1.12,.44 if change=='heading' else .49)
            if change=='ready':self.sample['grasp']['ready']=False
            self.assertEqual(c.update(1.12),('failed','turn_completion_not_retained'))
