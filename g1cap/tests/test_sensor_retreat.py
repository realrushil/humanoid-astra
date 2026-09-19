from copy import deepcopy
import unittest

from g1cap.arena_control import BoxControl


class SensorRetreatTests(unittest.TestCase):
    def setUp(self):
        self.sample=dict(grasp=dict(status='available',raised=True,ready=True),
            frame=dict(status='observed_floor_heading_frame',time_s=0.,segment=1,
                body_in_control_frame=[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]]),
            front=dict(status='available',time_s=0.,offset_body_m=-.4,
                normal_segment=[1.,0.,0.],normal_navigation_xy=[1.,0.]))
        self.wrist_ready=True
        self.action=[.2]*43+[0.,0.,0.,.75,0.,0.,0.]

    def controller(self):
        from g1cap.sensor_retreat import SensorRetreat
        return SensorRetreat(0.,.2,lambda t:deepcopy(self.sample),lambda:self.wrist_ready)

    def measure(self,t,progress):
        self.sample['front'].update(time_s=t,offset_body_m=-.4-progress)
        self.sample['frame']['time_s']=t

    def test_only_navigation_changes_and_waits_for_synchronized_wrist_handoff(self):
        c=self.controller();self.wrist_ready=False
        self.assertEqual(c.command(0.,self.action),self.action)
        self.measure(.02,0);self.wrist_ready=True
        command=c.command(.02,self.action)
        self.assertLess(command[43],0.)
        self.assertEqual(command[:43],self.action[:43])
        self.assertEqual(command[46:],self.action[46:])

    def test_measured_distance_and_stability_complete_without_truth_state(self):
        c=self.controller();c.command(0.,self.action)
        for i in range(1,81):
            t=i*.02;self.measure(t,min(.19,t))
            command=c.command(t,self.action)
            result=c.update(t)
            if result:break
        self.assertEqual(result,('completed','sensor_retreat_and_hold_completed'))
        self.assertEqual(command[43:46],[0.,0.,0.])
        self.assertAlmostEqual(c.measurements(t)['retreat_m'],.19)

    def test_grasp_loss_front_loss_and_control_frame_change_reject(self):
        for field in ('grasp','front','frame'):
            self.setUp();c=self.controller()
            self.sample[field]['status']='unavailable'
            with self.assertRaises(ValueError):c.command(.02,self.action)
        self.setUp();c=self.controller();self.sample['frame']['segment']=2
        with self.assertRaises(ValueError):c.command(.02,self.action)

    def test_box_control_carry_then_hold_retains_references_and_cancels_on_loss(self):
        from g1cap.sensor_retreat import SensorRetreat
        c=BoxControl(self.action,lambda obs:self.action,None,
            visual_grasp=lambda t:self.sample['grasp'],scene_hold=lambda phase,t,a:a,
            approach_feedback=lambda t:self.sample['front'],
            hand_clearance_feedback=lambda t:dict(status='available'),sensor_fault=lambda t:None,
            sensor_retreat_factory=lambda t,a,d:SensorRetreat(t,d,lambda now:deepcopy(self.sample),lambda:True))
        c.update({'time':0.})
        self.assertEqual(c.start('retreat_with_box',{'distance_m':.2},{'time':0.})['status'],'running')
        for i in range(81):
            t=i*.02;self.measure(t,min(.19,t))
            command=c.command({'time':t});result=c.update({'time':t})
            if result:break
        self.assertEqual(result['status'],'completed')
        self.assertEqual(command[:43],c.last_action[:43])
        self.assertEqual(c.start('hold_box',{'duration':1.},{'time':t})['status'],'running')
        c.cancel('test_cancel',{'time':t})
        t+=.02;self.measure(t,.19);c.update({'time':t})
        self.assertEqual(c.start('retreat_with_box',{'distance_m':.2},{'time':t})['status'],'running')
        self.sample['front']['status']='unavailable'
        self.assertEqual(c.command({'time':t+.02})[43:46],[0.,0.,0.])
        self.assertEqual(c.result['reason'],'sensor_approach_unavailable')
        self.assertIsNone(c.method)


if __name__=='__main__':unittest.main()
