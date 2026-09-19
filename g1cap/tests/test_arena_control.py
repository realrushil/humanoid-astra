import copy
import unittest

from g1cap.arena_control import BoxControl


def observation(time=0., clearance=.08):
    return dict(time=time, state_age_s=0., box_pos=[.7,.18,.18],
                box_quat=[1.,0.,0.,0.], root_pos=[.1,.18,0.], root_quat=[1.,0.,0.,0.], joint_pos=[0.]*43,
                wrists_world={side+'_wrist_yaw_link':{'pos':[0.,0.,0.],'xyzw':[0.,0.,0.,1.]} for side in ['left','right']},
                bilateral=True, clearance=clearance, tilt=0., stance_clear=True,
                supported=False, hand_forces_N={'left':10.,'right':10.},
                robot_source_peak_N=0.)


class BoxControlTests(unittest.TestCase):
    def test_missing_hand_measurement_stops_before_policy_and_retains_joint_references(self):
        calls=[];available=[False]
        def feedback(t):return {'status':'available' if available[0] else 'unavailable'}
        control=BoxControl([.2]*43+[0.]*7,lambda obs:calls.append(obs) or [0.]*50,None,
                           hand_clearance_feedback=feedback)
        self.assertEqual(control.start('pickup_box',{'object_id':'brown_box'},observation())['reason'],
                         'sensor_hand_clearance_unavailable')
        available[0]=True
        control.start('pickup_box',{'object_id':'brown_box'},observation())
        before=list(control.last_action)
        control.last_action[43:46]=[.1,.2,.3]
        available[0]=False
        action=control.command(observation(.02))
        self.assertEqual(action[:43],before[:43])
        self.assertEqual(action[43:46],[0.,0.,0.])
        self.assertEqual(calls,[])
        self.assertEqual(control.result['reason'],'sensor_hand_clearance_unavailable')

    def test_missing_approach_measurement_rejects_or_stops_without_native_action(self):
        calls=[];available=[False]
        def front(t):return {'status':'available' if available[0] else 'unavailable'}
        control=BoxControl([0.]*50,lambda obs:calls.append(obs) or [0.]*50,None,approach_feedback=front)
        result=control.start('pickup_box',{'object_id':'brown_box'},observation())
        self.assertEqual(result['reason'],'sensor_approach_unavailable')
        available[0]=True
        control.start('pickup_box',{'object_id':'brown_box'},observation())
        control.last_action[43:46]=[.1,.2,.3]
        available[0]=False
        action=control.command(observation(.02))
        self.assertEqual(action[43:46],[0.,0.,0.])
        self.assertEqual(calls,[])
        self.assertEqual(control.result['status'],'failed')

    def test_scene_hold_callback_receives_only_time_phase_and_own_reference(self):
        calls=[]
        def hold(phase,time_s,previous):
            calls.append((phase,time_s,list(previous)))
            result=list(previous)
            if phase=='verify_pickup':result[11]+=.001
            return result
        visual=lambda t:dict(status='available',raised=True,ready=False,attitude_ok=True)
        control=BoxControl([0.]*50,lambda obs:[0.]*50,None,visual_grasp=visual,scene_hold=hold)
        control.start('pickup_box',{'object_id':'brown_box'},observation())
        for i in range(1,16):control.update(observation(i*.02))
        action=control.command(observation(.3))
        self.assertAlmostEqual(action[11],.001)
        self.assertEqual(calls[-1][:2],('verify_pickup',.3))
        self.assertEqual(len(calls[-1]),3)

    def test_scene_hold_failure_cancels_without_returning_a_cached_navigation_command(self):
        def hold(phase,time_s,previous):raise ValueError('scene_wrist_hold_invalidated')
        visual=lambda t:dict(status='available',raised=True,ready=False,attitude_ok=True)
        action=[0.]*50;action[43]=.1;action[11]=.2
        control=BoxControl(action,lambda obs:action,None,visual_grasp=visual,scene_hold=hold)
        control.start('pickup_box',{'object_id':'brown_box'},observation())
        for i in range(1,16):control.update(observation(i*.02))
        actual=control.command(observation(.3))
        self.assertEqual(actual[43:46],[0.,0.,0.])
        self.assertAlmostEqual(actual[11],.2)
        self.assertEqual(control.result['status'],'failed')
        self.assertEqual(control.result['reason'],'scene_wrist_hold_invalidated')

    def setUp(self):
        self.calls=[]
        self.action=[0.]*50
        self.action[46]=.75
        self.control=BoxControl(self.action, lambda obs:self.action,
                                lambda obs,shift:self.calls.append(shift))

    def settled(self, clearance=.08):
        for step in range(51):
            self.control.update(observation(step*.02,clearance))
        return observation(1.,clearance)

    def test_invalid_height_issues_no_motion(self):
        state=self.settled()
        before=copy.deepcopy(self.control.last_action)
        result=self.control.start('raise_held_box',{'clearance_m':float('nan')},state)
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(self.control.last_action,before)
        self.assertEqual(self.calls,[])

    def test_unknown_object_and_stale_state_rejected(self):
        self.assertEqual(self.control.start('pickup_box',{'object_id':'apple'},observation())['status'],'rejected')
        state=observation();state['state_age_s']=1.
        self.assertEqual(self.control.start('pickup_box',{'object_id':'brown_box'},state)['reason'],'stale_observation')

    def test_concurrent_request_does_not_replace_operation(self):
        state=self.settled()
        self.assertEqual(self.control.start('hold_box',{'duration':1.},state)['status'],'running')
        self.assertEqual(self.control.start('pickup_box',{'object_id':'brown_box'},state)['reason'],'operation_active')
        self.assertEqual(self.control.method,'hold_box')

    def test_low_grasp_failure_retains_state_and_references(self):
        self.control.start('pickup_box',{'object_id':'brown_box'},observation())
        for step in range(1,16):
            state=observation(step*.02)
            self.control.command(observation((step-1)*.02))
            self.control.update(state)
        for step in range(16,116):
            state=observation(step*.02,.0437)
            self.control.command(observation((step-1)*.02,.0437))
            self.control.update(state)
        self.assertEqual(self.control.result['reason'],'clearance_below_goal')
        self.assertTrue(self.control.result['observation']['bilateral'])
        self.assertEqual(self.control.last_action,self.action)
        self.assertEqual(self.calls,[])

    def test_contact_failure_persists_after_new_request(self):
        state=self.settled()
        self.control.start('hold_box',{'duration':1.},state)
        state=observation(1.02);state['robot_source_peak_N']=12.
        self.control.update(state)
        self.assertEqual(self.control.result['reason'],'forbidden_source_contact')
        self.assertEqual(self.control.start('hold_box',{},state)['reason'],'episode_failed')

    def test_destination_contact_is_a_latched_episode_fault(self):
        state=self.settled()
        self.control.start('wait',{'duration':1.},state)
        state=observation(1.02)
        state['surfaces']={'destination':{'robot_contact_peak_N':12.}}
        self.control.update(state)
        self.assertEqual(self.control.result['reason'],'forbidden_destination_contact')
        state=observation(1.04)
        self.assertEqual(self.control.start('wait',{},state)['reason'],'episode_failed')

    def test_measured_floor_contact_stops_the_episode(self):
        state=self.settled();self.control.start('wait',{'duration':1.},state)
        state=observation(1.02);state['box_floor_contact_peak_N']=1.
        self.assertEqual(self.control.update(state)['reason'],'box_floor_contact')
        self.assertEqual(self.control.command(state)[43:46],[0.,0.,0.])

    def test_finite_but_missing_observation_is_rejected(self):
        state=observation();del state['hand_forces_N']
        self.assertEqual(self.control.start('hold_box',{},state)['status'],'rejected')

    def test_hold_requires_continuous_good_measurements(self):
        state=self.settled()
        self.control.start('hold_box',{'duration':1.},state)
        for step in range(51,102):self.control.update(observation(step*.02))
        self.assertEqual(self.control.result['status'],'completed')


if __name__=='__main__':unittest.main()

class VisualCompletionTests(unittest.TestCase):
    def test_visual_feedback_controls_pickup_completion_not_contact_flags(self):
        feedback=dict(status='available',raised=False,ready=False,attitude_ok=True)
        action=[0.]*50;action[46]=.75
        c=BoxControl(action,lambda obs:action,lambda *args:None,visual_grasp=lambda now:feedback)
        c.start('pickup_box',{'object_id':'brown_box'},observation())
        for i in range(1,21):c.update(observation(i*.02))
        self.assertEqual(c.phase,'acquire')
        feedback['raised']=True
        for i in range(21,36):
            s=observation(i*.02,0);s['bilateral']=False;s['stance_clear']=False;c.update(s)
        self.assertEqual(c.phase,'verify_pickup')
        feedback['ready']=True
        for i in range(36,87):
            s=observation(i*.02,0);s['bilateral']=False;s['stance_clear']=False;c.update(s)
        self.assertEqual(c.result['reason'],'visually_raised_and_stable')
        self.assertEqual(c.start('hold_box',{'duration':1},s)['status'],'running')

    def test_unavailable_visual_feedback_retains_arm_targets_and_cancels_motion(self):
        feedback=dict(status='available',raised=False,ready=False,attitude_ok=True)
        action=[.2]*50;action[43:46]=[0.,0.,0.]
        c=BoxControl(action,lambda obs:action,lambda *args:None,visual_grasp=lambda now:feedback)
        c.start('pickup_box',{'object_id':'brown_box'},observation())
        feedback['status']='unavailable'
        result=c.update(observation(.02))
        self.assertEqual(result['reason'],'visual_state_unavailable')
        self.assertEqual(c.last_action[:43],c._action(action)[:43])
        self.assertEqual(c.last_action[43:46],[0.,0.,0.])

    def test_command_cancels_previous_navigation_on_visual_loss(self):
        feedback=dict(status='available',raised=False,ready=False,attitude_ok=True)
        action=[.2]*50;action[43:46]=[.1,-.1,.15]
        c=BoxControl(action,lambda obs:action,lambda *args:None,visual_grasp=lambda now:feedback)
        c.start('pickup_box',{'object_id':'brown_box'},observation())
        self.assertEqual(c.command(observation())[43:46],c._action(action)[43:46])
        feedback['status']='unavailable'
        stopped=c.command(observation(.02))
        self.assertEqual(stopped[43:46],[0.,0.,0.])
        self.assertEqual(stopped[:43],c._action(action)[:43])
        self.assertEqual(c.result['reason'],'visual_state_unavailable')
