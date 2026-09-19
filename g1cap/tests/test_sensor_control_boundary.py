from copy import deepcopy
import tempfile
import threading
import unittest

from g1cap.arena_control import BoxControl
from g1cap.arena_session import ArenaSession


class SensorControlBoundaryTests(unittest.TestCase):
    def test_native_hidden_state_terminations_are_removed_and_unknown_terms_reject(self):
        from types import SimpleNamespace
        from g1cap.arena_scene import sensor_terminations
        timeout=object()
        cfg=SimpleNamespace(time_out=timeout,success=object(),object_dropped=object())
        sensor_terminations(cfg)
        self.assertIs(cfg.time_out,timeout)
        self.assertIsNone(cfg.success)
        self.assertIsNone(cfg.object_dropped)
        cfg.secret_contact=object()
        with self.assertRaises(ValueError):sensor_terminations(cfg)

    def control(self,fault=lambda t:None):
        self.calls=[]
        def acquire(obs):
            self.assertEqual(set(obs),{'time'})
            self.calls.append(obs)
            return [.2]*43+[.1,0.,0.,.75,0.,0.,0.]
        return BoxControl([.2]*43+[0.]*7,acquire,None,
            visual_grasp=lambda t:dict(status='available',time_s=t,raised=True,ready=True,attitude_ok=True),
            scene_hold=lambda phase,t,action:action,
            approach_feedback=lambda t:{'status':'available'},
            hand_clearance_feedback=lambda t:{'status':'available'},sensor_fault=fault)

    def test_time_only_pickup_hold_and_strict_rejection_of_truth_packet(self):
        c=self.control();c.update({'time':0.})
        self.assertEqual(c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})['status'],'running')
        c.command({'time':0.})
        for i in range(1,66):c.update({'time':i*.02})
        self.assertEqual(c.result['status'],'completed')
        self.assertEqual(set(c.result['observation']),{'time'})
        self.assertEqual(c.start('hold_box',{'duration':1.},{'time':1.3})['status'],'running')
        for i in range(66,116):c.update({'time':i*.02})
        self.assertEqual(c.result['reason'],'hold_verified')
        c=self.control()
        self.assertEqual(c.start('wait',{},dict(time=0.,box_mass_kg=.1))['reason'],'invalid_sensor_control_packet')

    def test_fault_stops_before_policy_and_keeps_joint_references(self):
        state=[None];c=self.control(lambda t:state[0])
        c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})
        c.last_action[43:46]=[.1,.2,.3];before=c.last_action[:43]
        state[0]='sensor_effort_saturation'
        self.assertEqual(c.command({'time':.02}),before+[0.,0.,0.]+c.last_action[46:])
        self.assertEqual(self.calls,[])
        self.assertEqual(c.terminal_reason,state[0])

    def test_malformed_time_stops_without_raising_or_reusing_navigation(self):
        for obs in ({},{'time':float('nan')},{'time':True}):
            c=self.control();c.last_action[43:46]=[.1,.2,.3]
            self.assertEqual(c.command(obs)[43:46],[0.,0.,0.])
            self.assertEqual(c.terminal_reason,'invalid_sensor_control_packet')

    def test_geometry_loss_prevents_visual_completion_and_hold_admission(self):
        c=self.control()
        visual=c.visual_grasp
        c.visual_grasp=lambda t:dict(visual(t),ready=False)
        c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})
        for i in range(1,65):c.update({'time':i*.02})
        c.visual_grasp=visual
        c.hand_clearance_feedback=lambda t:{'status':'unavailable'}
        self.assertEqual(c.update({'time':1.3})['reason'],'sensor_hand_clearance_unavailable')
        self.assertEqual(c.start('hold_box',{'duration':1.},{'time':1.3})['reason'],
                         'sensor_hand_clearance_unavailable')

    def test_lost_geometry_does_not_run_background_hold_after_cancellation(self):
        c=self.control();calls=[]
        c.scene_hold=lambda phase,t,action:calls.append(phase) or [0.]*50
        c.hand_clearance_feedback=lambda t:{'status':'unavailable'}
        before=list(c.last_action)
        self.assertEqual(c.command({'time':0.}),before)
        self.assertEqual(calls,[])

    def test_separate_loaded_stop_keeps_stabilization_after_geometry_fault(self):
        c=self.control();calls=[]
        def stop(now,previous):
            calls.append((now,list(previous)))
            action=list(previous);action[0]+=.01;return action
        c.loaded_stop=stop;c.update({'time':0.})
        self.assertEqual(c.start('hold_box',{'duration':1.},{'time':0.})['status'],'running')
        c.hand_clearance_feedback=lambda t:{'status':'unavailable'}
        c.last_action[43:46]=[.1,.2,.3]
        before=list(c.last_action)
        action=c.command({'time':.02})
        self.assertEqual(c.result['status'],'failed')
        self.assertEqual(c.result['reason'],'sensor_hand_clearance_unavailable')
        self.assertEqual(calls[0][1][43:46],[0.,0.,0.])
        self.assertAlmostEqual(action[0],before[0]+.01,places=6)
        self.assertEqual(action[43:46],[0.,0.,0.])
        c.command({'time':.04})
        self.assertEqual(len(calls),2)

    def test_loaded_stop_rejection_retains_references_and_records_failure(self):
        c=self.control()
        def reject(now,previous):raise ValueError('stop_top_unavailable')
        c.loaded_stop=reject;c.hand_clearance_feedback=lambda t:{'status':'unavailable'}
        before=list(c.last_action);c.last_action[43:46]=[.1,.2,.3]
        self.assertEqual(c.command({'time':0.}),before)
        self.assertEqual(c.stopping_failure,'stop_top_unavailable')

    def test_hard_sensor_fault_never_calls_loaded_stop(self):
        c=self.control(lambda t:'sensor_effort_saturation');calls=[]
        c.loaded_stop=lambda t,a:calls.append(t) or a
        c.visual_grasp=lambda t:dict(status='unavailable',reason='source_height_precision_insufficient')
        before=list(c.last_action);c.last_action[43:46]=[.1,.2,.3]
        self.assertEqual(c.command({'time':0.}),before)
        self.assertEqual(calls,[])
        self.assertEqual(c.result['reason'],'sensor_effort_saturation')

    def test_height_loss_stop_rejection_or_missing_callback_keeps_references(self):
        for callback_present in (False,True):
            with self.subTest(callback_present=callback_present):
                c=self.control();c.update({'time':0.})
                c.start('hold_box',{'duration':1.},{'time':0.})
                def reject(now,previous):raise ValueError('stop_top_unavailable')
                if callback_present:c.loaded_stop=reject
                c.visual_grasp=lambda t:dict(status='unavailable',reason='source_height_precision_insufficient')
                ordinary_calls=[]
                c.scene_hold=lambda phase,t,a:ordinary_calls.append(phase) or [0.]*50
                before=list(c.last_action);c.last_action[43:46]=[.1,.2,.3]
                self.assertEqual(c.command({'time':.02}),before)
                self.assertEqual(c.result['reason'],'visual_state_unavailable')
                self.assertEqual(c.result['status'],'failed')
                self.assertEqual(ordinary_calls,[])
                if callback_present:self.assertEqual(c.stopping_failure,'stop_top_unavailable')

    def test_two_distinct_fresh_images_admit_stabilization_not_completion(self):
        c=self.control();camera={'time_s':0.,'raised':False}
        c.visual_grasp=lambda now:dict(status='available',ready=False,attitude_ok=True,
            scene_status='unavailable',**camera)
        c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})
        for step in range(1,10):
            t=step*.02
            if step==5:camera.update(time_s=t,raised=True)
            c.update({'time':t})
            self.assertEqual(c.phase,'acquire')
        camera.update(time_s=.2,raised=True);c.update({'time':.2})
        self.assertEqual(c.phase,'verify_pickup');self.assertIsNone(c.result)
        self.assertEqual(c.phase_started,.2);self.assertEqual(c.last_action[43:46],[0.,0.,0.])

    def test_nonraised_or_discontinuous_fresh_image_restarts_admission_count(self):
        for middle in ({'time_s':.2,'raised':False},{'time_s':.1,'raised':True}):
            c=self.control();camera={'time_s':0.,'raised':False}
            c.visual_grasp=lambda now:dict(status='available',ready=False,attitude_ok=True,**camera)
            c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})
            for step in range(1,20):
                t=step*.02
                if step==5:camera.update(time_s=t,raised=True)
                if step==10:camera.update(middle)
                if step==15:camera.update(time_s=t,raised=True)
                c.update({'time':t})
                self.assertEqual(c.phase,'acquire')
            camera.update(time_s=.4,raised=True);c.update({'time':.4})
            self.assertEqual(c.phase,'verify_pickup')

    def test_missing_future_or_nonfinite_camera_time_cannot_admit_hold(self):
        for observed_at in (None,float('nan'),10.,True):
            c=self.control()
            c.visual_grasp=lambda t:dict(status='available',raised=True,ready=False,
                attitude_ok=True,scene_status='available',time_s=observed_at)
            c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})
            for step in range(1,31):c.update({'time':step*.02})
            self.assertEqual(c.phase,'acquire')

    def run_session(self,secret):
        c=self.control();now=[0.];seen=[]
        for name in ('start','update','cancel'):
            original=getattr(c,name)
            def record(*args,_name=name,_original=original):
                seen.append((_name,deepcopy(args[-1])))
                return _original(*args)
            setattr(c,name,record)
        with tempfile.TemporaryDirectory() as folder:
            s=ArenaSession(c,{},folder,sensor_observation=lambda t:{'time':now[0]},
                           controller_observation=lambda:{'time':now[0]})
            s.active_round='one'
            ticket=dict(method='pickup_box',args={'object_id':'brown_box'},round_id='one',
                        done=threading.Event(),cancelled=False)
            s.queue.put(ticket)
            s.tick(dict(time=0.,box_mass_kg=secret,robot_source_peak_N=secret))
            c.command({'time':now[0]})
            now[0]=.02;ticket['cancelled']=True
            s.tick(dict(time=.02,box_mass_kg=secret,robot_source_peak_N=secret))
            s.finish('requested_stop');s.close()
        return seen,c.last_action

    def test_session_start_update_cancel_finish_do_not_receive_hidden_records(self):
        first=self.run_session(0.)
        self.assertEqual(first,self.run_session(999.))
        self.assertEqual({name for name,obs in first[0]},{'start','update','cancel'})
        self.assertTrue(all(set(obs)=={'time'} for name,obs in first[0]))


if __name__=='__main__':unittest.main()
