import unittest
from g1cap.sensor_lift import ClearanceLift
from g1cap.arena_control import BoxControl

def grasp(t,gap=.052,ready=False):
    return dict(status='available',time_s=t,gap_m=gap,opposing_near_wrists=True,
        attitude_ok=True,settled_retention=True,ready=ready)

class SensorLiftTests(unittest.TestCase):
    def test_clearance_requires_elapsed_second_at_fast_and_mixed_camera_rates(self):
        for hz in (25,50):
            c=ClearanceLift(0.,.08)
            for i in range(hz):
                r=c.update(i/hz,grasp(i/hz,.086,True))
                self.assertIsNone(r['outcome'],(hz,i))
            self.assertEqual(c.update(1.,grasp(1.,.086,True))['outcome'],'completed',hz)
        mixed=[0.,.1,.2]+[round(.24+.04*i,10) for i in range(18)]
        c=ClearanceLift(0.,.08)
        for t in mixed:
            r=c.update(t,grasp(t,.086,True))
        self.assertIsNone(r['outcome'])
        for t in (.96,1.):r=c.update(t,grasp(t,.086,True))
        self.assertEqual(r['outcome'],'completed')

    def test_fast_camera_below_goal_dip_requires_new_full_second(self):
        for hz in (25,50):
            c=ClearanceLift(0.,.08)
            for i in range(int(1.48*hz)+1):
                t=round(i/hz,10)
                gap=.084 if abs(t-.48)<1e-9 else .086
                r=c.update(t,grasp(t,gap,True))
            self.assertIsNone(r['outcome'],hz)
            for i in range(int(1.48*hz)+1,int(1.52*hz)+1):
                t=round(i/hz,10)
                r=c.update(t,grasp(t,.086,True))
            self.assertEqual(r['outcome'],'completed',hz)
        c=ClearanceLift(0.,.08)
        mixed=[0.,.1,.2]+[round(.24+.04*i,10) for i in range(33)]
        for t in mixed:
            r=c.update(t,grasp(t,.084 if t==.48 else .086,True))
            if t<=1.48:self.assertIsNone(r['outcome'],t)
        self.assertEqual(r['outcome'],'completed')

    def test_measured_gap_drives_rate_limited_target_and_sustained_completion(self):
        c=ClearanceLift(0.,.08)
        for i in range(21):r=c.update(i*.1,grasp(i*.1))
        self.assertAlmostEqual(r['offset_m'],.02)
        for i in range(21,31):
            self.assertIsNone(c.update(i*.1,grasp(i*.1,.086,True))['outcome'])
        self.assertEqual(c.update(3.1,grasp(3.1,.086,True))['outcome'],'completed')

    def test_height_request_is_parameterized_and_dip_restarts_window(self):
        c=ClearanceLift(0.,.09)
        for i in range(10):self.assertIsNone(c.update(i*.1,grasp(i*.1,.096,True))['outcome'])
        self.assertIsNone(c.update(1.,grasp(1.,.092,True))['outcome'])
        for i in range(11,21):self.assertIsNone(c.update(i*.1,grasp(i*.1,.096,True))['outcome'])
        self.assertEqual(c.update(2.1,grasp(2.1,.096,True))['outcome'],'completed')

    def test_previous_lift_consumes_same_grasp_budget(self):
        c=ClearanceLift(0.,.08,.035)
        for i in range(16):r=c.update(i*.1,grasp(i*.1))
        self.assertAlmostEqual(r['offset_m'],.04)
        self.assertEqual(r['reason'],'clearance_lift_budget_exhausted')
        self.assertEqual(r['outcome'],'failed')

    def test_cached_image_cannot_increase_motion_and_stale_or_reordered_rejects(self):
        c=ClearanceLift(0.)
        c.update(0.,grasp(0.));self.assertEqual(c.update(.08,grasp(0.))['offset_m'],0.)
        with self.assertRaises(ValueError):c.update(.2,grasp(0.))
        c=ClearanceLift(0.);c.update(.1,grasp(.1))
        with self.assertRaises(ValueError):c.update(.12,grasp(0.))

    def test_invalid_or_unsettled_retention_and_invalid_parameters_reject(self):
        for field,value in [('settled_retention',False),('status','unavailable'),
            ('gap_m',.019),('gap_m',float('nan')),('time_s',True),
            ('opposing_near_wrists',False),('attitude_ok',False)]:
            c=ClearanceLift(0.);g=grasp(0.);g[field]=value
            with self.assertRaises(ValueError):c.update(0.,g)
            self.assertEqual(c.offset,0.)
        for target,used in [(True,0.),(.11,0.),(.04,0.),(.08,.041),(.08,-.001)]:
            with self.assertRaises(ValueError):ClearanceLift(0.,target,used)

    def test_no_motion_above_goal_but_readiness_and_deadline_still_required(self):
        c=ClearanceLift(0.)
        for i in range(81):r=c.update(i*.1,grasp(i*.1,.086,False))
        self.assertEqual(r['offset_m'],0.)
        self.assertEqual(r['reason'],'clearance_lift_timeout')


class SensorLiftControlTests(unittest.TestCase):
    def controller(self):
        self.camera=grasp(0.);self.wrist_ready=True;self.action=[.2]*43+[0.]*7
        self.camera.update(raised=False,scene_status='available')
        return BoxControl(self.action,lambda obs:self.action,None,
            visual_grasp=lambda t:dict(self.camera),scene_hold=lambda phase,t,a:a,
            approach_feedback=lambda t:{'status':'available'},
            hand_clearance_feedback=lambda t:{'status':'available'},sensor_fault=lambda t:None,
            sensor_lift_ready=lambda:self.wrist_ready)

    def step(self,c,i,gap=.052,ready=False):
        t=i*.02
        if i%5==0:self.camera.update(time_s=t,gap_m=gap,ready=ready,raised=gap>=.055)
        a=c.command({'time':t});c.update({'time':t})
        self.assertEqual(a[43:46],[0.,0.,0.])
        return a

    def test_explicit_raise_accepts_low_settled_grasp_and_retains_consumed_budget(self):
        c=self.controller();c.update({'time':0.})
        self.assertEqual(c.start('raise_held_box',{'clearance_m':.08},{'time':0.})['status'],'running')
        for i in range(101):self.step(c,i)
        self.assertAlmostEqual(c.lift_used_m,.02)
        for i in range(101,157):self.step(c,i,.086,True)
        self.assertEqual(c.result['status'],'completed')
        used=c.lift_used_m
        self.assertEqual(c.start('raise_held_box',{'clearance_m':.09},{'time':3.12})['status'],'running')
        self.assertEqual(c.clearance_lift.offset,used)
        c.cancel('worker_request_cancelled',{'time':3.12})
        self.assertTrue(c.lift_failed)
        self.assertEqual(c.start('raise_held_box',{}, {'time':3.12})['reason'],'clearance_lift_invalidated')
        self.assertEqual(c.start('pickup_box',{'object_id':'brown_box'},{'time':3.12})['reason'],
                         'retained_grasp_requires_raise_or_hold')
        self.assertEqual(c.lift_used_m,used)

    def test_pickup_enters_bounded_recovery_without_oracle_or_new_acquisition(self):
        c=self.controller();c.method='pickup_box';c.phase='verify_pickup';c.phase_started=0.
        c.update({'time':0.})
        for i in range(1,52):self.step(c,i)
        self.assertEqual(c.phase,'sensor_lift')
        self.assertEqual(c.method,'pickup_box')
        self.assertIsNotNone(c.clearance_lift)

    def test_handoff_waits_without_spending_budget_and_loss_invalidates(self):
        c=self.controller();self.wrist_ready=False
        self.assertEqual(c.start('raise_held_box',{}, {'time':0.})['status'],'running')
        for i in range(6):self.step(c,i)
        self.assertEqual(c.lift_used_m,0.)
        self.wrist_ready=True
        for i in range(6,16):self.step(c,i)
        self.assertGreater(c.lift_used_m,0.)
        self.camera['settled_retention']=False
        self.step(c,16)
        self.assertTrue(c.lift_failed)
        self.assertEqual(c.result['reason'],'lift_grasp_unsettled')

    def test_missing_hold_acknowledgement_and_geometry_loss_cannot_finish(self):
        c=self.controller();self.wrist_ready=False
        c.start('raise_held_box',{}, {'time':0.})
        for i in range(9):self.step(c,i)
        self.assertEqual(c.result['reason'],'lift_wrist_handoff_timeout')
        c=self.controller();c.start('raise_held_box',{}, {'time':0.})
        c.hand_clearance_feedback=lambda t:{'status':'unavailable'}
        self.step(c,0)
        self.assertEqual(c.result['reason'],'sensor_hand_clearance_unavailable')
