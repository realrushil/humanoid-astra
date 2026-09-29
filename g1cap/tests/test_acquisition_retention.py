"""Stop an observed failed grasp attempt without banning pregrasp approach."""
import unittest
from g1cap.acquisition_retention import AcquisitionRetention


def sample(t,gap=0.,opposed=False):
    return dict(status='available',time_s=t,gap_m=gap,opposing_near_wrists=opposed,
                gap_reference='observed_under_box_plane')

class AcquisitionRetentionTests(unittest.TestCase):
    def lifted(self):
        guard=AcquisitionRetention()
        self.assertIsNone(guard.update(0.,sample(0.,.03,True)))
        self.assertIsNone(guard.update(.04,sample(.04,.03,True)))
        return guard
    def test_pregrasp_approach_is_not_a_lost_grasp(self):
        guard=AcquisitionRetention()
        for i in range(50):self.assertIsNone(guard.update(i*.04,sample(i*.04)))
    def test_two_lift_images_then_sustained_near_plane_loss_fails(self):
        guard=self.lifted()
        for i in range(8):self.assertIsNone(guard.update(.08+i*.04,sample(.08+i*.04)))
        self.assertEqual(guard.update(.40,sample(.40)),'acquisition_grasp_attempt_lost')
    def test_one_lift_image_or_opposed_supported_box_is_insufficient(self):
        guard=AcquisitionRetention();guard.update(0.,sample(0.,.04,True))
        for i in range(1,20):self.assertIsNone(guard.update(i*.04,sample(i*.04)))
        guard=self.lifted()
        for i in range(2,30):self.assertIsNone(guard.update(i*.04,sample(i*.04,0.,True)))
    def test_brief_loss_or_airborne_box_does_not_confirm_return_to_plane(self):
        guard=self.lifted()
        for i in range(2,8):self.assertIsNone(guard.update(i*.04,sample(i*.04)))
        for i in range(8,30):self.assertIsNone(guard.update(i*.04,sample(i*.04,.03,False)))
    def test_repeated_stale_future_missing_and_gapped_samples_cannot_complete_window(self):
        for bad in ('repeat','stale','future','missing','gap'):
            guard=self.lifted();guard.update(.08,sample(.08))
            for i in range(1,9):
                t=.08+i*.04;s=sample(t)
                if bad=='repeat':s['time_s']=.08
                if bad=='stale':s['time_s']=t-.2
                if bad=='future':s['time_s']=t+.01
                if bad=='missing':s.pop('gap_m')
                if bad=='gap':t+=.3;s['time_s']=t
                self.assertIsNone(guard.update(t,s),bad)
    def test_reference_change_does_not_complete_previous_loss_window(self):
        guard=self.lifted()
        for i in range(2,8):guard.update(i*.04,sample(i*.04))
        s=sample(.32);s['gap_reference']='source_height_plane'
        self.assertIsNone(guard.update(.32,s))
        for i in range(9,14):self.assertIsNone(guard.update(i*.04,sample(i*.04)))

    def test_fresh_source_height_can_detect_a_failed_partial_lift(self):
        guard=AcquisitionRetention()
        for i in range(11):
            t=i*.04
            s=sample(t,.03 if i<2 else 0.,i<2)
            s['gap_reference']='source_height_plane'
            result=guard.update(t,s)
            if i<10:self.assertIsNone(result)
        self.assertEqual(result,'acquisition_grasp_attempt_lost')

    def test_reference_change_requires_new_lift_evidence_in_both_directions(self):
        for first,second in (('observed_under_box_plane','source_height_plane'),
                             ('source_height_plane','observed_under_box_plane')):
            guard=AcquisitionRetention()
            for i in range(24):
                # Old-reference lift must not authorize a later failure on a
                # different reference, even after a full loss dwell.
                t=i*.04;lift=i<2 or i in (13,14)
                s=sample(t,.03 if lift else 0.,lift)
                s['gap_reference']=first if i<2 else second
                result=guard.update(t,s)
                if i<23:self.assertIsNone(result,(first,i))
            self.assertEqual(result,'acquisition_grasp_attempt_lost')

    def test_unrecognized_height_reference_cannot_supply_lift_evidence(self):
        guard=AcquisitionRetention()
        for i in range(20):
            t=i*.04;s=sample(t,.03 if i<2 else 0.,i<2)
            s['gap_reference']='remembered_source_height' if i<2 else 'source_height_plane'
            self.assertIsNone(guard.update(t,s))

class AcquisitionRetentionIntegrationTests(unittest.TestCase):
    def test_lost_attempt_ends_pickup_zeros_navigation_and_retry_has_new_evidence(self):
        from g1cap.arena_control import BoxControl
        current=sample(0.);calls=[]
        action=[0.]*43+[.1,0.,0.,.75,0.,0.,0.]
        def visual(now):return dict(current,raised=False,ready=False,attitude_ok=True)
        def acquire(obs):calls.append(obs['time']);return list(action)
        control=BoxControl(action,acquire,None,visual_grasp=visual,
            scene_hold=lambda phase,t,a:a,approach_feedback=lambda t:dict(status='available'),
            hand_clearance_feedback=lambda t:dict(status='available'),sensor_fault=lambda t:None)
        control.update({'time':0.})
        self.assertEqual(control.start('pickup_box',{'object_id':'brown_box'},{'time':0.})['status'],'running')
        for i in range(1,23):
            t=i*.02;camera_t=(i//2)*.04;lift=.04<=camera_t<=.08
            current.clear();current.update(sample(camera_t,.03 if lift else 0.,lift))
            control.update({'time':t});command=control.command({'time':t})
        self.assertEqual(control.result['status'],'failed')
        self.assertEqual(control.result['reason'],'acquisition_grasp_attempt_lost')
        self.assertEqual(command[43:46],[0.,0.,0.])
        count=len(calls);control.command({'time':.44});self.assertEqual(len(calls),count)
        self.assertEqual(control.start('pickup_box',{'object_id':'brown_box'},{'time':.44})['status'],'running')
        for i in range(23,43):
            t=i*.02;current.clear();current.update(sample((i//2)*.04))
            control.update({'time':t});control.command({'time':t})
        self.assertIsNone(control.result)
        self.assertEqual(control.phase,'acquire')
        self.assertEqual(control.last_observation_time,.84)
