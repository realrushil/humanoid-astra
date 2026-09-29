import unittest
from g1cap.carry_admission import CarryAdmission
from g1cap.arena_control import BoxControl

def retained(t=1.,**extra):
    return dict(status='available',time_s=t,track_epoch=1,segment=1,retained=True,settled=True,**extra)

class CarryAdmissionTests(unittest.TestCase):
    def test_retention_alone_cannot_arm(self):
        g=CarryAdmission()
        self.assertFalse(g.observe(1.,retained()))
        with self.assertRaisesRegex(ValueError,'pickup_not_verified'):g.arm(1.,retained(),object())

    def test_verified_pickup_allows_later_owner_binding_without_gap(self):
        g=CarryAdmission();self.assertTrue(g.record_pickup(1.,retained()))
        owner=object();g.arm(1.1,retained(1.1),owner)
        moving=retained(1.12);moving['settled']=False
        self.assertTrue(g.observe(1.12,moving));g.require_owner(owner)

    def test_failed_or_unsettled_completion_cannot_create_proof(self):
        for change in ({'status':'unavailable'},{'settled':False},{'time_s':.8},{'time_s':1.1}):
            g=CarryAdmission();r=retained();r.update(change)
            self.assertFalse(g.record_pickup(1.,r))
            with self.assertRaises(ValueError):g.arm(1.02,retained(1.02),object())

    def test_lost_identity_or_retention_latches_until_new_pickup(self):
        for change in ({'retained':False},{'status':'unavailable'},{'track_epoch':2},{'segment':2}):
            g=CarryAdmission();g.record_pickup(1.,retained());r=retained(1.04);r.update(change)
            self.assertFalse(g.observe(1.04,r));self.assertFalse(g.observe(1.08,retained(1.08)))
            with self.assertRaises(ValueError):g.arm(1.08,retained(1.08),object())
            g.reset();self.assertTrue(g.record_pickup(1.1,retained(1.1)))

    def test_repeated_camera_expires_and_control_gap_cannot_bridge(self):
        for now,sample in ((1.16,retained()),(1.2,retained(1.2)),(.98,retained(.98))):
            g=CarryAdmission();g.record_pickup(1.,retained())
            self.assertFalse(g.observe(now,sample))

    def test_owner_replacement_or_loss_invalidates_armed_carry(self):
        for replacement in (None,object()):
            g=CarryAdmission();g.record_pickup(1.,retained());owner=object();g.arm(1.,retained(),owner)
            with self.assertRaisesRegex(ValueError,'owner_changed'):g.require_owner(replacement)
            self.assertFalse(g.observe(1.04,retained(1.04)))

    def test_missing_owner_does_not_arm_or_erase_verified_pickup(self):
        g=CarryAdmission();g.record_pickup(1.,retained())
        with self.assertRaisesRegex(ValueError,'owner_unavailable'):g.arm(1.02,retained(1.02),None)
        owner=object();g.arm(1.04,retained(1.04),owner);g.require_owner(owner)

    def test_rearming_with_missing_owner_latches_an_existing_owner_loss(self):
        g=CarryAdmission();g.record_pickup(1.,retained());owner=object();g.arm(1.,retained(),owner)
        with self.assertRaisesRegex(ValueError,'owner_changed'):g.arm(1.02,retained(1.02),None)
        self.assertFalse(g.observe(1.04,retained(1.04)))

class ControllerPickupEventTests(unittest.TestCase):
    def control(self):
        self.retention=retained()
        return BoxControl([0.]*50,lambda o:[0.]*50,None,
            visual_grasp=lambda t:dict(status='available',ready=True),scene_hold=lambda p,t,a:a,
            approach_feedback=lambda t:dict(status='available'),hand_clearance_feedback=lambda t:dict(status='available'),
            sensor_fault=lambda t:None,carry_retention=lambda t:self.retention)

    def test_only_completed_pickup_records_eligibility(self):
        for method,status,expected in [('pickup_box','completed',True),('pickup_box','failed',False),
                ('pickup_box','cancelled',False),('hold_box','completed',False),('raise_held_box','completed',False)]:
            c=self.control();c.method=method;c.phase='verify_pickup'
            c._finish(status,'test_outcome',{'time':1.})
            self.assertEqual(c.carry_admission.observe(1.,self.retention),expected,(method,status))

    def test_update_monitors_retention_even_after_operation_finished(self):
        c=self.control();c.method='pickup_box';c.phase='verify_pickup';c._finish('completed','verified',{'time':1.})
        self.retention=retained(1.02);self.retention['retained']=False
        c.update({'time':1.02})
        self.retention=retained(1.04)
        self.assertFalse(c.carry_admission.observe(1.04,self.retention))

    def test_new_admitted_pickup_erases_previous_proof(self):
        c=self.control();c.method='pickup_box';c.phase='verify_pickup';c._finish('completed','verified',{'time':1.})
        self.assertTrue(c.carry_admission.observe(1.,self.retention))
        result=c.start('pickup_box',{'object_id':'brown_box'},{'time':1.02})
        self.assertEqual(result['status'],'running')
        self.assertFalse(c.carry_admission.observe(1.02,retained(1.02)))

if __name__=='__main__':unittest.main()
