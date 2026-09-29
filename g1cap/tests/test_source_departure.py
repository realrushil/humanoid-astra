import unittest,math
from g1cap.source_departure import SourceDeparture

def sample(t,clear=False,settled=False,front=True):
    return dict(time_s=t,track_epoch=1,segment=1,retained=True,settled=settled,
                direction_available=front,front_normal_xy=[1.,0.],drive_clear=True,
                sweep=dict(status='clear' if clear else 'insufficient_margin',lower_m=.04 if clear else -.1,time_s=t,yaw_rad=-math.pi/3))
class DepartureTests(unittest.TestCase):
    def test_stops_for_sweep_then_requires_one_second_fresh_settling(self):
        c=SourceDeparture(0.,-math.pi/3,sample(0.))
        self.assertLess(c.update(.02,sample(.02))['navigation'][0],0)
        self.assertEqual(c.update(.04,sample(.04,True))['navigation'],[0.,0.,0.])
        for i in range(3,54):
            t=i*.02;r=c.update(t,sample(t,True,True,False))
            if i<53:self.assertIsNone(r['outcome'])
        self.assertEqual(r['outcome'],'completed')
    def test_front_loss_stops_without_claiming_distance_and_can_settle(self):
        c=SourceDeparture(0.,-math.pi/3,sample(0.))
        r=c.update(.02,sample(.02,front=False));self.assertEqual(r['navigation'],[0.,0.,0.]);self.assertIsNone(r['outcome'])
        for i in range(2,53):r=c.update(i*.02,sample(i*.02,True,True,False))
        self.assertEqual(r['outcome'],'completed')
    def test_retention_loss_and_track_change_latch_failure(self):
        for key,value in [('retained',False),('track_epoch',2),('segment',2)]:
            c=SourceDeparture(0.,-math.pi/3,sample(0.));s=sample(.02);s[key]=value
            self.assertEqual(c.update(.02,s)['outcome'],'failed')
            self.assertEqual(c.update(.04,sample(.04,True,True))['outcome'],'failed')
    def test_old_or_wrong_sweep_never_passes(self):
        for key,value in [('time_s',-.1),('yaw_rad',0.),('lower_m',float('nan'))]:
            c=SourceDeparture(0.,-math.pi/3,sample(0.));s=sample(.02,True,True);s['sweep'][key]=value
            r=c.update(.02,s);self.assertEqual(r['navigation'],[0.,0.,0.]);self.assertNotEqual(r['outcome'],'completed')
    def test_duplicate_image_never_accumulates_dwell(self):
        c=SourceDeparture(0.,-math.pi/3,sample(0.));s=sample(.02,True,True)
        for i in range(1,5):self.assertIsNone(c.update(i*.02,s)['outcome'])
        self.assertEqual(c.update(.2,s)['outcome'],'failed')
    def test_lost_margin_resets_dwell_and_does_not_restart_drive(self):
        c=SourceDeparture(0.,-math.pi/3,sample(0.))
        for i in range(1,30):c.update(i*.02,sample(i*.02,True,True))
        r=c.update(.6,sample(.6));self.assertEqual(r['navigation'],[0.,0.,0.])
        self.assertIsNone(c.update(.62,sample(.62,True,True))['outcome'])
    def test_no_motion_without_drive_screen(self):
        c=SourceDeparture(0.,-math.pi/3,sample(0.));s=sample(.02);s['drive_clear']=False
        self.assertEqual(c.update(.02,s)['navigation'],[0.,0.,0.])
    def test_settling_timeout_is_failure(self):
        c=SourceDeparture(0.,-math.pi/3,sample(0.))
        for i in range(1,253):r=c.update(i*.02,sample(i*.02,front=False))
        self.assertEqual(r['outcome'],'failed')
if __name__=='__main__':unittest.main()
