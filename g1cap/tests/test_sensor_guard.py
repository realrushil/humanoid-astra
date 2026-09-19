import math
import unittest

from g1cap.arena_sensors import proprioception_packet
from g1cap.hand_sensors import DEX3_JOINTS,hand_position_packet
from g1cap.sensor_guard import SensorGuard


class SensorGuardTests(unittest.TestCase):
    def setUp(self):
        self.body_names=[f'body_{i}' for i in range(29)]
        self.names=self.body_names+list(DEX3_JOINTS)
        self.guard=SensorGuard(self.names,{n:10. for n in self.body_names})

    def update(self,step=0,torque=0.,up=(0.,0.,1.),front=None,hand=None):
        body=proprioception_packet(step=step,time_s=step*.02,joint_names=self.body_names,
            q=[0.]*29,dq=[0.]*29,tau_est=[torque]+[0.]*28,gyro=[0.]*3,accel=[0.,0.,9.81])
        hands=hand_position_packet(step=step,time_s=step*.02,left=[0.]*7,right=[0.]*7)
        return self.guard.update(body,hands,list(up),
            front or {'status':'unavailable'},hand or {'status':'unavailable'})

    def test_initialization_allows_unknown_geometry_but_never_unknown_body(self):
        self.assertIsNone(self.update()['fault'])
        self.assertIsNone(self.guard.fault(0.))
        self.assertEqual(self.guard.fault(.02),'sensor_guard_stale')
        with self.assertRaises(ValueError):SensorGuard(self.names,{})

    def test_tilt_uses_estimated_up_and_latches(self):
        angle=math.radians(16)
        self.assertEqual(self.update(up=(math.sin(angle),0.,math.cos(angle)))['fault'],'sensor_body_tilt_limit')
        self.assertEqual(self.update(1)['fault'],'sensor_body_tilt_limit')

    def test_saturation_requires_continuous_half_second_and_resets_below_threshold(self):
        for step in range(25):self.assertIsNone(self.update(step,9.5)['fault'])
        self.assertIsNone(self.update(25,9.4)['fault'])
        for step in range(26,51):self.assertIsNone(self.update(step,9.5)['fault'])
        self.assertEqual(self.update(51,9.5)['fault'],'sensor_effort_saturation')

    def test_gap_is_not_counted_as_a_long_saturated_interval(self):
        self.update(0,10.)
        self.assertEqual(self.update(30,10.)['fault'],'sensor_sample_gap')

    def test_geometry_uses_observed_local_clearance_and_allows_outside_front_band(self):
        self.assertIsNone(self.update(hand={'status':'available','margin_m':None})['fault'])
        self.assertEqual(self.update(1,front={'status':'available','clearance_m':-.001})['fault'],
                         'sensor_front_clearance_limit')
        self.setUp()
        self.assertEqual(self.update(hand={'status':'available','margin_m':-.001})['fault'],
                         'sensor_hand_clearance_limit')

    def test_nonfinite_missing_and_duplicate_samples_reject(self):
        for up in [(0.,0.,float('nan')),(0.,0.,0.)]:
            self.setUp()
            self.assertEqual(self.update(up=up)['fault'],'invalid_sensor_measurement')
        self.setUp();self.update()
        self.assertEqual(self.update()['fault'],'sensor_sample_gap')


if __name__=='__main__':unittest.main()
