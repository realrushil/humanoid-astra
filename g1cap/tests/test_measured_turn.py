import math
import unittest
try:
    import numpy as np
except ImportError:
    np=None

def MeasuredHeadingTurn(*args, **kwargs):
    from g1cap.measured_turn import MeasuredHeadingTurn as Controller
    return Controller(*args, **kwargs)

def frame(angle=0.):
    c,s=math.cos(angle),math.sin(angle)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])

@unittest.skipIf(np is None,'numpy runtime required')
class TurnTests(unittest.TestCase):
    def test_signed_heading_and_shared_rotation(self):
        turn=MeasuredHeadingTurn(0.,-.5,frame(3.05),[0.,0.,1.])
        r=turn.update(.02,frame(2.95),False)
        self.assertAlmostEqual(r['yaw_rad'],-.1)
        self.assertLess(r['yaw_rate_command_rad_s'],0.)
        np.testing.assert_allclose(turn.rotation(),frame(-.1),atol=1e-12)

    def test_heading_crosses_pi_without_jump(self):
        turn=MeasuredHeadingTurn(0.,.5,frame(3.1),[0.,0.,1.])
        r=turn.update(.02,frame(-3.1),False)
        self.assertAlmostEqual(r['yaw_rad'],2*math.pi-6.2)

    def test_completion_requires_settling_and_ready(self):
        turn=MeasuredHeadingTurn(0.,.5,frame(),[0.,0.,1.])
        for i in range(1,56):
            r=turn.update(i*.02,frame(.49),False)
        self.assertIsNone(r['outcome'])
        r=turn.update(1.12,frame(.49),True)
        self.assertEqual(r['outcome'],'completed')
        self.assertEqual(r['yaw_rate_command_rad_s'],0.)

    def test_overshoot_reverses_after_settle(self):
        turn=MeasuredHeadingTurn(0.,.5,frame(),[0.,0.,1.])
        turn.update(.02,frame(.49),False)
        for i in range(2,53):r=turn.update(i*.02,frame(.58),False)
        self.assertLess(r['yaw_rate_command_rad_s'],0.)

    def test_missing_step_and_angle_envelope(self):
        with self.assertRaises(ValueError):MeasuredHeadingTurn(0.,math.pi,frame(),[0.,0.,1.])
        turn=MeasuredHeadingTurn(0.,.5,frame(),[0.,0.,1.])
        with self.assertRaises(ValueError):turn.update(.1,frame(),False)

    def test_finite_timeout(self):
        turn=MeasuredHeadingTurn(0.,.5,frame(),[0.,0.,1.])
        for i in range(1,501):r=turn.update(i*.02,frame(),False)
        self.assertEqual(r['outcome'],'failed')
        self.assertEqual(r['yaw_rate_command_rad_s'],0.)

    def test_active_minimum_stays_above_native_standing_switch(self):
        for sign in (-1.,1.):
            turn=MeasuredHeadingTurn(0.,sign*.5,frame(),[0,0,1],minimum_yaw_rate=.08)
            r=turn.update(.02,frame(sign*.45),False)
            self.assertAlmostEqual(r['yaw_rate_command_rad_s'],sign*.08)
            # The minimum must not prevent zero command in settling/completion.
            r=turn.update(.04,frame(sign*.49),False)
            self.assertEqual(r['yaw_rate_command_rad_s'],0.)
            for i in range(3,54):r=turn.update(i*.02,frame(sign*.49),True)
            self.assertEqual(r['outcome'],'completed')
            self.assertEqual(r['yaw_rate_command_rad_s'],0.)

    def test_minimum_does_not_change_cap_and_rejects_invalid_settings(self):
        turn=MeasuredHeadingTurn(0.,.5,frame(),[0,0,1],minimum_yaw_rate=.08)
        self.assertAlmostEqual(turn.update(.02,frame(),False)['yaw_rate_command_rad_s'],.18)
        for minimum in (0.,-.1,.19,float('nan'),True):
            with self.assertRaises(ValueError):
                MeasuredHeadingTurn(0.,.5,frame(),[0,0,1],minimum_yaw_rate=minimum)

    def sustained(self):
        return MeasuredHeadingTurn(0.,.5,frame(),[0,0,1],minimum_yaw_rate=.08,
                                   require_sustained_readiness=True)

    def test_fresh_camera_readiness_not_repeated_control_ticks_earns_completion(self):
        turn=self.sustained()
        # First ready image arrives after the initial settling second. Cached
        # copies at50Hz must not earn extra10Hz observation samples.
        for i in range(1,106):
            t=i*.02;camera=(i//5)*.1
            r=turn.update(t,frame(.49),camera>=1.1,observed_at_s=camera)
            if i<105:self.assertIsNone(r['outcome'])
            self.assertEqual(r['yaw_rate_command_rad_s'],0.)
        self.assertEqual(r['outcome'],'completed')
        self.assertEqual(r['ready_window']['fresh_samples'],11)
        self.assertAlmostEqual(r['ready_window']['first_camera_s'],1.1)

    def test_false_ready_or_heading_excursion_restarts_window(self):
        for fault in ('ready','heading'):
            with self.subTest(fault=fault):
                turn=self.sustained()
                for i in range(1,76):
                    t=i*.02;camera=(i//5)*.1
                    ready=not(fault=='ready' and i==25)
                    angle=.44 if fault=='heading' and i==25 else .49
                    r=turn.update(t,frame(angle),ready,observed_at_s=camera)
                    self.assertIsNone(r['outcome'])
                for i in range(76,81):
                    r=turn.update(i*.02,frame(.49),True,observed_at_s=(i//5)*.1)
                self.assertEqual(r['outcome'],'completed')
                self.assertAlmostEqual(r['ready_window']['first_camera_s'],.6)

    def test_expired_future_missing_and_reversed_camera_time_are_rejected(self):
        for camera in (None,-.2,.03,float('nan')):
            with self.subTest(camera=camera):
                with self.assertRaises(ValueError):self.sustained().update(.02,frame(.49),True,observed_at_s=camera)
        turn=self.sustained();turn.update(.02,frame(.49),True,observed_at_s=.02)
        with self.assertRaises(ValueError):turn.update(.04,frame(.49),True,observed_at_s=.01)

    def test_camera_gap_cannot_bridge_a_readiness_window(self):
        turn=self.sustained()
        for i in range(1,16):
            r=turn.update(i*.02,frame(.49),True,observed_at_s=0. if i<=7 else .16)
        # Both images were current on arrival, but their160ms separation
        # invalidates the first sample; repeat packets cannot restore it.
        self.assertEqual(r['ready_window']['fresh_samples'],1)
        self.assertAlmostEqual(r['ready_window']['first_camera_s'],.16)

if __name__=='__main__':unittest.main()
