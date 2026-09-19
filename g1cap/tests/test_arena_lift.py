"""Supported-grasp recovery through the same controller and retained world history."""
import unittest
from g1cap.arena_control import BoxControl,METHODS
from test_arena_retreat import state

ACTION=[0.]*46+[.75,0.,0.,0.]
def observation(i,clearance=0.):
    row=state(i)
    row.update(clearance=clearance,supported=abs(clearance)<=.01,robot_source_peak_N=0.,tilt=0.)
    row['loaded_contacts']['maximum_hand_N']=15.
    row['box_pos'][2]+=clearance
    return row

class Motion:
    def command(self,*args):return args[-2]

class SupportedLiftIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.targets=[]
        def motion(obs,translation):self.targets.append(translation);return Motion()
        self.control=BoxControl(ACTION,lambda obs:ACTION,motion)
        for i in range(51):self.control.update(observation(i))

    def test_supported_lift_is_normal_method_and_rejects_unstable_touch(self):
        self.assertIn('lift_supported_box',METHODS)
        row=observation(51);row['bilateral']=False;self.control.update(row)
        result=self.control.start('lift_supported_box',{},row)
        self.assertEqual(result['reason'],'stable_supported_grasp_required')
        self.assertEqual(self.targets,[])

    def test_failed_pickup_can_continue_without_reset_or_missing_observations(self):
        earlier={'status':'failed','reason':'pose_hold_timeout'}
        self.control.result=earlier
        result=self.control.start('lift_supported_box',{},observation(50))
        self.assertEqual(result['status'],'running')
        self.assertEqual(self.targets,[[0.,0.,.06]])
        for i in range(51,452):
            action=self.control.command(observation(i-1,.06))
            self.assertEqual(action,ACTION)
            self.control.update(observation(i,.06))
        self.assertEqual(self.control.result['status'],'completed')
        self.assertEqual(self.control.result['reason'],'supported_grasp_lifted')
        self.assertEqual(earlier,{'status':'failed','reason':'pose_hold_timeout'})
        self.assertAlmostEqual(self.control.last_observation_time,9.02)
        self.assertEqual(self.control.start('hold_box',{'duration':1.},observation(451,.06))['status'],'running')

    def test_contact_fault_cannot_be_recovered_by_new_lift_request(self):
        row=observation(51);row['robot_source_peak_N']=6.;self.control.update(row)
        self.assertEqual(self.control.start('lift_supported_box',{},row)['reason'],'episode_failed')

    def test_force_limit_stops_lift_and_retains_navigation_zero(self):
        self.assertEqual(self.control.start('lift_supported_box',{},observation(50))['status'],'running')
        row=observation(51,.02);row['loaded_contacts']['maximum_hand_N']=26.
        self.control.update(row)
        self.assertEqual(self.control.result['reason'],'preparation_force_limit')
        self.assertEqual(self.control.phase,'idle')
        self.assertEqual(self.control.last_action[43:46],[0.,0.,0.])

if __name__=='__main__':unittest.main()
