import copy
import importlib
from pathlib import Path
import unittest

try:
    import numpy as np
    import pinocchio as pin
    import scipy.optimize
except ImportError:
    pin=None


@unittest.skipIf(pin is None,'native numerical runtime required')
class SceneWristHoldTests(unittest.TestCase):
    def test_heading_follow_persists_and_repeated_turn_does_not_reanchor_or_double_rotate(self):
        motion=dict(self.motion);motion['body_in_control_frame']=motion.pop('body_in_segment')
        hold=self.mod.SceneWristHold(self.model,self.names,self.packet,motion,self.reference,
                                    frame_key='body_in_control_frame')
        anchors=[p.copy() for p in hold.anchor_targets];preload=hold.preload.copy()
        hold.raise_targets(.02);hold.follow_heading(motion,[0,0,1])
        for i,angle in enumerate((.1,.2,-.1),1):
            motion['time_s']=i*.1
            rotation=pin.exp3(np.array([0.,0.,angle]))
            motion['body_in_control_frame']=pin.SE3(rotation,np.zeros(3)).homogeneous.tolist()
            # Ordinary hold updates must retain heading-follow after a tool ends.
            hold.update(self.packet_at(i*.1),motion)
            for a,b in zip(hold.targets,anchors):
                expected=pin.SE3(rotation,np.zeros(3))*b
                expected.translation[2]+=.02
                np.testing.assert_allclose(a.homogeneous,expected.homogeneous,atol=1e-12)
            hold.follow_heading(motion,[0,0,1])  # Next relative turn: idempotent.
            np.testing.assert_allclose(hold.preload,preload,atol=0)
        hold.raise_targets(.03)
        self.assertEqual(hold.vertical_lift_m,.03)
        motion['time_s']=.4;hold.update(self.packet_at(.4),motion)
        for a,b in zip(hold.targets,anchors):
            expected=pin.SE3(rotation,np.zeros(3))*b;expected.translation[2]+=.03
            np.testing.assert_allclose(a.homogeneous,expected.homogeneous,atol=1e-12)

    def test_heading_follow_requires_active_synchronized_floor_owner(self):
        hold=self.controller()
        with self.assertRaises(ValueError):hold.follow_heading(self.motion,[0,0,1])
    def test_equal_vertical_lift_preserves_preload_and_target_when_holding(self):
        hold=self.controller();original=[x.copy() for x in hold.targets]
        self.assertTrue(hasattr(hold,'raise_targets'),'paired lift API missing')
        hold.raise_targets(.02)
        for a,b in zip(hold.targets,original):
            np.testing.assert_allclose(a.translation-b.translation,[0.,0.,.02],atol=1e-12)
            np.testing.assert_allclose(a.rotation,b.rotation,atol=1e-12)
        preload=hold.preload.copy();hold.update(self.packet_at(.1),self.motion_at(.1))
        goal=hold.goal.copy();hold.update(self.packet_at(.2),self.motion_at(.2))
        np.testing.assert_allclose(hold.preload,preload,atol=0)
        np.testing.assert_allclose(hold.goal,goal,atol=1e-6)
        for value in (.01,.041,float('nan'),True):
            with self.assertRaises(ValueError):hold.raise_targets(value)
    def setUp(self):
        self.mod=importlib.import_module('g1cap.scene_wrist_hold')
        self.model=pin.buildModelFromUrdf(str(Path(__file__).parents[1]/'g1cap/assets/arena_g1_rev1_0_kinematics.urdf'))
        self.names=list(self.model.names)[1:]
        self.body=[n for n in self.names if 'hand_' not in n]
        self.q={n:0. for n in self.body}
        for side in ('left','right'):
            self.q[side+'_shoulder_pitch_joint']=-.5
            self.q[side+'_elbow_joint']=.6
        self.packet=self.packet_at(0.)
        self.reference=[self.q.get(n,.1) for n in self.names]+[0.,0.,0.,.75,0.,0.,0.]
        self.reference[self.names.index('left_elbow_joint')]+=.04
        self.motion=self.motion_at(0.)

    def packet_at(self,t):
        from g1cap.arena_sensors import proprioception_packet
        return proprioception_packet(step=round(t/.02),time_s=t,joint_names=self.body,
            q=list(self.q.values()),dq=[0.]*29,tau_est=[0.]*29,gyro=[0.]*3,accel=[0.,0.,9.81])

    def motion_at(self,t):return dict(status='tracked_local_segment',time_s=t,segment=1,body_in_segment=np.eye(4).tolist())

    def controller(self):return self.mod.SceneWristHold(self.model,self.names,self.packet,self.motion,self.reference)

    def test_stationary_hold_retains_loaded_reference_without_integrating_error(self):
        hold=self.controller()
        for t in (.1,.2,.3):
            hold.update(self.packet_at(t),self.motion_at(t))
            np.testing.assert_allclose(hold.goal,self.reference,atol=1e-6)
        self.assertAlmostEqual(hold.preload[hold.arm_names.index('left_elbow_joint')],.04)

    def test_floor_control_frame_is_explicit_and_not_published_as_scene_position(self):
        motion=dict(self.motion)
        motion['body_in_control_frame']=motion.pop('body_in_segment')
        motion['horizontal_position_observed']=False
        hold=self.mod.SceneWristHold(self.model,self.names,self.packet,motion,self.reference,
                                    frame_key='body_in_control_frame')
        motion['time_s']=.1
        hold.update(self.packet_at(.1),motion)
        np.testing.assert_allclose(hold.goal,self.reference,atol=1e-6)
        self.assertNotIn('body_in_segment',motion)
        with self.assertRaises(ValueError):
            self.mod.SceneWristHold(self.model,self.names,self.packet,self.motion,self.reference,
                                    frame_key='world_pose')

    def test_body_motion_has_paired_scene_targets_and_only_arm_commands_slew(self):
        hold=self.controller();motion=self.motion_at(.1)
        for t in (.02,.04,.06,.08):hold.command(t,self.reference)
        transform=np.eye(4);transform[:3,:3]=pin.exp3(np.array([0.,.025,0.]));transform[:3,3]=[.003,0.,-.008]
        motion['body_in_segment']=transform.tolist()
        result=hold.update(self.packet_at(.1),motion)
        self.assertLess(result['max_position_error_m'],.002)
        self.assertLess(result['max_rotation_error_rad'],.02)
        output=hold.command(.1,self.reference)
        self.assertAlmostEqual(max(abs(a-b) for a,b in zip(output,self.reference)),.02,places=6)
        for i in range(50):
            if i not in hold.action_ids:self.assertEqual(output[i],self.reference[i])
        self.assertTrue(any(abs(output[i]-self.reference[i])>1e-8 for i in hold.action_ids))

    def test_truth_fields_time_mismatch_and_changed_segment_are_not_consumed(self):
        for change in ('truth','time','segment','missing'):
            hold=self.controller();packet=self.packet_at(.1);motion=self.motion_at(.1)
            if change=='truth':packet['box_mass_kg']=.1
            if change=='time':motion['time_s']=.2
            if change=='segment':motion['segment']=2
            if change=='missing':motion.update(status='unavailable',body_in_segment=None)
            with self.assertRaises(ValueError):hold.update(packet,motion)
            with self.assertRaises(ValueError):hold.command(.02,self.reference)

    def test_unreachable_target_and_stale_or_reordered_commands_withhold_correction(self):
        hold=self.controller();motion=self.motion_at(.1);motion['body_in_segment'][0][3]=1.
        with self.assertRaises(ValueError):hold.update(self.packet_at(.1),motion)
        with self.assertRaises(ValueError):hold.command(.02,self.reference)
        hold=self.controller();hold.command(.02,self.reference)
        with self.assertRaises(ValueError):hold.command(.01,self.reference)
        with self.assertRaises(ValueError):self.controller().command(.2,self.reference)


if __name__=='__main__':unittest.main()
