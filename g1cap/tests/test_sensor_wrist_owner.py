"""Exercise the production sensor-only wrist owner without importing Isaac."""
import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace,MethodType
import unittest
try:
    import numpy as np
    import pinocchio as pin
except ImportError:pin=None

@unittest.skipIf(pin is None,'native numerical runtime required')
class SensorWristOwnerTests(unittest.TestCase):
    def setUp(self):
        from g1cap.arena_control import BoxControl
        from g1cap.arena_perception import ArenaBoxPerception
        from g1cap.arena_sensors import proprioception_packet
        self.packet_factory=proprioception_packet
        root=Path(__file__).resolve().parents[1]
        module=ast.parse((root/'g1cap/arena_world.py').read_text())
        klass=next(n for n in module.body if isinstance(n,ast.ClassDef) and n.name=='ArenaWorld')
        selected={'scene_hold_command','sensor_turn_observation','sensor_turn_controller','loaded_stop_command'}
        methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in selected]
        namespace={'__package__':'g1cap','json':json}
        exec(compile(ast.Module(body=methods,type_ignores=[]),str(root/'g1cap/arena_world.py'),'exec'),namespace)
        model=pin.buildModelFromUrdf(str(root/'g1cap/assets/arena_g1_rev1_0_kinematics.urdf'))
        names=list(model.names)[1:];self.body=[n for n in names if 'hand_' not in n];self.q={n:0. for n in self.body}
        for side in ('left','right'):self.q[side+'_shoulder_pitch_joint']=-.5;self.q[side+'_elbow_joint']=.6
        self.action=[self.q.get(n,.1) for n in names]+[0.,0.,0.,.75,0.,0.,0.]
        self.grasp=dict(status='available',time_s=0.,gap_m=.052,ready=False,raised=False,
            settled_retention=True,opposing_near_wrists=True,attitude_ok=True)
        self.gyro=[0.,0.,0.]
        self.floor=dict(status='observed_candidate',normal_body=[0,0,1],offset_m=.75)
        perception=ArenaBoxPerception(model);perception.advance_imu(0.,self.gyro)
        perception.latest_carry_frame=perception.carry_frame.update(0.,perception.motion.rotation,self.floor)
        perception.feedback=lambda t:dict(self.grasp)
        w=SimpleNamespace(term=SimpleNamespace(sensor_model=model),names=names,scene_wrist_hold=None,
            scene_wrist_failure=None,floor_hold_pending=False,floor_hold_active=False,lift_anchor_m=0.,
            sensor_recorder=SimpleNamespace(latest_packet=self.packet(0.)),files={'scene-hold':io.StringIO()},
            box_perception=perception)
        w.scene_hold_command=MethodType(namespace['scene_hold_command'],w)
        w.loaded_stop_command=MethodType(namespace['loaded_stop_command'],w)
        w.sensor_turn_observation=MethodType(namespace['sensor_turn_observation'],w)
        w.sensor_turn_controller=MethodType(namespace['sensor_turn_controller'],w)
        w.control=BoxControl(self.action,lambda obs:self.action,None,visual_grasp=w.box_perception.feedback,
            scene_hold=w.scene_hold_command,approach_feedback=lambda t:{'status':'available'},
            hand_clearance_feedback=lambda t:{'status':'available'},sensor_fault=lambda t:None,
            sensor_lift_ready=lambda:w.floor_hold_active and not w.floor_hold_pending and w.scene_wrist_failure is None,
            sensor_turn_factory=w.sensor_turn_controller)
        self.world=w

    def packet(self,t):
        return self.packet_factory(step=round(t/.02),time_s=t,joint_names=self.body,q=list(self.q.values()),
            dq=[0.]*29,tau_est=[0.]*29,gyro=self.gyro,accel=[0.,0.,9.81])

    def step(self,i):
        t=i*.02;w=self.world;w.sensor_recorder.latest_packet=self.packet(t)
        if t>w.box_perception.motion.imu_time:w.box_perception.advance_imu(t,self.gyro)
        if i%5==0:
            if t>w.box_perception.carry_frame.time:
                w.box_perception.latest_carry_frame=w.box_perception.carry_frame.update(t,w.box_perception.motion.rotation,self.floor)
            self.grasp['time_s']=t
        a=w.control.command({'time':t});w.control.update({'time':t});return a

    def test_between_image_admission_keeps_preload_then_lifts_without_new_writer(self):
        c=self.world.control
        self.assertEqual(c.start('raise_held_box',{}, {'time':.04})['status'],'running')
        for i in range(2,6):a=self.step(i)
        np.testing.assert_allclose(a,self.action,atol=1e-7)
        hold=self.world.scene_wrist_hold;preload=hold.preload.copy()
        for i in range(6,31):self.step(i)
        self.assertGreater(c.lift_used_m,0.)
        self.assertIs(self.world.scene_wrist_hold,hold)
        np.testing.assert_allclose(hold.preload,preload,atol=0.)
        for actual,anchor in zip(hold.targets,hold.anchor_targets):
            np.testing.assert_allclose(actual.translation-anchor.translation,[0.,0.,c.lift_used_m],atol=1e-12)
        for i,(a,b) in enumerate(zip(c.last_action,self.action)):
            if i not in hold.action_ids:self.assertAlmostEqual(a,b,places=6)
        targets=[t.copy() for t in hold.targets]
        c.cancel('worker_request_cancelled',{'time':.6})
        for i in range(31,36):self.step(i)
        for a,b in zip(targets,hold.targets):np.testing.assert_allclose(a.homogeneous,b.homogeneous,atol=0.)
        self.assertTrue(c.lift_failed)

    def test_completed_lift_continues_through_idle_and_carry_without_reanchoring(self):
        c=self.world.control;c.start('raise_held_box',{}, {'time':0.})
        for i in range(21):self.step(i)
        self.grasp.update(gap_m=.086,ready=True,raised=True)
        for i in range(21,82):self.step(i)
        self.assertEqual(c.result['status'],'completed')
        hold=self.world.scene_wrist_hold;targets=[t.copy() for t in hold.targets]
        for i in range(82,91):self.step(i)
        self.world.scene_hold_command('sensor_retreat',1.8,c.last_action)
        self.assertIs(self.world.scene_wrist_hold,hold)
        for a,b in zip(targets,hold.targets):np.testing.assert_allclose(a.homogeneous,b.homogeneous,atol=0.)
        self.assertEqual(c.start('raise_held_box',{'clearance_m':.09},{'time':1.8})['status'],'running')
        used=c.lift_used_m
        for i in range(91,101):self.step(i)
        self.assertGreater(c.lift_used_m,used)
        self.assertIs(self.world.scene_wrist_hold,hold)

    def test_pickup_and_idle_use_each_imu_sample_without_reanchoring(self):
        c=self.world.control;c.method='pickup_box';c.phase='verify_pickup';self.grasp.update(gap_m=.09,raised=True)
        self.step(0);hold=self.world.scene_wrist_hold
        self.assertEqual(hold.frame_key,'body_in_control_frame')
        targets=[p.copy() for p in hold.targets];preload=hold.preload.copy()
        self.gyro=[0.,.1,0.]
        before=list(c.last_action);self.step(1);self.step(2)
        self.assertEqual(hold.updated_at,.04)
        self.assertEqual(self.world.box_perception.latest_carry_frame['time_s'],0.)
        self.assertGreater(max(abs(a-b) for a,b in zip(before,c.last_action)),0.)
        c.cancel('worker_request_cancelled',{'time':.04});self.step(3)
        self.assertIs(self.world.scene_wrist_hold,hold)
        np.testing.assert_allclose(hold.preload,preload,atol=0.)
        for a,b in zip(targets,hold.targets):np.testing.assert_allclose(a.homogeneous,b.homogeneous,atol=0.)
        for i,(a,b) in enumerate(zip(before,c.last_action)):
            if i not in hold.action_ids:self.assertAlmostEqual(a,b,places=6)

    def test_unavailable_floor_invalidates_hold_even_when_local_grasp_is_visible(self):
        c=self.world.control;c.method='pickup_box';c.phase='verify_pickup';self.step(0)
        self.floor=dict(status='unavailable')
        for i in range(1,6):self.step(i)
        self.assertEqual(c.result['reason'],'floor_control_frame_unavailable')
        hold=self.world.scene_wrist_hold;old_segment=hold.segment
        self.floor=dict(status='observed_candidate',normal_body=[0,0,1],offset_m=.75)
        for i in range(6,11):self.step(i)
        self.assertIs(self.world.scene_wrist_hold,hold)
        self.assertNotEqual(self.world.box_perception.latest_carry_frame['segment'],old_segment)
        self.assertEqual(self.world.scene_wrist_failure,'floor_control_frame_unavailable')

    def test_real_owner_continues_heading_after_turn_completion_wait_and_cancel(self):
        w=self.world;c=w.control;c.method='pickup_box';c.phase='verify_pickup'
        # This fixture begins from an already associated carried source plane.
        w.box_perception.source_plane=object()
        self.grasp.update(gap_m=.1,raised=True,ready=True)
        self.step(0);c.cancel('test_setup',{'time':0.})
        hold=w.scene_wrist_hold;preload=hold.preload.copy()
        self.step(1)
        self.assertEqual(c.start('turn_with_box',{'yaw_rad':.1},{'time':.02})['status'],'running')
        self.gyro=[0.,0.,.1]
        for i in range(2,121):
            if i==51:self.gyro=[0.,0.,0.]
            self.step(i)
        self.assertEqual(c.result['status'],'completed')
        self.assertEqual(c.start('wait',{'duration':.2},{'time':2.4})['status'],'running')
        target=hold.targets[0].rotation.copy();self.gyro=[0.,0.,.02]
        for i in range(121,132):self.step(i)
        self.assertGreater(np.max(abs(hold.targets[0].rotation-target)),1e-4)
        self.assertIs(w.scene_wrist_hold,hold)
        self.assertEqual(c.start('turn_with_box',{'yaw_rad':-.1},{'time':2.62})['status'],'running')
        self.step(132);c.cancel('worker_request_cancelled',{'time':2.64})
        target=hold.targets[0].rotation.copy()
        for i in range(133,138):self.step(i)
        self.assertEqual(c.last_action[43:46],[0.,0.,0.])
        self.assertGreater(np.max(abs(hold.targets[0].rotation-target)),1e-4)
        np.testing.assert_allclose(hold.preload,preload,atol=0)
        self.assertIs(w.scene_wrist_hold,hold)

    def test_height_only_loss_then_edge_loss_keeps_the_same_screened_owner(self):
        w=self.world;c=w.control;c.method='pickup_box';c.phase='verify_pickup'
        self.grasp.update(gap_m=.09,raised=True);self.step(0)
        hold=w.scene_wrist_hold;preload=hold.preload.copy();before=list(c.last_action)
        anchors=[p.homogeneous.copy() for p in hold.anchor_targets]
        w.loaded_stop_clearance=SimpleNamespace(retention=dict(time_s=0.,available=True),
            screen=lambda *args:dict(status='clear',lower_m=.04,required_margin_m=.03))
        w.loaded_stop_evidence_error=None;w.sensor_guard=SimpleNamespace(fault=lambda t:None)
        w.files['loaded-stop']=io.StringIO();w.sensor_recorder.latest_hands={}
        c.loaded_stop=w.loaded_stop_command
        # Height precision can fail while BOTH ordinary collision callbacks
        # remain available. The later edge loss must not find a poisoned owner.
        self.grasp.update(status='unavailable',reason='source_height_precision_insufficient')
        self.gyro=[0.,.1,0.];self.step(1)
        self.assertEqual(c.result['status'],'failed')
        self.assertEqual(c.result['reason'],'visual_state_unavailable')
        failed_result=dict(c.result)
        self.assertEqual(c.last_action[43:46],[0.,0.,0.])
        self.assertIs(w.scene_wrist_hold,hold)
        self.assertIsNone(w.scene_wrist_failure)
        self.assertEqual(hold.updated_at,.02)
        np.testing.assert_allclose(hold.preload,preload,atol=0.)
        self.assertEqual(w.loaded_stop_feedback['status'],'active')
        self.assertGreater(max(abs(a-b) for a,b in zip(before,c.last_action)),0.)
        last=json.loads(w.files['scene-hold'].getvalue().splitlines()[-1])
        self.assertEqual(last['phase'],'stopped');self.assertEqual(last['stop_screen']['status'],'clear')
        c.hand_clearance_feedback=lambda t:dict(status='unavailable')
        before=list(c.last_action);self.step(2)
        self.assertEqual(c.result,failed_result)
        self.assertEqual(hold.updated_at,.04)
        self.assertGreater(max(abs(a-b) for a,b in zip(before,c.last_action)),0.)
        self.assertEqual([r['time_s'] for r in map(json.loads,w.files['loaded-stop'].getvalue().splitlines())],[.02,.04])
        for actual,anchor in zip(hold.anchor_targets,anchors):
            np.testing.assert_allclose(actual.homogeneous,anchor,atol=0.)
        # An actual clearance rejection prevents all later corrections under
        # this owner. It cannot silently reanchor or declare the task complete.
        w.loaded_stop_clearance.screen=lambda *args:dict(status='insufficient_margin',lower_m=.02,required_margin_m=.03)
        before=list(c.last_action);self.step(3);self.step(4)
        self.assertEqual(c.last_action,before)
        self.assertEqual(w.scene_wrist_failure,'coupled_stop_path_margin_insufficient')
        self.assertEqual(w.loaded_stop_feedback['status'],'failed')
        self.assertIs(w.scene_wrist_hold,hold)
        self.assertEqual(c.result,failed_result)

    def test_height_loss_during_held_wait_and_idle_uses_screened_stop(self):
        w=self.world;c=w.control;c.method='pickup_box';c.phase='verify_pickup'
        self.grasp.update(gap_m=.09,raised=True);self.step(0)
        hold=w.scene_wrist_hold;preload=hold.preload.copy()
        anchors=[p.homogeneous.copy() for p in hold.anchor_targets]
        c.cancel('test_setup',{'time':0.})
        self.assertEqual(c.start('wait',{'duration':.02},{'time':0.})['status'],'running')
        w.loaded_stop_clearance=SimpleNamespace(retention=dict(time_s=0.,available=True),
            screen=lambda *args:dict(status='clear',lower_m=.04,required_margin_m=.03))
        w.loaded_stop_evidence_error=None;w.sensor_guard=SimpleNamespace(fault=lambda t:None)
        w.files['loaded-stop']=io.StringIO();w.sensor_recorder.latest_hands={}
        c.loaded_stop=w.loaded_stop_command
        self.grasp.update(status='unavailable',reason='source_height_precision_insufficient')
        self.gyro=[0.,.1,0.]
        for i in (1,2):
            before=list(c.last_action);self.step(i)
            self.assertIsNone(w.scene_wrist_failure)
            self.assertIs(w.scene_wrist_hold,hold)
            self.assertEqual(w.loaded_stop_feedback['status'],'active')
            self.assertEqual(hold.updated_at,i*.02)
            self.assertGreater(max(abs(a-b) for a,b in zip(before,c.last_action)),0.)
            self.assertEqual(c.last_action[43:46],[0.,0.,0.])
        # Completing a wait measures only elapsed time, not successful holding.
        self.assertEqual(c.result['reason'],'dwell_elapsed')
        np.testing.assert_allclose(hold.preload,preload,atol=0.)
        for actual,anchor in zip(hold.anchor_targets,anchors):
            np.testing.assert_allclose(actual.homogeneous,anchor,atol=0.)

    def test_height_loss_with_missing_retention_rejects_correction(self):
        w=self.world;c=w.control;c.method='pickup_box';c.phase='verify_pickup'
        self.grasp.update(gap_m=.09,raised=True);self.step(0)
        hold=w.scene_wrist_hold;before=list(c.last_action)
        w.loaded_stop_clearance=SimpleNamespace(retention=None)
        w.loaded_stop_evidence_error=None;w.sensor_guard=SimpleNamespace(fault=lambda t:None)
        w.files['loaded-stop']=io.StringIO();w.sensor_recorder.latest_hands={}
        c.loaded_stop=w.loaded_stop_command
        self.grasp.update(status='unavailable',reason='source_height_precision_insufficient')
        self.gyro=[0.,.1,0.];self.step(1)
        self.assertEqual(c.last_action,before)
        self.assertEqual(c.result['reason'],'visual_state_unavailable')
        self.assertEqual(c.stopping_failure,'stop_visual_retention_unavailable')
        self.assertEqual(w.loaded_stop_feedback['status'],'failed')
        self.assertEqual(hold.updated_at,0.)
        self.assertIs(w.scene_wrist_hold,hold)
