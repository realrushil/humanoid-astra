"""Measured geometry recovery after a lost source; no simulator object truth."""
import ast,io,json,time,unittest
from pathlib import Path
from types import MethodType,SimpleNamespace
import numpy as np
from g1cap.arena_perception import ArenaBoxPerception
from g1cap.arena_control import BoxControl
from g1cap.source_stop import SourceStopClearance
from g1cap.sensor_robot_geometry import RobotGeometry
from g1cap.observed_approach import ApproachEstimate
from g1cap.observed_hand import HandClearanceEstimate,hand_shapes

class AcquisitionGeometryRecoveryTests(unittest.TestCase):
    def test_idle_recovery_earns_fresh_clearance_without_repairing_active_source(self):
        root=Path(__file__).parent/'fixtures/rotated-support'
        fixture=json.loads((root/'measurement.json').read_text());arrays=np.load(root/'head.npz')
        packet=fixture['packet'];hands=json.loads((root/'hands.json').read_text())
        geometry=RobotGeometry('g1cap/assets/arena_g1_rev1_0_kinematics.urdf',
            'g1cap/assets/arena_g1_rev1_0_bounds.json',packet['joint_names'])
        frontend=ArenaBoxPerception(geometry.model);prior=fixture['prior_box']
        frontend.track.initialize(time_s=prior['observed_at_s'],now_s=prior['observed_at_s'],
            center=prior['center_camera_m'],rotation=prior['axes_camera'],size=prior['dimensions_m'])
        tree=ast.parse(Path('g1cap/arena_world.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ArenaWorld')
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('observe_source_obstacles','observe_measured_source','begin_acquisition')]
        namespace={'__package__':'g1cap','time':time,'json':json,'np':np}
        exec(compile(ast.Module(body=methods,type_ignores=[]),'obstacle_observation','exec'),namespace)
        owner=SimpleNamespace(approach_geometry=geometry,approach=ApproachEstimate(),
            observed_hand=HandClearanceEstimate(),control=SimpleNamespace(phase='acquire'),
            box_perception=frontend,loaded_stop_clearance=SourceStopClearance(
                'g1cap/assets/arena_g1_rev1_0_kinematics.urdf',
                'g1cap/assets/arena_g1_rev1_0_bounds.json',packet['joint_names']+hands['joint_names']))
        # Isolated candidate overlays may add a measured-reference observer.
        # Bind its real implementation so this recovery test also exercises
        # its source-owner lifecycle, rather than replacing it with a no-op.
        owner.files={'source-references':io.StringIO()}
        if 'observe_measured_source' in namespace:
            owner.observe_measured_source=MethodType(namespace['observe_measured_source'],owner)
        camera=dict(calibration=fixture['calibration'],rgb=arrays['rgb'],depth_m=arrays['depth_m'])
        source=None
        for i in range(5):
            # Repeated measured image with synthetic increasing timestamps tests
            # estimator lifecycle only; it is not new physical sensor evidence.
            t=.28+i*.04;packet=dict(packet,step=14+i*2,time_s=t)
            camera=dict(camera,step=packet['step'],time_s=t)
            hands=dict(hands,step=packet['step'],time_s=t)
            frontend.advance_imu(t,packet['gyro_rad_s'])
            visual=frontend.update(packet,camera,fixture['up_body'])
            owner.approach.advance_imu(t,packet['gyro_rad_s'],geometry.lower_body_points(packet))
            owner.observed_hand.advance(t,packet['gyro_rad_s'],hand_shapes(geometry.model,geometry.bounds,packet,hands))
            if i==0:
                frontend.begin_carry(t);source=frontend.source_plane;observed_obstacle=source.geometry;source.plane=None
            if i>=2:
                owner.control.phase='idle'
                # Height precision may fail while measured obstacle geometry
                # remains. That must not authorize fallback-front memory.
                visual['source_obstacle']=observed_obstacle
            namespace['observe_source_obstacles'](owner,visual,packet,camera)
            if i==1:
                self.assertEqual(owner.approach.feedback(t)['status'],'unavailable')
                self.assertEqual(owner.observed_hand.feedback(t)['status'],'unavailable')
            if i==2:
                self.assertEqual(owner.approach.feedback(t)['status'],'unavailable')
                self.assertEqual(owner.observed_hand.feedback(t)['status'],'unavailable')
            if i>=3:
                self.assertEqual(owner.approach.feedback(t)['status'],'available')
                self.assertEqual(owner.observed_hand.feedback(t)['status'],'available')
                self.assertEqual(frontend.feedback(t)['status'],'unavailable')
                self.assertIs(frontend.source_plane,source);self.assertIsNone(source.plane)
                self.assertIsNone(owner.loaded_stop_clearance.front_memory)
        # Actual measured callbacks must admit explicit pickup without relaxing
        # geometry checks. Source replacement happens only in the begin hook.
        def begin(now):
            namespace['begin_acquisition'](owner,now)
        owner.box_perception=frontend;owner.policy=SimpleNamespace(reset=lambda:None)
        owner.step_index=packet['step'];owner.files['gr00t-resets']=io.StringIO()
        control=BoxControl([0.]*50,lambda obs:[0.]*50,None,
            acquisition_begin=begin,acquisition_grasp=frontend.acquisition_feedback,
            visual_grasp=frontend.feedback,scene_hold=lambda phase,t,a:a,
            approach_feedback=owner.approach.feedback,hand_clearance_feedback=owner.observed_hand.feedback,
            sensor_fault=lambda t:None,sensor_lift_ready=lambda:False)
        owner.control=control
        self.assertEqual(control.start('pickup_box',{'object_id':'brown_box'},{'time':t})['status'],'running')
        self.assertIsNot(frontend.source_plane,source)
        self.assertFalse(frontend.feedback(t)['ready'])

        t+=.04;packet=dict(packet,step=packet['step']+2,time_s=t)
        hands=dict(hands,step=packet['step'],time_s=t);camera=dict(camera,step=packet['step'],time_s=t)
        frontend.advance_imu(t,packet['gyro_rad_s'])
        visual=frontend.update(packet,camera,fixture['up_body'])
        owner.approach.advance_imu(t,packet['gyro_rad_s'],geometry.lower_body_points(packet))
        owner.observed_hand.advance(t,packet['gyro_rad_s'],hand_shapes(geometry.model,geometry.bounds,packet,hands))
        namespace['observe_source_obstacles'](owner,visual,packet,camera)
        self.assertEqual(owner.approach.feedback(t)['status'],'available')
        self.assertEqual(owner.observed_hand.feedback(t)['status'],'available')
