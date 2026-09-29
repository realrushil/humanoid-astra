"""RGB identity/freshness and a bounded pixel path to generated Python."""
import base64
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np


class WristCameraContractTests(unittest.TestCase):
    def test_rgb_packet_is_bounded_and_contains_no_depth_or_hidden_geometry(self):
        from g1cap.camera_rgb import rgb_packet, public_rgb
        image=np.zeros((240,320,3),dtype=np.uint8);image[:,:,0]=123
        packet=rgb_packet(image,'left_wrist',10,.2,{'intrinsic':[[100,0,160],[0,100,120],[0,0,1]],
            'parent_frame':'left_wrist_yaw_link','world_pose':'private'})
        packet['depth_m']='private';packet['world_pose']='private'
        public=public_rgb(packet,.24)
        self.assertEqual(public['status'],'available')
        self.assertEqual(public['time_s'],.2)
        self.assertAlmostEqual(public['age_s'],.04)
        pixels=base64.b64decode(public['rgb_base64'])
        self.assertEqual(pixels[:3],bytes([123,0,0]))
        self.assertEqual(len(pixels),public['width']*public['height']*3)
        self.assertLess(len(json.dumps(public).encode()),8000)
        self.assertNotIn('depth_m',public);self.assertNotIn('world_pose',public)
        self.assertEqual(public['calibration']['intrinsic'][0],[100,0,160])
        self.assertNotIn('world_pose',public['calibration'])
        self.assertEqual(public_rgb(packet,.36)['status'],'unavailable')
        self.assertEqual(public_rgb(packet,.19)['status'],'unavailable')
        with self.assertRaises(ValueError):rgb_packet(image,'overview',10,.2,{})

    def test_wrist_configuration_is_rgb_only_and_does_not_change_head(self):
        from g1cap.camera_rgb import configure_wrist_cameras
        head=SimpleNamespace(prim_path='head',width=640,height=480,data_types=['rgb','distance_to_image_plane'],
            update_period=.02,offset=SimpleNamespace(pos=(0,0,0),rot=(1,0,0,0),convention='ros'),
            spawn=SimpleNamespace(focal_length=20.,clipping_range=(.1,100.)))
        scene=SimpleNamespace(robot_head_cam=head)
        configure_wrist_cameras(scene)
        self.assertEqual(head.data_types,['rgb','distance_to_image_plane'])
        for side in ['left','right']:
            camera=getattr(scene,side+'_wrist_cam')
            self.assertEqual(camera.data_types,['rgb'])
            self.assertIn(side+'_wrist_yaw_link',camera.prim_path)
            self.assertEqual((camera.width,camera.height),(320,240))
            self.assertEqual(camera.offset.convention,'world')
            # The retained installed CameraCfg.OffsetCfg uses XYZW. World
            # camera axes are x forward, z up before conversion to USD optical.
            from scipy.spatial.transform import Rotation
            axes=Rotation.from_quat(camera.offset.rot).apply(np.eye(3))
            angle=np.deg2rad(35)
            np.testing.assert_allclose(axes[0],[np.cos(angle),0,-np.sin(angle)],atol=1e-12)
            np.testing.assert_allclose(axes[2],[np.sin(angle),0,np.cos(angle)],atol=1e-12)

    def test_session_serves_pixels_but_cannot_request_overview(self):
        from g1cap.camera_rgb import rgb_packet
        from g1cap.arena_session import ArenaSession
        from test_arena_session import FakeControl
        frame=rgb_packet(np.zeros((20,20,3),np.uint8),'left_wrist',0,0.,{})
        with tempfile.TemporaryDirectory() as folder:
            session=ArenaSession(FakeControl(),{'backend':'arena'},folder,
                sensor_observation=lambda t:dict(time=t,step=0,physics_step=0),
                camera_observation=lambda: {'left_wrist':frame})
            try:
                session.tick({'time':0.,'step':0,'physics_step':0})
                session.active_round='r'
                result=session.dispatch('observe_camera',['left_wrist'],{},'r')
                self.assertEqual(result['camera'],'left_wrist')
                self.assertEqual(result['status'],'available')
                self.assertEqual(session.dispatch('observe_camera',['overview'],{},'r')['status'],'rejected')
                self.assertEqual(session.dispatch('observe_camera',['right_wrist'],{},'r')['status'],'unavailable')
                frame['time_s']=3.
                self.assertEqual(result['time_s'],0.)
                self.assertNotIn('rgb_base64',session.observe())
            finally:session.close()

    def test_real_worker_decodes_camera_pixels_with_standard_library(self):
        from g1cap.camera_rgb import rgb_packet, public_rgb
        from g1cap.execution import execute_policy
        image=np.full((240,320,3),123,dtype=np.uint8)
        packet=public_rgb(rgb_packet(image,'left_wrist',0,0.,
            {'intrinsic':np.eye(3).tolist(),'world_pose':'private'}),0.)
        source='''import base64
def run(robot, task):
    p=robot.observe_camera('left_wrist')
    pixels=base64.b64decode(p['rgb_base64'])
    assert len(pixels)==p['width']*p['height']*3
    assert pixels[0]==123
    assert p['calibration']['intrinsic'][0][0]==1
    assert 'world_pose' not in p['calibration']
    print(p['camera'],p['time_s'])
'''
        result=execute_policy(source,{},'camera-test',lambda *args:packet,
                              tools={'observe_camera'},wall_timeout=5.)
        self.assertEqual(result['status'],'completed',result)

    def test_sensor_snapshot_accepts_wrist_rgb_with_individual_capture_times(self):
        from g1cap.visual_observation import validate_snapshot
        from test_visual_observation import snapshot_fixture
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);packet=snapshot_fixture(root,10)
            packet['observation']['observation_mode']='sensor_estimates_v1'
            packet['observation']['camera_names']=['head','left_wrist','right_wrist']
            head=packet['frames'][0]
            packet['frames']=[dict(head,camera=name,time_s=.18,step=9,modality='rgb',calibration_id='a'*64)
                              for name in ['head','left_wrist','right_wrist']]
            validate_snapshot(packet,root,'session-a')
            packet['frames'][1]['time_s']=.3
            with self.assertRaises(ValueError):validate_snapshot(packet,root,'session-a')
            packet['frames'][1]['time_s']=.18;packet['frames'][1]['modality']='depth'
            with self.assertRaises(ValueError):validate_snapshot(packet,root,'session-a')

class RGBStreamTests(unittest.TestCase):
    def test_recorder_publishes_each_native_rgb_with_resolvable_static_calibration(self):
        from test_sensor_cadence import native, Tensor, recorder_class
        from g1cap.camera_rgb import public_rgb
        source,model=native();source.physics_dt=.005
        head=source.scene['robot_head_cam']
        for side in ('left','right'):
            wrist=deepcopy(head)
            del wrist.data.output['distance_to_image_plane']
            wrist.data.output['rgb']=Tensor(np.full((1,2,2,4),23 if side=='left' else 47,np.uint8))
            source.scene[side+'_wrist_cam']=wrist
        with tempfile.TemporaryDirectory() as folder:
            recorder=recorder_class()(Path(folder)/'sensors',source,model,rgbd_period_steps=2,wrist_cameras=True)
            try:
                for step in (0,2):
                    for name in ('robot_head_cam','left_wrist_cam','right_wrist_cam'):
                        sensor=source.scene[name]
                        sensor._timestamp=Tensor([step*.02])
                        sensor._timestamp_last_update=Tensor([step*.02])
                        sensor.frame=SimpleNamespace(warp=Tensor([step+1]))
                    recorder.capture(step,step*.02)
                    for name,color in (('left_wrist',23),('right_wrist',47)):
                        packet=public_rgb(recorder.rgb_stream.latest[name],step*.02)
                        self.assertEqual(base64.b64decode(packet['rgb_base64'])[0],color)
                        self.assertEqual(packet['calibration']['offset_quaternion_xyzw'],[0,0,0,1])
                        self.assertEqual(packet['calibration']['parent_frame'],name+'_yaw_link')
                        self.assertNotIn('depth',json.dumps(packet))
                        self.assertEqual(packet['step'],step)
            finally:recorder.close()

    def test_repeated_native_frame_cannot_become_a_new_capture(self):
        from g1cap.camera_rgb import RGBStream
        with tempfile.TemporaryDirectory() as folder:
            stream=RGBStream(folder,{'left_wrist':{'parent_frame':'left_wrist_yaw_link'}})
            image=np.zeros((240,320,3),np.uint8)
            stream.capture('left_wrist',image,0,0.,{'native_time_s':0.,'updated_at_s':0.,'frame':1})
            self.assertEqual(stream.latest['left_wrist']['time_s'],0.)
            with self.assertRaisesRegex(ValueError,'advance'):
                stream.capture('left_wrist',image,2,.04,{'native_time_s':.04,'updated_at_s':.04,'frame':1})
            self.assertNotIn('left_wrist',stream.latest)
            stream.capture('left_wrist',image,4,.08,{'native_time_s':.08,'updated_at_s':.08,'frame':5})
            frames=list(Path(folder).glob('*-left_wrist.png'))
            self.assertEqual(len(frames),2)
            stream.close()

    def test_stale_native_buffer_is_not_exported(self):
        from g1cap.camera_rgb import RGBStream
        with tempfile.TemporaryDirectory() as folder:
            stream=RGBStream(folder,{'head':{}})
            with self.assertRaisesRegex(ValueError,'stale'):
                stream.capture('head',np.zeros((20,20,3),np.uint8),2,.04,
                               {'native_time_s':.04,'updated_at_s':.02,'frame':1})
            self.assertFalse(stream.latest)
            stream.close()
