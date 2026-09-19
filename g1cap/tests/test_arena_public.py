"""Counterfactual checks across the worker-visible serialization boundary."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from g1cap.arena_session import ArenaSession
from g1cap.interactive import generation_request,agent_feedback
from test_arena_session import FakeControl


class ArenaPublicTests(unittest.TestCase):
    @patch('g1cap.arena_session.time.monotonic',return_value=100.)
    def run_counterfactual(self,secret,clock):
        with tempfile.TemporaryDirectory() as folder:
            control=FakeControl()
            task=dict(backend='arena',object_id='brown_box',task_kind='table_transfer',surface_id='destination',
                instruction='Move the brown box to the green table.',deadline_s=30.,
                box_mass_kg=secret,source_bounds={'private':secret},fixture=str(secret),surfaces={'private':secret})
            packet=dict(observation_mode='sensor_estimates_v1',time=1.,step=50,physics_step=200,
                box={'status':'accepted','dimensions_m':[.2,.21,.19]},grasp={'ready':False})
            session=ArenaSession(control,task,folder,sensor_observation=lambda now:deepcopy(packet))
            try:
                session.session_id='same-episode'
                session.tick(dict(time=1.,step=50,physics_step=200,box_mass_kg=secret,box_pos=[secret]*3,
                    hand_forces_N={'left':secret},hold_observation_v2={'ready':bool(secret)},private=secret))
                ticket=dict(round_id='same-episode:0',method='hold_box',done=__import__('threading').Event())
                session._resolve(ticket,dict(status='failed',reason=f'contact_force_{secret}',
                    observation={'box_mass_kg':secret},measurements={'private':secret}))
                with patch('g1cap.arena_session.time.monotonic',return_value=session.received_at):
                    status=session.status();status['metrics']={'private':secret}
                    result=dict(index=0,execution={'status':'completed'},tools=session.tool_results,
                        start_observation=session.observe(),end_observation=session.observe())
                    request=generation_request(status,'same program',[agent_feedback(result,status)])
                self.assertEqual(request['task']['control_track'],'mixed_controller_sensor_publication')
                self.assertNotIn('source_bounds',request['task'])
                self.assertNotIn('box_mass_kg',json.dumps(request))
                self.assertEqual(request['feedback'][0]['tools'][0]['result']['reason'],'controller_stopped')
                self.assertEqual(ticket['result']['observation']['box']['dimensions_m'],[.2,.21,.19])
                return request
            finally:session.close()

    def test_hidden_truth_cannot_change_task_observations_or_nested_feedback(self):
        self.assertEqual(self.run_counterfactual(987.123),self.run_counterfactual(654.789))

    def test_sensor_changes_remain_visible_and_returned_values_are_copies(self):
        with tempfile.TemporaryDirectory() as folder:
            packet={'time':0.,'step':0,'physics_step':0,'box':{'dimensions_m':[.2,.2,.2]}}
            session=ArenaSession(FakeControl(),{},folder,sensor_observation=lambda t:deepcopy(packet))
            try:
                session.tick({'time':0.});observed=session.observe();observed['box']['dimensions_m'][0]=10
                self.assertEqual(session.observe()['box']['dimensions_m'][0],.2)
                packet['box']['dimensions_m'][0]=.3;session.tick({'time':0.})
                self.assertEqual(session.observe()['box']['dimensions_m'][0],.3)
                session.finish('forbidden_source_contact')
                self.assertEqual(session.status()['terminal_reason'],'episode_terminated')
                self.assertEqual(session.observe()['episode_status'],'episode_terminated')
            finally:session.close()

    def test_public_estimates_strip_extra_fields_and_expire_old_scene_pose(self):
        from g1cap.arena_public import sensor_observation
        from g1cap.arena_sensors import proprioception_packet
        packet=proprioception_packet(step=20,time_s=.4,joint_names=['joint'],q=[0.],dq=[0.],tau_est=[1.],gyro=[0.]*3,accel=[0.,0.,9.81])
        box=dict(status='accepted',center_camera_m=[0.,0.,.4],axes_camera=[[1,0,0],[0,1,0],[0,0,1]],
            dimensions_m=[.2]*3,observed_at_s=.4,age_s=0.,box_mass_kg=100.)
        motion=dict(status='tracked',time_s=.1,segment=1,body_in_segment=[[1]*4]*4,private=999)
        result=sensor_observation(packet,box,{'status':'available','ready':False,'private':999},motion)
        self.assertNotIn('box_mass_kg',json.dumps(result));self.assertNotIn('private',json.dumps(result))
        self.assertIsNone(result['scene_motion']['body_in_segment'])
        self.assertEqual(result['scene_motion']['status'],'unavailable')
        self.assertIsNone(result['object_load']['mass_kg'])

    def test_sensor_snapshot_accepts_only_onboard_camera(self):
        from test_visual_observation import snapshot_fixture
        from g1cap.visual_observation import validate_snapshot
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);packet=snapshot_fixture(root)
            packet['observation']['observation_mode']='sensor_estimates_v1'
            with self.assertRaises(ValueError):validate_snapshot(packet,root,'session-a')
            packet['frames']=packet['frames'][:1]
            validate_snapshot(packet,root,'session-a')

    def test_loaded_stop_publication_is_current_and_strips_private_fields(self):
        from g1cap.arena_public import sensor_observation
        packet=dict(time_s=1.,step=50)
        stop=dict(status='active',time_s=1.,clearance_lower_m=.04,required_margin_m=.03,box_mass_kg=999)
        value=sensor_observation(packet,{'status':'unavailable'},{'status':'unavailable'},None,loaded_stop=stop)
        self.assertEqual(value['loaded_stop']['status'],'active')
        self.assertNotIn('box_mass_kg',value['loaded_stop'])
        stop.update(status='failed',reason='private_contact_999')
        value=sensor_observation(packet,{'status':'unavailable'},{'status':'unavailable'},None,loaded_stop=stop)
        self.assertEqual(value['loaded_stop']['reason'],'controller_stopped')
        stop['time_s']=.9
        value=sensor_observation(packet,{'status':'unavailable'},{'status':'unavailable'},None,loaded_stop=stop)
        self.assertEqual(value['loaded_stop'],dict(status='unavailable',reason='loaded_stop_status_stale'))
