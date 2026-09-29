"""Fresh acquisition flushes pending policy actions without resetting physics."""
import unittest
from g1cap.arena_control import BoxControl


class AcquisitionRestartTests(unittest.TestCase):
    def test_world_reset_logs_policy_only_and_rejects_remote_failure(self):
        # Run the actual dependency-free owner method without importing Isaac.
        import ast
        import io
        import json
        from pathlib import Path
        import time
        from types import SimpleNamespace
        tree=ast.parse((Path(__file__).parents[1]/'g1cap/arena_world.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ArenaWorld')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='begin_acquisition')
        namespace=dict(json=json,time=time)
        exec(compile(ast.Module(body=[method],type_ignores=[]),'arena_world.begin_acquisition','exec'),namespace)
        calls=[];log=io.StringIO()
        owner=SimpleNamespace(step_index=123,policy=SimpleNamespace(reset=lambda:calls.append('policy')),
                              files={'gr00t-resets':log},box_perception=None)
        namespace['begin_acquisition'](owner,2.46)
        self.assertEqual(calls,['policy'])
        record=json.loads(log.getvalue())
        self.assertEqual((record['step'],record['time_s'],record['status']),(123,2.46,'reset'))
        def failure():raise RuntimeError('remote unavailable')
        owner.policy.reset=failure
        with self.assertRaisesRegex(ValueError,'acquisition_policy_reset_failed'):
            namespace['begin_acquisition'](owner,2.48)
        self.assertEqual(json.loads(log.getvalue().splitlines()[-1])['status'],'failed')

    def test_source_commits_only_after_successful_policy_reset_and_survives_commands(self):
        import ast,io,json,time
        from pathlib import Path
        from types import SimpleNamespace
        from test_carry_perception import CarryPerceptionTests
        tree=ast.parse((Path(__file__).parents[1]/'g1cap/arena_world.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ArenaWorld')
        methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and
                 n.name in ('begin_acquisition','scene_hold_command')]
        namespace=dict(json=json,time=time,__package__='g1cap')
        exec(compile(ast.Module(body=methods,type_ignores=[]),'arena_world.admission','exec'),namespace)
        frontend=CarryPerceptionTests().frontend();frontend.begin_carry(1.)
        original=frontend.source_plane;calls=[]
        def reset():
            self.assertIs(frontend.source_plane,original)
            calls.append('reset')
        owner=SimpleNamespace(step_index=50,policy=SimpleNamespace(reset=reset),
            files={'gr00t-resets':io.StringIO()},box_perception=frontend,
            control=self.control)
        namespace['begin_acquisition'](owner,1.)
        new=frontend.source_plane
        self.assertIsNot(new,original)
        self.assertTrue(new.initialized)
        action=[0.]*50
        for t in (1.,1.02,1.04):
            self.assertEqual(namespace['scene_hold_command'](owner,'acquire',t,action),action)
            self.assertIs(frontend.source_plane,new)
        # Invalid geometry must reject before resetting policy.
        with self.assertRaises(ValueError):namespace['begin_acquisition'](owner,1.2)
        self.assertEqual(calls,['reset']);self.assertIs(frontend.source_plane,new)
        def fail():raise RuntimeError('reset failed')
        owner.policy.reset=fail
        with self.assertRaisesRegex(ValueError,'acquisition_policy_reset_failed'):
            namespace['begin_acquisition'](owner,1.04)
        self.assertIs(frontend.source_plane,new)

    def setUp(self):
        self.queue=[];self.resets=[];self.sensor_value=1.;self.requests=[]
        self.action=[.2]*43+[0.,0.,0.,.75,0.,0.,0.]
        def begin(now):self.queue.clear();self.resets.append(now)
        def acquire(obs):
            if not self.queue:self.queue.extend([self.sensor_value]*3)
            value=self.queue.pop(0);self.requests.append(value)
            result=list(self.action);result[0]=value;return result
        self.control=BoxControl(self.action,acquire,None,acquisition_begin=begin,
            visual_grasp=lambda t:dict(status='available',time_s=t,ready=False,raised=False,attitude_ok=True),
            scene_hold=lambda phase,t,a:a,approach_feedback=lambda t:dict(status='available'),
            hand_clearance_feedback=lambda t:dict(status='available'),sensor_fault=lambda t:None)

    def test_retry_fetches_current_input_instead_of_resuming_old_chunk(self):
        c=self.control
        c.update({'time':0.})
        self.assertEqual(c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})['status'],'running')
        c.command({'time':0.});self.assertEqual(self.requests,[1.])
        self.assertEqual(len(self.queue),2)
        c.cancel('prior_program_cancelled',{'time':0.})
        prior_result=dict(c.result)
        # Robot time and sensors advance within the same controller/episode.
        self.sensor_value=2.;c.update({'time':.02})
        self.assertEqual(c.history[-1]['time'],.02)
        self.assertEqual(c.start('pickup_box',{'object_id':'brown_box'},{'time':.02})['status'],'running')
        c.command({'time':.02})
        self.assertEqual(self.requests,[1.,2.])
        self.assertEqual(self.resets,[0.,.02])
        self.assertEqual(c.phase_started,.02)
        self.assertEqual(prior_result['status'],'cancelled')
        self.assertEqual(c.last_observation_time,.02)

    def test_rejected_or_duplicate_request_does_not_reset_policy(self):
        c=self.control
        self.assertEqual(c.start('pickup_box',{'object_id':'wrong'},{'time':0.})['status'],'rejected')
        self.assertFalse(self.resets)
        c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})
        self.assertEqual(c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})['reason'],'operation_active')
        self.assertEqual(self.resets,[0.])

    def test_reset_failure_prevents_acquisition_without_erasing_episode(self):
        c=self.control;c.update({'time':0.})
        def unavailable(now):raise ValueError('acquisition_policy_reset_failed')
        c.acquisition_begin=unavailable
        result=c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})
        self.assertEqual(result['status'],'rejected')
        self.assertEqual(result['reason'],'acquisition_policy_reset_failed')
        self.assertIsNone(c.method);self.assertFalse(self.requests)
        self.assertEqual(c.last_observation_time,0.)
        self.assertEqual(c.last_action[43:46],[0.,0.,0.])

    def test_explicit_retry_uses_fresh_pickup_geometry_without_bypassing_retained_grasp(self):
        c=self.control;c.update({'time':0.})
        c.visual_grasp=lambda t:dict(status='unavailable')
        c.acquisition_grasp=lambda t:dict(status='available',opposing_near_wrists=False)
        self.assertEqual(c.start('pickup_box',{'object_id':'brown_box'},{'time':0.})['status'],'running')
        c.cancel('retry_test',{'time':0.})
        c.sensor_lift_ready=lambda t:None
        c.acquisition_grasp=lambda t:dict(status='available',opposing_near_wrists=True)
        self.assertEqual(c.start('pickup_box',{'object_id':'brown_box'},{'time':.02})['reason'],
                         'retained_grasp_requires_raise_or_hold')
        self.assertEqual(self.resets,[0.])
        self.assertEqual(c.start('hold_box',{'duration':1.},{'time':.02})['reason'],
                         'visual_state_unavailable')
