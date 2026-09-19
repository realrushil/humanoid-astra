"""Check the information passed to the coding agent, separate from the scorer."""
import json
from pathlib import Path
import tempfile
import unittest

from g1cap import interactive


class InteractiveTests(unittest.TestCase):
    def test_snapshot_service_publishes_aligned_images_once_without_stepping(self):
        from g1cap import arena_session_runtime as runtime
        from test_visual_observation import snapshot_fixture
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);inbox=root/'session/visual-inbox';inbox.mkdir(parents=True)
            packet=snapshot_fixture(root,10);calls=[]
            class World:
                raw={'step':10,'physics_step':40,'time':.2}
                camera_captured_at_unix_s=100.
                def export_camera_frames(self,directory):
                    calls.append('export')
                    return snapshot_fixture(directory,10)['frames']
            class Session:
                session_id='session-a'
                def observe(self):return packet['observation']
            request=inbox/('a'*32+'.json')
            request.write_text(json.dumps({'snapshot_id':'a'*32,'session_id':'session-a'}))
            runtime.service_visual_snapshot(World(),Session(),root)
            saved=json.loads((root/'session/visual-snapshots'/('a'*32)/'manifest.json').read_text())
            self.assertEqual(saved['observation']['step'],10)
            self.assertEqual(saved['physics_step'],40)
            self.assertEqual(len(saved['frames']),2)
            request.write_text(json.dumps({'snapshot_id':'a'*32,'session_id':'session-a'}))
            runtime.service_visual_snapshot(World(),Session(),root)
            self.assertEqual(calls,['export'])
            request=inbox/('b'*32+'.json')
            request.write_text(json.dumps({'snapshot_id':'b'*32,'session_id':'different'}))
            runtime.service_visual_snapshot(World(),Session(),root)
            self.assertTrue((root/'session/visual-snapshots'/('b'*32)/'error.json').is_file())
            self.assertEqual(calls,['export'])

    def test_visual_rounds_use_captured_state_and_retain_only_previous_pair(self):
        from test_visual_observation import snapshot_fixture
        class Remote:
            max_rounds=2
            def __init__(self,root):self.output=root;self.step=0
            def start(self):return self.status()
            def status(self):
                return dict(session_id='session-a',task={'backend':'arena'},terminal_reason=None,
                            observation={'session_id':'session-a','step':self.step,'physics_step':self.step*4,'time':self.step*.02},metrics={})
            def visual_snapshot(self,directory):
                directory.mkdir(parents=True,exist_ok=True)
                self.step+=1
                return snapshot_fixture(directory,self.step)
            def execute(self,source):
                self.step+=10
                return dict(index=0,execution={'status':'completed'},tools=[])
            def close(self):pass
        class Agent:
            vision_mode='direct'
            def __init__(self):self.requests=[]
            def generate(self,request,directory):self.requests.append(request);return 'def run(robot, task):\n robot.observe()\n'
        with tempfile.TemporaryDirectory() as folder:
            agent=Agent();remote=Remote(Path(folder))
            report=interactive.run_interactive(remote,folder,agent=agent)
            self.assertIsNone(report['error'])
            self.assertEqual(agent.requests[0]['observation']['step'],1)
            self.assertEqual([p['step'] for p in agent.requests[1]['visual_observations']],[1,12])
            self.assertEqual(agent.requests[1]['observation']['step'],12)

    def test_feedback_retains_execution_without_evaluator_metrics(self):
        status={'task':{'backend':'arena'},'observation':{'session_id':'s','time':2.},
                'metrics':{'private_evaluator_sentinel':True}}
        result={'index':0,'execution':{'status':'completed','stdout':'observed'},
                'tools':[{'method':'hold_box','result':{'status':'completed'}}],
                'start_observation':{'time':1.},'end_observation':{'time':2.}}
        feedback=interactive.agent_feedback(result,status)
        request=interactive.generation_request(status,'previous source',[feedback])
        self.assertNotIn('private_evaluator_sentinel',json.dumps(request))
        self.assertEqual(request['feedback'][0]['tools'],result['tools'])
        self.assertEqual(request['previous_source'],'previous source')
        self.assertEqual(feedback['start_observation']['time'],1.)

    def test_two_rounds_preserve_prior_code_and_report_score_separately(self):
        class Remote:
            max_rounds=2
            def __init__(self,root):self.output=root;self.step=0;self.closed=False
            def start(self):return self.status()
            def status(self):
                return dict(session_id='same',task={'backend':'arena'},terminal_reason=None,
                            observation={'session_id':'same','step':self.step,'time':self.step*.02},
                            metrics={'private_evaluator_sentinel':self.step})
            def execute(self,source):
                self.step+=10
                return dict(index=self.step//10-1,execution={'status':'completed'},tools=[])
            def close(self):self.closed=True
        class Agent:
            def __init__(self):self.requests=[]
            def generate(self,request,directory):
                self.requests.append(request)
                return 'def run(robot, task):\n robot.observe()\n'
        with tempfile.TemporaryDirectory() as folder:
            remote=Remote(Path(folder));agent=Agent()
            report=interactive.run_interactive(remote,folder,agent=agent)
            self.assertIsNone(report['error'])
            self.assertEqual(agent.requests[1]['observation']['step'],10)
            self.assertEqual(agent.requests[1]['previous_source'],'def run(robot, task):\n robot.observe()\n')
            self.assertNotIn('private_evaluator_sentinel',json.dumps(agent.requests))
            self.assertEqual(report['final_status']['metrics']['private_evaluator_sentinel'],20)
            self.assertTrue(remote.closed)
