import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


class WorkspaceAgentTests(unittest.TestCase):
    def test_sensor_context_uses_onboard_image_and_sensor_contract(self):
        from test_visual_observation import snapshot_fixture
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);packet=snapshot_fixture(root)
            packet['observation']['observation_mode']='sensor_estimates_v1';packet['frames']=packet['frames'][:1]
            cli=self.fixture(root,'''assert sys.argv.count('--image')==1
prompt=sys.stdin.read()
assert 'head then overview' not in prompt
Path('policy.py').write_text('def run(robot, task):\\n robot.observe()\\n')
''')
            m.WorkspaceCodexAgent(cli=cli,vision_mode='direct').generate(
                dict(protocol='recorded_observation',backend='arena',api='sensor contract',
                    task={'observation_mode':'sensor_estimates_v1'},observation=packet['observation'],
                    visual_root=str(root),visual_observations=[packet]),root/'generation')
            resources=root/'generation/resources'
            self.assertTrue((resources/'tool_docs/arena_sensor.md').is_file())
            self.assertFalse((resources/'tool_docs/arena_box.md').exists())
            self.assertFalse((resources/'g1cap/arena_control.py').exists())
            self.assertTrue((resources/'g1cap/arena_public.py').is_file())

    def test_sensor_mode_cannot_reuse_a_privileged_workspace_or_snapshot(self):
        from test_visual_observation import snapshot_fixture
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);packet=snapshot_fixture(root)
            cli=self.fixture(root,'''sys.stdin.read()
Path('policy.py').write_text('def run(robot, task):\\n robot.observe()\\n')
''')
            legacy=dict(protocol='persistent_session',backend='arena',api='legacy',observation=packet['observation'])
            agent=m.WorkspaceCodexAgent(cli=cli)
            agent.generate(legacy,root/'generation-0')
            sensor={**legacy,'task':{'observation_mode':'sensor_estimates_v1'},
                    'observation':{**packet['observation'],'observation_mode':'sensor_estimates_v1'}}
            with self.assertRaisesRegex(ValueError,'observation mode'):
                agent.generate(sensor,root/'generation-1')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);packet=snapshot_fixture(root)
            sensor=dict(protocol='recorded_observation',backend='arena',api='sensor',
                task={'observation_mode':'sensor_estimates_v1'},observation=packet['observation'],
                visual_root=str(root),visual_observations=[packet])
            with self.assertRaisesRegex(ValueError,'observation mode'):
                m.WorkspaceCodexAgent(cli=cli,vision_mode='direct').generate(sensor,root/'generation')

    def test_luna_direct_vision_keeps_explicit_model_and_image_arguments(self):
        from test_visual_observation import snapshot_fixture
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);packet=snapshot_fixture(root)
            cli=self.fixture(root,'''assert sys.argv[sys.argv.index('--model')+1]=='gpt-5.6-luna'
assert 'model_reasoning_effort="low"' in sys.argv
assert sys.argv.count('--image')==2
sys.stdin.read()
Path('policy.py').write_text('def run(robot, task):\\n robot.observe()\\n')
''')
            m.WorkspaceCodexAgent(cli=cli,vision_mode='direct').generate(
                dict(protocol='recorded_observation',backend='arena',api='docs',observation=packet['observation'],
                     visual_root=str(root),visual_observations=[packet]),root/'generation')
            meta=json.loads((root/'generation/evidence/generation.json').read_text())
            self.assertEqual((meta['model'],meta['reasoning'],meta['image_count']),('gpt-5.6-luna','low',2))
            self.assertTrue(any(p.name=='arena_alignment.py' for p in (root/'generation/resources').rglob('*.py')))

    def test_structured_revision_cannot_reuse_a_previous_note(self):
        from test_visual_observation import snapshot_fixture
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);packet=snapshot_fixture(root)
            cli=self.fixture(root,'''sys.stdin.read()
Path('policy.py').write_text('def run(robot, task):\\n robot.observe()\\n')
if not Path('first-round').exists():
 Path('observation.md').write_text('First observation.')
 Path('first-round').touch()
''')
            agent=m.WorkspaceCodexAgent(cli=cli,model='gpt-6-astra',reasoning='medium',vision_mode='structured')
            request=dict(protocol='persistent_session',backend='arena',api='docs',observation=packet['observation'],
                         visual_root=str(root),visual_observations=[packet])
            agent.generate(request,root/'generation-0')
            with self.assertRaisesRegex(RuntimeError,'observation.md'):
                agent.generate(request,root/'generation-1')

    def test_recorded_case_does_not_claim_a_live_world(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            cli=self.fixture(root,'''prompt=sys.stdin.read()
assert 'No simulation is running' in prompt
assert 'simulation keeps running' not in prompt
Path('policy.py').write_text('def run(robot, task):\\n robot.observe()\\n')
''')
            m.WorkspaceCodexAgent(cli=cli).generate(
                dict(protocol='recorded_observation',backend='arena',api='docs'),root/'generation')
            self.assertTrue((root/'generation/resources/tool_docs/arena_box.md').is_file())

    def test_images_are_attached_from_read_only_resources_and_note_archived(self):
        from test_visual_observation import snapshot_fixture
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);packet=snapshot_fixture(root)
            cli=self.fixture(root,'''args=sys.argv
paths=[args[i+1] for i,x in enumerate(args) if x=='--image']
assert len(paths)==2 and all('/resources/observations/' in p for p in paths)
assert all(Path(p).read_bytes().startswith(b'\\x89PNG') for p in paths)
assert 'features.view_image=false' in args
assert args[-2:]==['--','-']
prompt=sys.stdin.read()
assert 'observation.md' in prompt
Path('policy.py').write_text('def run(robot, task):\\n robot.observe()\\n')
Path('observation.md').write_text('Visible red pixel; depth unknown.')
''')
            request=dict(protocol='persistent_session',backend='arena',api='contract',
                         observation=packet['observation'],visual_root=str(root),visual_observations=[packet])
            m.WorkspaceCodexAgent(cli=cli,model='gpt-6-astra',reasoning='medium',vision_mode='structured').generate(
                request,root/'generation')
            meta=json.loads((root/'generation/evidence/generation.json').read_text())
            self.assertEqual(meta['image_count'],2)
            self.assertEqual((root/'generation/evidence/observation.md').read_text(),'Visible red pixel; depth unknown.')
            task=json.loads((root/'generation/resources/task.json').read_text())
            self.assertNotIn('visual_root',task)
            self.assertIn('visual_observation.md',(root/'generation/resources/API.md').read_text())

    def test_invalid_image_input_prevents_model_call(self):
        from test_visual_observation import snapshot_fixture
        m=self.module()
        for mode in ('off','direct'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                root=Path(folder);packet=snapshot_fixture(root)
                (root/'0-head.png').write_bytes(b'changed')
                agent=m.WorkspaceCodexAgent(cli='/does/not/exist',model='gpt-6-astra',reasoning='medium',vision_mode=mode)
                with self.assertRaises((ValueError,RuntimeError)):
                    agent.generate(dict(protocol='persistent_session',backend='arena',api='docs',
                        observation=packet['observation'],visual_root=str(root),visual_observations=[packet]),root/'generation')
                meta=json.loads((root/'generation/evidence/generation.json').read_text())
                self.assertFalse(meta['cli_started'])

    def test_explicit_model_matches_command_and_recorded_settings(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            cli=self.fixture(root,'''assert sys.argv[sys.argv.index('--model')+1]=='gpt-6-astra'
assert 'model_reasoning_effort="medium"' in sys.argv
sys.stdin.read()
Path('policy.py').write_text('def run(robot, task):\\n robot.observe()\\n')
''')
            m.WorkspaceCodexAgent(cli=cli, timeout=300, model='gpt-6-astra', reasoning='medium').generate(
                {'api':'contract'}, root/'generation')
            meta=json.loads((root/'generation/evidence/generation.json').read_text())
            self.assertEqual((meta['model'],meta['reasoning'],meta['wall_timeout_s']),
                             ('gpt-6-astra','medium',300))

    def test_unsupported_settings_fail_before_cli_launch(self):
        m=self.module()
        for settings in ({'model':'unknown'}, {'model':'gpt-6-astra','reasoning':'low'},
                         {'vision_mode':'stream'}, {'timeout':301}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                m.WorkspaceCodexAgent(cli='/does/not/exist',**settings)

    def test_arena_context_excludes_unavailable_sonic_tools(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            cli=self.fixture(root,'sys.stdin.read()\nPath("policy.py").write_text("def run(robot, task):\\n robot.observe()\\n")')
            m.WorkspaceCodexAgent(cli=cli).generate(
                {'protocol':'persistent_session','backend':'arena','api':'box tool contract'},root/'generation-0')
            resources=root/'generation-0/resources'
            self.assertTrue((resources/'tool_docs/arena_box.md').is_file())
            self.assertFalse((resources/'tool_docs/sonic_motion.md').exists())
            self.assertTrue((resources/'g1cap/arena_control.py').is_file())
            self.assertTrue((resources/'g1cap/arena_lift.py').is_file())
            self.assertTrue((resources/'g1cap/toolkit/hand_clearance.py').is_file())
            self.assertTrue((resources/'g1cap/toolkit/acquisition_wrists.py').is_file())
            self.assertFalse((resources/'g1cap/session_tools.py').exists())
            self.assertIn('arena_box.md',(resources/'API.md').read_text())

    def test_persistent_rounds_keep_workspace_and_use_continuation_prompt(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            cli=self.fixture(root,'''prompt=sys.stdin.read()
assert 'same physical episode' in prompt
assert 'single-turn' not in prompt
counter=Path('counter.txt')
n=int(counter.read_text())+1 if counter.exists() else 1
counter.write_text(str(n))
Path('policy.py').write_text('def run(robot, task):\\n robot.observe()\\n')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':3,'output_tokens':2}}))
''')
            agent=m.WorkspaceCodexAgent(cli=cli)
            for index in range(2):
                agent.generate({'protocol':'persistent_session','api':'world frame'},root/f'generation-{index}')
            self.assertEqual((root/'workspace/counter.txt').read_text(),'2')
            for index in range(2):
                resources=root/f'generation-{index}/resources'
                self.assertIn('duration_elapsed',(resources/'tool_docs/sonic_motion.md').read_text())
                self.assertIn('tool_docs/README.md',(resources/'API.md').read_text())
                self.assertTrue((resources/'g1cap/session_tools.py').exists())
                self.assertFalse((resources/'g1cap/toolkit/robot.py').exists())
                self.assertFalse((resources/'g1cap/toolkit/api.py').exists())
                hashes=json.loads((root/f'generation-{index}/evidence/public-source-sha256.json').read_text())
                self.assertEqual(hashes['tool_docs/sonic_motion.md'],m.digest((resources/'tool_docs/sonic_motion.md').read_bytes()))

    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('g1cap.workspace_agent'))
        from g1cap import workspace_agent
        return workspace_agent

    def fixture(self, root, body):
        cli = root/'codex-fixture'
        cli.write_text('#!'+sys.executable+'\nimport sys,json\nfrom pathlib import Path\n'
                       'if sys.argv[1:]==["login","status"]:\n print("Logged in using ChatGPT")\n sys.exit(0)\n'
                       'if "sandbox" in sys.argv:\n'
                       # Native enforcement is verified separately using the actual installed CLI.
                       ' print(json.dumps({"ok": True}))\n sys.exit(0)\n'+body)
        cli.chmod(0o700)
        return str(cli)

    def test_workspace_generation_reads_file_and_saves_settings(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            cli = self.fixture(root, '''args=sys.argv
assert '--sandbox' not in args
assert 'default_permissions="g1-trial"' in args
assert 'approval_policy="never"' in args
assert 'features.shell_tool=true' in args
assert args[args.index('--model')+1]=='gpt-5.6-luna'
assert 'model_reasoning_effort="low"' in args
prompt=sys.stdin.read()
assert 'policy.py' in prompt
Path('notes.txt').write_text('prepared using shell')
Path('policy.py').write_text('def run(robot, task):\\n robot.hold(1)\\n')
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':12,'output_tokens':8}}))
''')
            trial = root/'trial'
            (trial/'evidence').mkdir(parents=True)
            agent = m.WorkspaceCodexAgent(cli=cli)
            source = agent.generate({'task':{}, 'api':'robot.observe()', 'backend':'mock'}, trial)
            self.assertIn('robot.hold(1)', source)
            self.assertTrue((trial/'workspace/notes.txt').exists())
            meta = json.loads((trial/'evidence/generation.json').read_text())
            self.assertEqual(meta['usage']['input_tokens'],12)
            self.assertEqual(meta['status'],'completed')
            self.assertFalse(meta['command_network_access'])
            self.assertTrue((trial/'resources/API.md').exists())

    def test_missing_or_symlink_submission_fails_without_source_fallback(self):
        m = self.module()
        for body in ['pass', 'Path("policy.py").symlink_to("../resources/API.md")']:
            with self.subTest(body=body), tempfile.TemporaryDirectory() as root:
                root=Path(root)
                cli=self.fixture(root,body)
                trial=root/'trial'
                (trial/'evidence').mkdir(parents=True)
                with self.assertRaisesRegex(RuntimeError,'regular policy.py'):
                    m.WorkspaceCodexAgent(cli=cli).generate({'api':'docs'},trial)
                meta=json.loads((trial/'evidence/generation.json').read_text())
                self.assertEqual(meta['status'],'invalid_submission')

    def test_native_profile_does_not_grant_trial_evidence_or_home(self):
        m = self.module()
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            config=m.permission_config(root/'workspace',root/'resources')
            self.assertEqual(config['default_permissions'],'g1-trial')
            fs=config['permissions.g1-trial.filesystem']
            self.assertEqual(fs[':root'],'deny')
            self.assertEqual(fs[str(root/'workspace')],'write')
            self.assertEqual(fs[str(root/'resources')],'read')
            self.assertNotIn(str(root),fs)
            self.assertFalse(config['permissions.g1-trial.network.enabled'])

    def test_native_boundary_failure_prevents_model_launch(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            cli=root/'codex-fixture'
            cli.write_text('#!'+sys.executable+'\nimport sys\nassert "sandbox" in sys.argv\nsys.exit(71)\n')
            cli.chmod(0o700)
            trial=root/'trial'
            (trial/'evidence').mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError,'boundary check failed'):
                m.WorkspaceCodexAgent(cli=str(cli)).generate({'api':'docs'},trial)
            meta=json.loads((trial/'evidence/generation.json').read_text())
            self.assertFalse(meta['cli_started'])
            self.assertEqual(meta['status'],'setup_error')

    def test_timeout_preserves_evidence_and_never_accepts_partial_policy(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            cli=self.fixture(root,'import time\nPath("policy.py").write_text("partial")\ntime.sleep(10)')
            trial=root/'trial'
            (trial/'evidence').mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError,'timeout'):
                m.WorkspaceCodexAgent(cli=cli,timeout=.1).generate({'api':'docs'},trial)
            meta=json.loads((trial/'evidence/generation.json').read_text())
            self.assertEqual(meta['status'],'timeout')
