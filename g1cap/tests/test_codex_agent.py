import json
from pathlib import Path
import sys
import tempfile
import unittest
from g1cap.codex_agent import CodexAgent


class CodexAdapterTests(unittest.TestCase):
    def cli_fixture(self, directory, body):
        path=directory/'codex-fixture'
        path.write_text('#!'+sys.executable+'\nimport sys\nif sys.argv[1:]==["login","status"]:\n print("Logged in using ChatGPT")\n sys.exit(0)\n'+body)
        path.chmod(0o700)
        return str(path)

    def test_cli_receives_low_cost_model_and_returns_structured_source(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root)
            cli=self.cli_fixture(d, '''import json,sys
from pathlib import Path
args=sys.argv
assert args[args.index('--model')+1]=='gpt-5.6-luna'
assert 'model_reasoning_effort="low"' in args
assert '--ignore-user-config' in args
assert args[args.index('--sandbox')+1]=='read-only'
request=sys.stdin.read()
assert 'previous_source' in request
Path(args[args.index('--output-last-message')+1]).write_text(json.dumps({'source':'def run(robot, task):\\n robot.hold(1)'}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':12,'output_tokens':8}}))
''')
            agent=CodexAgent(cli=cli)
            source=agent.generate({'previous_source':None}, d/'generation')
            self.assertIn('robot.hold(1)',source)
            meta=json.loads((d/'generation/generation.json').read_text())
            self.assertEqual(meta['usage'],{'input_tokens':12,'output_tokens':8})
            self.assertEqual(meta['model'],'gpt-5.6-luna')

    def test_failed_generation_preserves_evidence_without_model_fallback(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root)
            cli=self.cli_fixture(d,'import sys\nprint("unavailable",file=sys.stderr)\nsys.exit(7)\n')
            with self.assertRaises(RuntimeError):
                CodexAgent(cli=cli).generate({},d/'generation')
            meta=json.loads((d/'generation/generation.json').read_text())
            self.assertEqual(meta['returncode'],7)
            self.assertEqual(meta['model'],'gpt-5.6-luna')
            self.assertIn('unavailable',(d/'generation/codex-stderr.txt').read_text())

    def test_cli_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root)
            cli=self.cli_fixture(d,'import time\ntime.sleep(10)\n')
            with self.assertRaisesRegex(RuntimeError,'timeout'):
                CodexAgent(cli=cli,timeout=.15).generate({},d/'generation')
            self.assertEqual(json.loads((d/'generation/generation.json').read_text())['status'],'timeout')

    def test_non_subscription_auth_is_rejected_before_generation(self):
        with tempfile.TemporaryDirectory() as root:
            d=Path(root)
            cli=d/'api-auth-fixture'
            cli.write_text('#!'+sys.executable+'\nprint("Logged in using an API key")\n')
            cli.chmod(0o700)
            with self.assertRaisesRegex(RuntimeError,'ChatGPT'):
                CodexAgent(cli=str(cli)).generate({},d/'generation')
