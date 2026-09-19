"""Subscription-authenticated Codex CLI adapter; no API key extraction or fallback."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

from .runner import write_json


def find_codex():
    candidates = [shutil.which('codex'), '/Applications/ChatGPT.app/Contents/Resources/codex',
                  '/Applications/Codex.app/Contents/Resources/codex']
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError('Codex CLI not found. Install Codex and sign in with ChatGPT.')


class CodexAgent:
    metadata = {'kind': 'codex_cli', 'model': 'gpt-5.6-luna', 'reasoning': 'low',
                'authentication': 'saved_chatgpt_session', 'speed': 'standard', 'fallback': False}

    def __init__(self, cli=None, timeout=180):
        self.cli = cli or find_codex()
        self.timeout = timeout

    def generate(self, request, directory):
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        schema = {'type': 'object', 'properties': {'source': {'type': 'string'}},
                  'required': ['source'], 'additionalProperties': False}
        schema_path = directory/'response-schema.json'
        response_path = directory/'response.json'
        if response_path.exists():
            raise FileExistsError('refusing to reuse a generation response file')
        write_json(schema_path, schema)
        prompt = ('You are generating a short robot task policy for a pipeline smoke test. '
                  'Return only the requested JSON containing Python source. Do not use tools, '
                  'read local files, browse, install software, or delegate. '
                  'If feedback is supplied, revise the previous source based on that evidence.\n\n'
                  + json.dumps(request, allow_nan=False))
        (directory/'prompt.txt').write_text(prompt)
        metadata = {**self.metadata, 'status': 'running', 'usage': None}
        start = time.monotonic()
        # Preserve saved CLI authentication, but never pass API-key overrides.
        environment = {k:v for k,v in os.environ.items()
                       if k not in {'OPENAI_API_KEY', 'CODEX_API_KEY', 'OPENAI_BASE_URL'}}
        auth = subprocess.run([self.cli, 'login', 'status'], capture_output=True, text=True,
                              env=environment, timeout=15)
        if auth.returncode != 0 or 'Logged in using ChatGPT' not in auth.stdout + auth.stderr:
            metadata.update(status='authentication_error', returncode=auth.returncode)
            write_json(directory/'generation.json', metadata)
            raise RuntimeError('Codex must be signed in with ChatGPT for this subscription-only smoke run')
        metadata['authentication_verified'] = True
        process = None
        with tempfile.TemporaryDirectory(prefix='g1cap-agent-') as working_dir:
            command = [self.cli, 'exec', '--ignore-user-config', '--skip-git-repo-check',
                       '--ephemeral', '--sandbox', 'read-only', '--model', 'gpt-5.6-luna',
                       '-c', 'model_reasoning_effort="low"', '-c', 'web_search="disabled"',
                       '-c', 'features.shell_tool=false', '-c', 'project_doc_max_bytes=0',
                       '--json', '--output-schema', str(schema_path),
                       '--output-last-message', str(response_path), '-']
            with (directory/'codex-events.jsonl').open('wb') as events, (directory/'codex-stderr.txt').open('wb') as errors:
                try:
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=events, stderr=errors,
                                               cwd=working_dir, env=environment, start_new_session=True)
                    process.communicate(prompt.encode(), timeout=self.timeout)
                    metadata['status'] = 'completed' if process.returncode == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    metadata['status'] = 'timeout'
                except OSError as error:
                    metadata.update(status='failed', error=str(error))
                finally:
                    if process is not None:
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                        metadata['returncode'] = process.returncode
                    metadata['wall_time'] = round(time.monotonic() - start, 3)
        event_path = directory/'codex-events.jsonl'
        if event_path.stat().st_size <= 8_000_000:
            for line in event_path.read_text(errors='replace').splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('type') == 'turn.completed':
                    metadata['usage'] = event.get('usage')
        write_json(directory/'generation.json', metadata)
        if metadata['status'] != 'completed':
            raise RuntimeError(f'Codex generation {metadata["status"]}; see generation.json and codex-stderr.txt')
        if not response_path.exists() or response_path.stat().st_size > 100_000:
            raise RuntimeError('Codex did not return a bounded structured response')
        value = json.loads(response_path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get('source'), str) or len(value['source'].encode()) > 65536:
            raise RuntimeError('Codex response must contain Python source of at most 65536 bytes')
        return value['source']
