"""Shell-enabled Codex preparation in a native sandbox; the parent submits the policy."""
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

from .codex_agent import CodexAgent, find_codex
from .runner import digest, write_json
from .visual_observation import stage_visual_observations


def permission_config(workspace, resources, read_roots=()):
    """No --sandbox flag: it would override these finer-grained native permissions."""
    filesystem = {':root': 'deny', ':minimal': 'read',
                  str(Path(sys.base_prefix).resolve()): 'read',
                  str(workspace): 'write', str(resources): 'read'}
    filesystem.update({str(path): 'read' for path in read_roots})
    if Path('/opt/homebrew').is_dir():
        # Homebrew executables use symlinks and shared libraries outside Python's prefix.
        filesystem['/opt/homebrew'] = 'read'
    return {'default_permissions': 'g1-trial',
            'permissions.g1-trial.filesystem': filesystem,
            'permissions.g1-trial.network.enabled': False,
            'approval_policy': 'never', 'allow_login_shell': False,
            'shell_environment_policy.inherit': 'none',
            'shell_environment_policy.set': {'PATH': '/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin',
                'TMPDIR': str(workspace/'.tmp'), 'XDG_CACHE_HOME': str(workspace/'.cache'),
                'PYTHONDONTWRITEBYTECODE': '1'},
            'model_reasoning_effort': 'low', 'web_search': 'disabled', 'project_doc_max_bytes': 0,
            'features.shell_tool': True, 'features.shell_snapshot': False,
            'features.multi_agent': False, 'features.multi_agent_v2': False,
            'features.plugins': False, 'features.hooks': False,
            'features.skip_host_skill_discovery': True, 'features.skill_search': False,
            'features.enable_mcp_apps': False, 'features.view_image': False,
            'features.image_generation': False}


def config_arguments(config):
    def toml(value):
        if isinstance(value, dict):
            return '{'+', '.join(json.dumps(k)+'='+toml(v) for k, v in value.items())+'}'
        return json.dumps(value)
    return [part for key, value in config.items() for part in ('-c', key+'='+toml(value))]


def verify_boundary(cli, config, workspace, resources, evidence, environment):
    """Harmless sentinels exercise the same native policy before any model request."""
    hidden = evidence/'boundary-sentinel.txt'
    visible = resources/'boundary-sentinel.txt'
    hidden.write_text('private sentinel')
    visible.write_text('read-only sentinel')
    probe = resources/'boundary_check.py'
    probe.write_text('''import json, socket, sys
from pathlib import Path
workspace, visible, hidden = map(Path, sys.argv[1:])
assert visible.read_text() == 'read-only sentinel'
(workspace/'boundary-write.txt').write_text('allowed')
checks = []
for name, operation in [
    ('resource_write', lambda: visible.write_text('changed')),
    ('evidence_read', lambda: hidden.read_text()),
    ('evidence_write', lambda: hidden.write_text('changed')),
    ('outside_write', lambda: (hidden.parent.parent/'outside.txt').write_text('changed')),
]:
    try:
        operation()
    except PermissionError:
        checks.append(name)
    else:
        raise RuntimeError(name+' was not denied')
try:
    with socket.socket() as sock:
        sock.settimeout(1)
        sock.connect(('127.0.0.1', 9))
except PermissionError:
    checks.append('network')
else:
    raise RuntimeError('network was not denied')
print(json.dumps({'ok': True, 'denied': checks}))
''')
    command = [cli, 'sandbox', '-P', 'g1-trial', *config_arguments(config),
               '-C', str(workspace), '--', str(Path(sys.executable).resolve()), str(probe),
               str(workspace), str(visible), str(hidden)]
    result = subprocess.run(command, cwd=workspace, env=environment, capture_output=True,
                            text=True, timeout=20)
    value = {'returncode': result.returncode, 'stdout': result.stdout[-4000:],
             'stderr': result.stderr[-4000:]}
    write_json(evidence/'boundary-check.json', value)
    if result.returncode != 0:
        raise RuntimeError('native workspace boundary check failed; see boundary-check.json')
    try:
        if json.loads(result.stdout).get('ok') is not True:
            raise ValueError('missing success')
    except (ValueError, AttributeError) as error:
        raise RuntimeError('invalid native workspace boundary result') from error
    if hidden.read_text() != 'private sentinel' or visible.read_text() != 'read-only sentinel':
        raise RuntimeError('boundary sentinel was modified')


class WorkspaceCodexAgent:
    metadata = {**CodexAgent.metadata, 'kind': 'codex_workspace',
                'command_network_access': False, 'vision': False}

    def __init__(self, cli=None, timeout=180, *, model='gpt-5.6-luna', reasoning='low', vision_mode='off'):
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError('generation timeout must be in (0,300] seconds')
        if (model, reasoning) not in (('gpt-5.6-luna','low'), ('gpt-6-astra','medium')):
            raise ValueError('supported experiment settings are Luna/low or Astra/medium')
        if vision_mode not in ('off','direct','structured'):
            raise ValueError('vision_mode must be off, direct or structured')
        self.cli = cli or find_codex()
        self.timeout = timeout
        self.model, self.reasoning, self.vision_mode = model, reasoning, vision_mode
        self.metadata = {**type(self).metadata, 'model': model, 'reasoning': reasoning,
                         'vision': vision_mode != 'off', 'vision_mode': vision_mode}

    def generate(self, request, directory):
        root = Path(directory).resolve()
        evidence, workspace, resources = root/'evidence', root/'workspace', root/'resources'
        persistent = request.get('protocol') in ('persistent_session','recorded_observation')
        arena = request.get('backend') == 'arena'
        sensor_mode=arena and request.get('task',{}).get('observation_mode')=='sensor_estimates_v1'
        if sensor_mode and request.get('observation',{}).get('observation_mode')!='sensor_estimates_v1':
            raise ValueError('task and snapshot observation mode differ')
        if persistent:
            workspace = root.parent/'workspace'
            # This marker lives outside the agent-writable workspace. Previously
            # seen privileged data cannot be made unseen by changing a flag.
            marker=workspace.with_name('workspace-observation-mode.json')
            mode='sensor_estimates_v1' if sensor_mode else 'legacy'
            previous=json.loads(marker.read_text()) if marker.exists() else None
            if (previous is not None and previous!=mode) or (sensor_mode and workspace.exists() and previous is None):
                raise ValueError('persistent workspace observation mode cannot change')
            marker.parent.mkdir(parents=True,exist_ok=True)
            write_json(marker,mode)
        evidence.mkdir(parents=True, exist_ok=True)
        workspace.mkdir(exist_ok=persistent)
        resources.mkdir(exist_ok=False)
        (workspace/'.tmp').mkdir(exist_ok=persistent)
        (workspace/'.cache').mkdir(exist_ok=persistent)
        package = Path(__file__).resolve().parent
        project = package.parent
        # Snapshot only public runtime code, not historical candidates, test cases or run results.
        if sensor_mode:
            public_sources=[package/'arena_public.py',package/'execution.py']
        elif arena:
            public_sources = [package/name for name in ('arena_control.py','arena_observation.py','arena_retreat.py','arena_lift.py','arena_grasp.py','arena_motion.py','arena_placement.py','arena_surfaces.py',
                'arena_alignment.py','arena_retraction.py','arena_placement_geometry.py','toolkit/box_alignment.py',
                'toolkit/hand_clearance.py','toolkit/acquisition_wrists.py','toolkit/convex_clearance.py',
                'arena_session.py','execution.py','toolkit/loaded_wrists.py','toolkit/arena_approach.py')]
        elif persistent:
            # The current world-frame API, not the legacy episode-frame Robot.
            public_sources = [package/name for name in (
                'models.py', 'execution.py', 'session_tools.py', 'session_task.py', 'stationary_task.py',
                'scene.py', 'sim_state.py', 'workstation_task.py', 'mobility_task.py', 'toolkit/stationary.py',
                'toolkit/arm_planner.py', 'toolkit/bimanual.py', 'toolkit/motion_sequence.py', 'toolkit/sonic_backend.py')]
        else:
            public_sources = [package/'__init__.py', package/'models.py', package/'observations.py',
                              *sorted((package/'toolkit').glob('*.py'))]
        hashes = {}
        for path in public_sources:
            relative = Path('g1cap')/path.relative_to(package)
            destination = resources/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            content = path.read_bytes()
            destination.write_bytes(content)
            hashes[relative.as_posix()] = digest(content)
        document = request['api'].replace(
            'Return source code as the source field of the requested JSON schema.',
            'Write your final submitted program to policy.py in the workspace.')
        if persistent:
            arena_document='arena_sensor.md' if sensor_mode else 'arena_box.md'
            documents=([package/'tool_docs'/arena_document] if arena else
                       [p for p in sorted((package/'tool_docs').glob('*.md')) if p.name not in ('arena_box.md','visual_observation.md')])
            if arena and self.vision_mode!='off' and not sensor_mode:documents.append(package/'tool_docs/visual_observation.md')
            for path in documents:
                content=path.read_bytes()
                relative=Path('tool_docs')/path.name
                destination=resources/relative
                destination.parent.mkdir(exist_ok=True)
                destination.write_bytes(content)
                hashes[relative.as_posix()]=digest(content)
            if not arena and not (resources/'tool_docs/README.md').exists():
                raise RuntimeError('persistent tool documentation is missing')
            document+= (f'\nRead tool_docs/{arena_document} for every available Arena operation.\n' if arena else
                        '\nRead tool_docs/README.md and tool_docs/programming.md, then the document for each tool you intend to use.\n')
            if self.vision_mode!='off' and not sensor_mode:document+='\nRead tool_docs/visual_observation.md before using images.\n'
        (resources/'API.md').write_text(document+'\n')
        read_roots = [p.resolve() for p in (project/'papers', project/'references/cap-x',
                      project/'references/ASPIRE', project/'references/GR00T-WholeBodyControl') if p.is_dir()]
        if sensor_mode:read_roots=[]  # Only reviewed sensor-tool resources, no legacy scene/solution context.
        (resources/'README.md').write_text(
            '# Trial resources\n\nRead API.md and task.json first. Python source is provided for inspection.\n'
            'The submitted worker supports one self-contained file using math and robot APIs; '
            'your workspace can contain other analysis/helper files. It has no simulator endpoint.\n\n'
            'Read-only paper/reference directories:\n'+''.join(f'- {p}\n' for p in read_roots))
        config = permission_config(workspace, resources, read_roots)
        config['model_reasoning_effort'] = self.reasoning
        write_json(evidence/'codex-config.json', config)
        write_json(evidence/'public-source-sha256.json', hashes)
        prompt = (f'Complete the single-turn G1 programming task in {resources}/task.json. '
                  f'Read {resources}/API.md and inspect the read-only resources as useful. '
                  'You may use shell commands and write helper files anywhere in this workspace. '
                  'Write the final, self-contained Python program defining run(robot, task) to policy.py. '
                  'Do not execute a robot trial: the parent will submit this file once after you finish. '
                  'No vision, simulator endpoint, network access, delegation or additional model calls. '
                  'The submitted policy has the restricted runtime documented in API.md; '
                  'workspace helper files are not bundled with it. Keep the program simple and readable. '
                  'Finish within the session budget; do not leave background processes. '
                  'Your final chat message can briefly describe what you wrote.')
        if persistent:
            prompt = (f'Continue the G1 task in the same physical episode. Read {resources}/task.json '
                      f'and {resources}/API.md; the task file includes current observations and prior feedback. '
                      'The simulation keeps running while you work. Prior actions and elapsed time remain. '
                      'Your workspace persists across revisions. Write policy.py defining run(robot, task), '
                      'starting with fresh robot.observe() and accounting for previous actions. '
                      'Use shell/Python as useful, but keep inspection focused and the submitted program simple. '
                      'Do not run a robot trial yourself: the parent submits your file to the existing session. '
                      'No network, additional model calls, delegation or background processes. '
                      'The submitted worker supports the documented robot APIs and math; helper files are not bundled. '
                      'Finish promptly with a brief final message.')
        if request.get('protocol')=='recorded_observation':
            prompt=prompt.replace('Continue the G1 task in the same physical episode.',
                                  'Write code from a saved observation for an offline evaluation. No simulation is running.')
            prompt=prompt.replace('The simulation keeps running while you work. Prior actions and elapsed time remain.',
                                  'The packet records a historical state. This exercise does not execute or validate physical motion.')
            prompt=prompt.replace('the parent submits your file to the existing session.',
                                  'the parent saves your code for offline review only.')
        (evidence/'prompt.txt').write_text(prompt)
        # The trusted CLI retains saved subscription auth; shell commands inherit only config's env.
        environment = {k:v for k,v in os.environ.items() if k in {
            'PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'SHELL', 'CODEX_HOME', 'SSL_CERT_FILE', 'SSL_CERT_DIR'}}
        environment['TMPDIR'] = str(workspace/'.tmp')
        metadata = {**self.metadata, 'status': 'setup_error', 'usage': None,
                    'cli_started': False, 'wall_timeout_s': self.timeout}
        start = time.monotonic()
        process = None
        try:
            # A revision must produce its own note, even though helper files persist.
            if self.vision_mode=='structured':
                (workspace/'observation.md').unlink(missing_ok=True)
            snapshots=request.get('visual_observations',[])
            if self.vision_mode=='off' and (snapshots or request.get('visual_root')):
                raise ValueError('text-only generation cannot receive image resources')
            paths=[]
            public_request={k:v for k,v in request.items() if k not in ('api','visual_root','visual_observations')}
            if self.vision_mode!='off':
                if not arena or not persistent:raise ValueError('vision requires a persistent Arena session')
                if sensor_mode and any(p['observation'].get('observation_mode')!='sensor_estimates_v1' for p in snapshots):
                    raise ValueError('camera history observation mode differs')
                snapshots=stage_visual_observations(snapshots,request['visual_root'],resources,
                                                    request['observation']['session_id'])
                if snapshots[-1]['observation']!=request['observation']:
                    raise ValueError('current request state must match the latest camera packet')
                public_request['visual_observations']=snapshots
                paths=[resources/f['file'] for p in snapshots for f in p['frames']]
                metadata.update(image_count=len(paths),visual_observations=snapshots)
                prompt+=(' Attached images are onboard head RGB, previous then current if two packets. Read tool_docs/arena_sensor.md. '
                    if sensor_mode else ' Attached images follow task.json packet order: previous then current if two packets, head then overview in each. Read tool_docs/visual_observation.md. ')
                if self.vision_mode=='structured':
                    prompt+='Write observation.md with brief visible facts, changes, uncertainty, measured feedback and next code action before writing policy.py. '
            else:metadata['image_count']=0
            write_json(resources/'task.json',public_request)
            (evidence/'prompt.txt').write_text(prompt)
            verify_boundary(self.cli, config, workspace, resources, evidence, environment)
            auth = subprocess.run([self.cli, 'login', 'status'], env=environment,
                                  capture_output=True, text=True, timeout=15)
            if auth.returncode or 'Logged in using ChatGPT' not in auth.stdout+auth.stderr:
                raise RuntimeError('Codex must be signed in using ChatGPT')
            command = [self.cli, 'exec', '--ignore-user-config', '--ignore-rules',
                       '--skip-git-repo-check', '--ephemeral', '--model', self.model,
                       *config_arguments(config), '--json', '--output-last-message',
                       str(evidence/'final-message.txt'),
                       *[part for path in paths for part in ('--image',str(path))], '--', '-']
            write_json(evidence/'command.json', command)
            with (evidence/'codex-events.jsonl').open('wb') as events, (evidence/'codex-stderr.txt').open('wb') as errors:
                metadata.update(status='running', cli_started=True)
                process = subprocess.Popen(command, cwd=workspace, env=environment, stdin=subprocess.PIPE,
                                           stdout=events, stderr=errors, start_new_session=True)
                try:
                    process.communicate(prompt.encode(), timeout=self.timeout)
                    metadata['status'] = 'completed' if process.returncode == 0 else 'failed'
                except subprocess.TimeoutExpired:
                    metadata['status'] = 'timeout'
                finally:
                    # Also stop any children that remain in the session's process group.
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                    metadata['returncode'] = process.returncode
            events_path = evidence/'codex-events.jsonl'
            if events_path.stat().st_size <= 16_000_000:
                for line in events_path.read_text(errors='replace').splitlines():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get('type') == 'turn.completed':
                        metadata['usage'] = event.get('usage')
            if metadata['status'] != 'completed':
                raise RuntimeError(f'Codex generation {metadata["status"]}; see codex-stderr.txt')
            policy = workspace/'policy.py'
            # Open without following links; snapshot bytes before the trusted runner executes them.
            try:
                descriptor = os.open(policy, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                with os.fdopen(descriptor, 'rb') as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode) or info.st_size > 65536:
                        raise ValueError('not a bounded regular file')
                    source = stream.read(65537).decode()
                if not source.strip() or len(source.encode()) > 65536:
                    raise ValueError('empty or oversized source')
            except (OSError, ValueError) as error:
                metadata['status'] = 'invalid_submission'
                raise RuntimeError('Codex must write a nonempty regular policy.py of at most 65536 bytes') from error
            metadata['code_sha256'] = digest(source.encode())
            if self.vision_mode=='structured':
                note=workspace/'observation.md'
                if note.is_symlink() or not note.is_file() or not 0<note.stat().st_size<=8192:
                    metadata['status']='context_protocol_error'
                    (evidence/'policy.py').write_text(source)
                    raise RuntimeError('structured vision requires a bounded regular observation.md')
                (evidence/'observation.md').write_bytes(note.read_bytes())
            return source
        except Exception as error:
            metadata['error'] = f'{type(error).__name__}: {error}'[:4000]
            raise
        finally:
            metadata['wall_time_s'] = round(time.monotonic()-start, 3)
            write_json(evidence/'generation.json', metadata)
