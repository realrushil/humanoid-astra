"""Mac-side artifact transport to the prepared Linux simulation project."""
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import uuid
from dataclasses import asdict


class RemoteSonicRunner:
    def __init__(self, host, root, gpu, port=15556, observation_config=None):
        if host.startswith('-') or not host or not root.startswith('/'):
            raise ValueError('explicit SSH host and absolute remote project root required')
        self.host, self.root, self.gpu, self.port = host, PurePosixPath(root), gpu, port
        self.observation_config = observation_config

    def __call__(self, source, task, output_dir):
        if task.name not in ('waypoint', 'route'):
            raise ValueError('real SONIC supports waypoint and route tasks')
        output = Path(output_dir)
        parent = output.parent
        if output.exists() or (parent/'remote-artifacts').exists():
            raise FileExistsError('use a fresh parent directory for each remote episode')
        remote = self.root/'runs'/('remote-'+uuid.uuid4().hex)
        policy = remote.with_suffix('.py')
        remote_task = remote.with_suffix('.task.json')
        local_policy = parent/'candidate.py'
        if local_policy.read_text() != source:
            raise ValueError('candidate artifact does not match requested source')
        subprocess.run(['scp', '-q', str(local_policy), f'{self.host}:{policy}'], check=True, timeout=30)
        task_file = parent/'submitted-task.json'
        task_file.write_text(json.dumps(task.to_dict(), indent=2)+'\n')
        subprocess.run(['scp', '-q', str(task_file), f'{self.host}:{remote_task}'], check=True, timeout=30)
        command = ['timeout', '--signal=TERM', '--kill-after=8s', '210s',
                   str(self.root/'.venv-sonic/bin/python'), '-u', '-m', 'g1cap.sonic_runtime',
                   '--root', str(self.root), '--out', str(remote), '--gpu', str(self.gpu),
                   '--port', str(self.port), '--policy', str(policy), '--task-file', str(remote_task)]
        if self.observation_config is not None:
            local_observation = parent/'submitted-observation.json'
            local_observation.write_text(json.dumps(asdict(self.observation_config), indent=2)+'\n')
            remote_observation = remote.with_suffix('.observation.json')
            subprocess.run(['scp', '-q', str(local_observation), f'{self.host}:{remote_observation}'],
                           check=True, timeout=30)
            command += ['--observation-file', str(remote_observation)]
        shell = 'cd '+shlex.quote(str(self.root))+' && '+shlex.join(command)
        completed = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                    self.host, shell], capture_output=True, text=True, timeout=230)
        (parent/'remote.stdout.txt').write_text(completed.stdout)
        (parent/'remote.stderr.txt').write_text(completed.stderr)
        (parent/'remote.json').write_text(json.dumps(dict(host=self.host, root=str(self.root),
            artifact_path=str(remote), returncode=completed.returncode), indent=2)+'\n')
        # Collect startup failures as well as successful policy execution evidence.
        received = parent/'remote-artifacts'
        subprocess.run(['scp', '-q', '-r', f'{self.host}:{remote}', str(received)], check=True, timeout=60)
        if not (received/'episode/summary.json').exists():
            raise RuntimeError('Remote initialization failed; see remote-artifacts/runtime/startup.json and logs')
        (received/'episode').rename(output)
        summary = json.loads((output/'summary.json').read_text())
        summary['remote_returncode'] = completed.returncode
        if completed.returncode:
            summary['clean_completion'] = False
            if summary['execution_status'] == 'completed':
                summary['execution_status'] = 'remote_runtime_error'
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary
