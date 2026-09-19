"""Authoritative episode lifecycle and evidence; no generated code owns scoring."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import time

from .execution import execute_policy
from .toolkit.robot import Robot
from .observations import ObservationStream


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def digest(value):
    return hashlib.sha256(value).hexdigest()


def run_episode(source, task, output_dir, *, wall_timeout=10, max_requests=1000, backend=None,
                observation_config=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    robot = Robot(task, backend=backend, observation_config=observation_config)
    backend_name = robot.backend.name
    metadata = getattr(robot.backend, 'metadata', {'physics': False,
                        'robot_model': 'synthetic G1 interface', 'controller': None, 'checkpoint': None})
    configuration = {'task': task.to_dict(), 'backend': backend_name, 'wall_timeout': wall_timeout,
                     'max_requests': max_requests, 'schema_version': 1}
    if observation_config is not None:
        configuration['observation'] = {'delay_s': observation_config.delay_s,
            'position_bias_xy': list(observation_config.position_bias_xy), 'yaw_bias': observation_config.yaw_bias,
            'position_noise_std': observation_config.position_noise_std, 'yaw_noise_std': observation_config.yaw_noise_std,
            'seed': observation_config.seed, 'metadata': ObservationStream.metadata_label}
    package = Path(__file__).parent
    source_hashes = {p.relative_to(package).as_posix(): digest(p.read_bytes())
                     for p in sorted(package.rglob('*.py'))}
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'episode_id': robot.episode_id,
                'backend': backend_name, **metadata, 'python': platform.python_version(),
                'code_sha256': digest(source.encode()), 'package_source_sha256': source_hashes,
                'config_sha256': digest(json.dumps(configuration, sort_keys=True).encode()),
                'configuration': configuration}
    write_json(output_dir/'manifest.json', manifest)
    write_json(output_dir/'task.json', task.to_dict())
    write_json(output_dir/'initial_state.json', robot.trace[0]['observation'])
    if observation_config is not None:
        initial_observation = next((event['observation'] for event in robot.trace if event['type'] == 'state'),
                                   {'state_available':False})
        write_json(output_dir/'initial_observation.json', initial_observation)
    (output_dir/'policy.py').write_text(source)
    robot.wall_deadline = time.monotonic()+wall_timeout
    execution = execute_policy(source, task.to_dict(), robot.episode_id, robot.dispatch,
                               wall_timeout=wall_timeout, max_requests=max_requests)
    robot._drain_measured()
    if execution['status'] == 'completed':
        robot.policy_exit()
    elif robot.terminal_reason is None:
        robot.finish(execution['status'])
    summary = robot.summary()
    summary.update({'episode_id': robot.episode_id, 'execution_status': execution['status'],
                    'code_sha256': manifest['code_sha256'], 'backend': backend_name,
                    'physics': metadata['physics']})
    # Keep task attainment and worker health separate: success reached before a later
    # policy error remains visible, but is not a clean end-to-end execution.
    summary['clean_completion'] = summary['success'] and execution['status'] == 'completed'
    if observation_config is not None:
        summary['clean_completion'] = summary['clean_completion'] and summary['estimated_success']
    write_json(output_dir/'execution.json', execution)
    write_json(output_dir/'summary.json', summary)
    (output_dir/'stderr.txt').write_text(execution['stderr'])
    with (output_dir/'trace.jsonl').open('w') as stream:
        for event in robot.trace:
            stream.write(json.dumps(event, allow_nan=False) + '\n')
    return summary
