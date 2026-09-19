"""One coding-agent invocation, one submitted program, at most one episode."""
from pathlib import Path

from .toolkit.api import api_documentation
from .runner import digest, run_episode, write_json


def run_trial(agent, task, output_dir, *, backend='mock', episode_runner=run_episode):
    if backend not in ('mock', 'sonic'):
        raise ValueError('backend must be mock or sonic')
    if backend == 'sonic' and episode_runner is run_episode:
        raise ValueError('SONIC requires an explicit episode runner')
    if task.name != 'waypoint':
        raise ValueError('first-stage trials support waypoint arrival, heading and settling')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    evidence = output/'evidence'
    evidence.mkdir()
    request = {'instruction': 'Reach the target position, face the target heading, and settle.',
               'task': task.to_dict(), 'backend': backend, 'protocol': 'single_turn',
               'observation': 'Selected simulator state fields (privileged) for SONIC; synthetic state for mock. '
                              'Read fresh state with robot.observe() during execution. No camera input.',
               'api': api_documentation(backend)}
    write_json(evidence/'request.json', request)
    report = {'schema_version': 1, 'protocol': 'single_turn', 'backend': backend,
              'physics': backend == 'sonic', 'agent': agent.metadata,
              'generation_sessions': 1, 'episode_attempts': 0,
              'status': 'generation_error', 'episode': None, 'error': None}
    try:
        source = agent.generate(request, output)
        if not isinstance(source, str) or len(source.encode()) > 65536:
            raise ValueError('submitted policy must be a Python string of at most 65536 bytes')
        # Agent process has ended. Only the parent writes official submission/evidence.
        (evidence/'candidate.py').write_text(source)
        report['code_sha256'] = digest(source.encode())
        report['status'] = 'episode_error'
        report['episode_attempts'] = 1
        report['episode'] = episode_runner(source, task, evidence/'episode')
        report['status'] = 'completed'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'[:4000]
    write_json(evidence/'report.json', report)
    episode = report['episode'] or {}
    (evidence/'report.md').write_text(
        '# Single-turn trial\n\n'
        f'Backend: **{backend}**. ' + ('Synthetic mock; no physics measured.\n\n' if backend == 'mock'
                                     else 'MuJoCo/SONIC simulation; no hardware qualification.\n\n') +
        f'Status: {report["status"]}. Episode attempts: {report["episode_attempts"]}.\n\n'
        f'Task outcome: {episode.get("terminal_reason", "not executed")}. '
        f'Clean completion: {episode.get("clean_completion", False)}.\n\n'
        f'Error: {report["error"] or "none"}.\n')
    return report
