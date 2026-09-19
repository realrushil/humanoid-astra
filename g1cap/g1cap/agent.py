"""An explicit replay agent and an iteration loop independent of the LLM provider."""
from pathlib import Path
from .runner import run_episode, write_json
from .toolkit.api import api_documentation


class ReplayAgent:
    """Saved candidates verify feedback plumbing; this is not model generation."""
    metadata = {'kind': 'replay', 'model': None, 'reasoning': None}

    def __init__(self, sources):
        self.sources = iter(sources)

    def generate(self, request, directory):
        return next(self.sources)


def run_iterations(agent, task, output_dir, *, attempts=2, backend='mock', episode_runner=run_episode):
    if attempts not in (1, 2):
        raise ValueError('smoke runs allow one initial generation and at most one repair')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = {'schema_version': 1, 'backend': backend, 'physics': backend == 'sonic',
              'agent': agent.metadata, 'episodes': [], 'generation_errors': [], 'episode_errors': []}
    feedback = None
    previous_source = None
    for index in range(attempts):
        directory = output_dir/f'iteration-{index:02d}'
        directory.mkdir()
        request = {'task': task.to_dict(), 'backend': backend, 'previous_source': previous_source,
                   'feedback': feedback, 'iteration': index, 'api': api_documentation(backend)}
        write_json(directory/'request.json', request)
        try:
            source = agent.generate(request, directory)
            if not isinstance(source, str) or len(source.encode()) > 65536:
                raise ValueError('candidate must be a Python string of at most 65536 bytes')
        except Exception as error:
            failure = {'iteration': index, 'type': type(error).__name__, 'error': str(error)[:4000]}
            report['generation_errors'].append(failure)
            write_json(directory/'generation_error.json', failure)
            break
        (directory/'candidate.py').write_text(source)
        try:
            summary = episode_runner(source, task, directory/'episode')
        except Exception as error:
            failure = {'iteration': index, 'type': type(error).__name__, 'error': str(error)[:4000]}
            report['episode_errors'].append(failure)
            write_json(directory/'episode_error.json', failure)
            break
        report['episodes'].append(summary)
        # Keep a bounded diagnostic slice. Complete traces stay in episode artifacts.
        trace_lines = (directory/'episode/trace.jsonl').read_text().splitlines()
        execution = __import__('json').loads((directory/'episode/execution.json').read_text())
        feedback = {'summary': summary, 'execution_error': execution['error'],
                    'stderr_tail': execution['stderr'][-2000:], 'trace_tail': trace_lines[-12:]}
        previous_source = source
    write_json(output_dir/'report.json', report)
    label = ('**Backend: mock; no physics or G1 performance measured.**' if backend == 'mock'
             else '**Backend: MuJoCo/SONIC; experimental tools, measured simulation outcomes.**')
    lines = ['# Pipeline smoke run', '', label, '',
             f'Agent: {agent.metadata["kind"]}', '', '| Candidate | Task outcome | Worker |', '| --- | --- | --- |']
    for index, episode in enumerate(report['episodes']):
        lines.append(f'| {index} | {episode["terminal_reason"]} | {episode["execution_status"]} |')
    for failure in report['generation_errors']:
        lines += ['', f'Generation failed: {failure["type"]}: {failure["error"]}']
    for failure in report['episode_errors']:
        lines += ['', f'Episode infrastructure failed: {failure["type"]}: {failure["error"]}']
    (output_dir/'report.md').write_text('\n'.join(lines) + '\n')
    return report
