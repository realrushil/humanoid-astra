import argparse
import json
from pathlib import Path

from .agent import ReplayAgent, run_iterations
from .runner import run_episode
from .tasks import make_task, task_from_dict


def main():
    parser = argparse.ArgumentParser(description='G1 policy smoke harness: explicit mock or remote MuJoCo/SONIC')
    parser.add_argument('command', choices=['demo', 'run', 'codex-smoke', 'trial'])
    parser.add_argument('--out', type=Path, required=True, help='new artifact directory')
    parser.add_argument('--task', choices=['waypoint', 'route', 'reach'], default='waypoint')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--attempts', type=int, choices=[1, 2], default=None)
    parser.add_argument('--task-file', type=Path, help='trusted task JSON; overrides --task and --seed')
    parser.add_argument('--generation-timeout', type=float, default=180, help='trial Codex wall budget, at most 300 seconds')
    parser.add_argument('--backend', choices=['mock', 'sonic'], default='mock')
    parser.add_argument('--ssh-host')
    parser.add_argument('--remote-root')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()
    task = task_from_dict(json.loads(args.task_file.read_text())) if args.task_file else make_task(args.task, args.seed)
    if args.command == 'trial' and (task.name != 'waypoint' or args.attempts not in (None, 1)):
        parser.error('trial supports waypoint tasks and exactly one generation/submission')
    episode_runner = run_episode
    if args.backend == 'sonic':
        if args.command == 'run' or task.name == 'reach':
            parser.error('remote SONIC supports demo/codex-smoke/trial waypoint and route tasks')
        if not args.ssh_host or not args.remote_root or args.gpu is None:
            parser.error('SONIC requires --ssh-host, --remote-root and --gpu')
        from .remote_sonic import RemoteSonicRunner
        episode_runner = RemoteSonicRunner(args.ssh_host, args.remote_root, args.gpu)
    if args.command == 'trial':
        from .workspace_agent import WorkspaceCodexAgent
        from .trial import run_trial
        report = run_trial(WorkspaceCodexAgent(timeout=args.generation_timeout), task, args.out,
                           backend=args.backend, episode_runner=episode_runner)
    elif args.command == 'run':
        if args.policy is None:
            parser.error('run requires --policy')
        report = run_episode(args.policy.read_text(), task, args.out)
    else:
        if args.command == 'codex-smoke':
            from .codex_agent import CodexAgent
            agent = CodexAgent()
        else:
            initial = 'def run(robot, task):\n robot.hold(0.1)\n'
            goal = ('robot.walk_to(task["target_position"][:2])\n robot.turn_to(task["target_yaw"])'
                    if args.task == 'waypoint' else 'robot.reach_right(task["target_position"])')
            reference = ('def run(robot, task):\n '+goal+'\n robot.hold(1.0)\n')
            if args.task == 'route':
                reference = (Path(__file__).resolve().parent.parent/'examples/route_reference.py').read_text()
            agent = ReplayAgent([initial, reference])
        report = run_iterations(agent, task, args.out, attempts=args.attempts or 2,
                                backend=args.backend, episode_runner=episode_runner)
    if report.get('protocol') == 'single_turn':
        episode = report['episode'] or {}
        compact = {k:report[k] for k in ('protocol', 'backend', 'status', 'episode_attempts', 'error')}
        compact.update(outcome=episode.get('terminal_reason'), clean_completion=episode.get('clean_completion'))
        failed = report['status'] != 'completed' or episode.get('execution_status') != 'completed'
    elif 'episodes' in report:
        compact = {'backend': args.backend, 'agent': report['agent'], 'outcomes':
                   [r['terminal_reason'] for r in report['episodes']], 'generation_errors': report['generation_errors'],
                   'episode_errors': report['episode_errors']}
        failed = bool(report['generation_errors'] or report['episode_errors']) or any(r['execution_status'] != 'completed' for r in report['episodes'])
    else:
        compact = report
        failed = report['execution_status'] != 'completed'
    print(json.dumps(compact, indent=2))
    print(f'Artifacts: {args.out.resolve()}')
    print('MOCK ONLY — no physical simulation or G1 performance measured.' if args.backend == 'mock'
          else 'MuJoCo/SONIC simulation — experimental tools; no physical robot deployment.')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
