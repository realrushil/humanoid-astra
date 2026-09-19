"""Linux entry point: one physics owner, queued tools and continuous recording."""
import json
from pathlib import Path
import signal
import time
import re


def service_visual_snapshot(world,session,output):
    """Owner-thread mailbox: publish images and measured state from one recorded step."""
    from .arena_session import atomic_json
    from .visual_observation import validate_snapshot
    output=Path(output)
    for request_file in sorted((output/'session/visual-inbox').glob('*.json')):
        identity=request_file.stem
        if not re.fullmatch('[0-9a-f]{32}',identity):
            request_file.unlink()
            continue
        destination=output/'session/visual-snapshots'/identity
        destination.mkdir(parents=True,exist_ok=True)
        try:
            if request_file.stat().st_size>1024:raise ValueError('oversized snapshot request')
            request=json.loads(request_file.read_text())
            if request['session_id']!=session.session_id or request['snapshot_id']!=identity:
                raise ValueError('snapshot request identity mismatch')
            if not (destination/'manifest.json').exists() and not (destination/'error.json').exists():
                obs=session.observe()
                if any(obs[key]!=world.raw[key] for key in ('step','physics_step','time')):
                    raise ValueError('snapshot state and camera capture differ')
                packet=dict(version=1,session_id=session.session_id,snapshot_id=identity,
                            step=world.raw['step'],physics_step=world.raw['physics_step'],
                            sim_time_s=world.raw['time'],captured_at_unix_s=world.camera_captured_at_unix_s,
                            observation=obs,frames=world.export_camera_frames(destination))
                validate_snapshot(packet,destination,session.session_id)
                atomic_json(destination/'manifest.json',packet)
        except Exception as error:
            atomic_json(destination/'error.json',dict(error=f'{type(error).__name__}: {error}'))
        finally:request_file.unlink()
        return


def main():
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
    parser=get_isaaclab_arena_cli_parser()
    parser.add_argument('--session-out',type=Path,required=True)
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--max-rounds',type=int,default=2)
    parser.add_argument('--policy-port',type=int,default=15555)
    args,_=parser.parse_known_args()
    recipe=json.loads(args.recipe.read_text())
    out=args.session_out
    out.mkdir(parents=True,exist_ok=True)
    def interrupted(signum,frame):raise InterruptedError(f'owned runtime received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    with SimulationAppContext(args):
        import torch
        from isaaclab_arena_environments.cli import get_arena_builder_from_cli,get_isaaclab_arena_environments_cli_parser
        from .arena_world import ArenaWorld
        from .arena_session import ArenaSession,atomic_json,session_limits
        from .arena_task import ArenaBoxTask, ArenaRetreatTask, ArenaTransferTask
        parser=get_isaaclab_arena_environments_cli_parser(parser)
        args,overrides=parser.parse_known_args()
        builder=get_arena_builder_from_cli(args,hydra_overrides=overrides)
        limits=session_limits(recipe)
        world=session=score=None
        started=time.monotonic()
        try:
            with torch.inference_mode():
                world=ArenaWorld(builder,out/'physics',recipe,args.policy_port)
                task=dict(backend='arena',object_id='brown_box',hold_duration_s=1.,
                    instruction='Pick up the brown box with both hands, hold it stably for one second, then place it back on its source table and release it.',
                    deadline_s=recipe.get('deadline',60.),box_size_m=world.box_size,box_mass_kg=world.box_mass,
                    source_bounds=world.geometry['source_parts']['source_top'])
                task_kind=recipe.get('task_kind','source_return')
                if task_kind=='retreat_hold':
                    task['retreat_distance_m']=recipe.get('retreat_distance_m',.5)
                    task['instruction']=f"Pick up the brown box with both hands, retreat {task['retreat_distance_m']} metres along world minus X, then stop and hold it stably. Keep holding; do not place the box."
                    score=ArenaRetreatTask(task['retreat_distance_m'])
                elif task_kind=='source_return':score=ArenaBoxTask(task['source_bounds'])
                elif task_kind=='table_transfer':
                    if recipe.get('fixture') not in ('two_empty_equal_height_tables_v1','box_transfer_v1'):raise ValueError('table transfer requires the empty-table fixture')
                    task.update(surface_id='destination',minimum_travel_m=.5,
                        surfaces={name:parts[0] for name,parts in world.support_parts.items()},
                        instruction='Pick up the brown box with both hands, carry it from the grey source table to the green destination table, place the whole box on the green tabletop and release both hands. Finish standing stably.')
                    score=ArenaTransferTask(task['surface_id'],task['minimum_travel_m'],
                        box_size_m=world.box_size,box_mass_kg=world.box_mass)
                else:raise ValueError('unsupported Arena task_kind')
                task['task_kind']=task_kind
                task.update(limits,fixture=recipe.get('fixture','native_bin'))
                session=ArenaSession(world.control,task,out/'session',max_rounds=args.max_rounds,
                    worker_timeout=limits['worker_timeout_s'],
                    sensor_observation=world.sensor_observation if world.box_perception is not None else None,
                    controller_observation=world.controller_observation if world.box_perception is not None else None)
                def publish():
                    status=session.status()
                    status.update(metrics=score.metrics(),wall_elapsed_s=time.monotonic()-started)
                    atomic_json(out/'session/status.json',status)
                    return status
                session.tick(world.raw);score.update(world.raw);publish()
                atomic_json(out/'ready.json',dict(session_id=session.session_id,backend='arena'))
                while not session.terminal_reason:
                    service_visual_snapshot(world,session,out)
                    session.poll_submission()
                    if session.terminal_reason:break
                    raw=world.step()
                    session.tick(raw);score.update(raw)
                    if raw['time']>=task['deadline_s']:session.finish('episode_deadline')
                    elif world.native_terminal:
                        score.failure=score.failure or 'native_terminal'
                        session.finish('native_terminal')
                    elif time.monotonic()-started>=limits['wall_timeout_s']:session.finish('wall_deadline')
                    elif (out/'session/inbox/stop').exists():session.finish('requested_stop')
                    elif session.active_round is None:
                        if world.box_perception is None and score.metrics()['success']:session.finish('task_success')
                        elif len(session.rounds)>=session.max_rounds:session.finish('round_limit')
                    if world.step_index%10==0 or session.terminal_reason:publish()
        except Exception as error:
            atomic_json(out/'error.json',dict(error=repr(error)))
            if session:session.finish('runtime_error')
            raise
        finally:
            if session:
                session.close()
                status=session.status()
                status.update(metrics=score.metrics(),wall_elapsed_s=time.monotonic()-started)
                atomic_json(out/'session/summary.json',status)
                atomic_json(out/'session/status.json',status)
            if world:world.close()


if __name__=='__main__':main()
