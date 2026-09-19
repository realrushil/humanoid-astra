"""Linux mailbox endpoint owning exactly one MuJoCo/SONIC session."""
import argparse
import json
import math
from pathlib import Path
import signal
import time

from .runner import digest, write_json
from .session import Session, atomic_json
from .session_task import SessionTask
from .sonic_runtime import SonicRuntime
from .toolkit.arm_planner import ArmPlanner


def task_from_recipe(recipe):
    """Select a fixed scene task before startup; None selects the legacy demo."""
    name=recipe.get('task','approach_reach')
    if name=='approach_reach':
        return None
    if name=='mobility':
        from .mobility_task import MobilityTask
        if set(recipe)-{'task','waypoints','final_yaw','deadline','corridor_half_width','start_world_xy','hand_posture','arm_posture'}:
            raise ValueError('unknown mobility recipe field')
        return MobilityTask(tuple(tuple(p) for p in recipe['waypoints']),
                            final_yaw=recipe.get('final_yaw'),deadline=recipe.get('deadline',180.),
                            corridor_half_width=recipe.get('corridor_half_width',.6),
                            start_world_xy=tuple(recipe.get('start_world_xy',(0.,0.))))
    if name!='workstation_reach':
        raise ValueError(f'unknown session task: {name}')
    if set(recipe)-{'task','target_id','deadline','hand_posture','arm_posture','layout_offset_xy'}:
        raise ValueError('unknown workstation recipe field')
    from .scene import workstation_scene
    from .workstation_task import WorkstationTask
    return WorkstationTask(workstation_scene(recipe.get('layout_offset_xy', (0.,0.))),
                           target_id=recipe.get('target_id','blue_lower'),
                           deadline=recipe.get('deadline',180.))


def demo_task(initial,recipe):
    """Freeze world targets from a declared startup-relative recipe, once only.

    This is scene construction, not goal movement after an action. Forward/left
    offsets follow the measured initial pelvis yaw, heights are absolute world Z.
    """
    forward=recipe.get('approach_forward',.18)
    wrist=recipe.get('wrist_offset',[.23,0.,-.05])
    c,s=math.cos(initial['pelvis_yaw']),math.sin(initial['pelvis_yaw'])
    target=[initial['right_wrist_position'][0]+c*wrist[0]-s*wrist[1],
            initial['right_wrist_position'][1]+s*wrist[0]+c*wrist[1],
            initial['right_wrist_position'][2]+wrist[2]]
    xy=[initial['pelvis_position'][0]+c*forward,initial['pelvis_position'][1]+s*forward]
    return SessionTask(xy,target,height=recipe.get('height',.70),deadline=recipe.get('deadline',600.))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--gpu',type=int,required=True)
    parser.add_argument('--port',type=int,default=15576)
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--max-rounds',type=int,default=4)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    inbox=args.out/'inbox'
    inbox.mkdir()
    recipe=json.loads(args.recipe.read_text()) if args.recipe else {}
    task=task_from_recipe(recipe)
    runtime=SonicRuntime(args.root,args.out/'runtime',args.gpu,args.port,
                         scene=task.scene if task is not None else None,
                         hand_posture=recipe.get('hand_posture','upstream_default'),
                         arm_posture=recipe.get('arm_posture','upstream_default'))
    session=None
    def interrupted(signum,frame):
        raise RuntimeError(f'session interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGHUP,interrupted)
    try:
        runtime.start()
        initial=runtime.publisher.latest_state()
        planner=ArmPlanner(initial['model_path'])
        task=task if task is not None else demo_task(initial,recipe)
        session=Session(runtime.publisher,planner,task,args.out/'session',max_rounds=args.max_rounds).start()
        package=Path(__file__).parent
        write_json(args.out/'manifest.json',dict(protocol='persistent_session',physics=True,
            session_id=session.session_id,recipe=recipe,backend=runtime.backend.metadata,
            source_sha256={str(p.relative_to(package)):digest(p.read_bytes()) for p in package.rglob('*.py')}))
        atomic_json(args.out/'ready.json',dict(session_id=session.session_id,task=task.to_dict()))
        while not session.status()['terminal_reason']:
            if (inbox/'stop').exists():
                session.finish('client_stopped')
                break
            index=len(session.rounds)
            request=inbox/f'round-{index:02d}.json'
            if request.exists():
                try:
                    if request.stat().st_size>100000:
                        raise ValueError('oversized submission')
                    value=json.loads(request.read_text())
                    if value['session_id']!=session.session_id:
                        raise ValueError('stale session submission')
                    session.execute(value['source'])
                except Exception as error:
                    write_json(args.out/'request_error.json',dict(error=f'{type(error).__name__}: {error}'))
                    session.finish('submission_error')
            time.sleep(.03)
        return 0
    except Exception as error:
        atomic_json(args.out/'error.json',dict(error=f'{type(error).__name__}: {error}'))
        return 1
    finally:
        if session is not None: session.close()
        runtime.close()


if __name__=='__main__':
    raise SystemExit(main())
