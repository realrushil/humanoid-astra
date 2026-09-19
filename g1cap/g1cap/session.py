"""One physical session, continuous evidence, and sequential replaceable workers."""
import copy
import inspect
import json
from pathlib import Path
import threading
import time
import uuid
import math

from .execution import execute_policy
from .runner import digest, write_json
from .session_task import SessionEvaluator
from .toolkit.sonic_backend import LeasedPlanner


def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    write_json(temporary,value)
    temporary.replace(path)


class Session:
    """Only the supervisor thread calls the live publisher's command methods.

    Tool threads change desired commands under a lock. Velocity and timed motion expire after
    150 ms without a tool refresh; supported stationary holds persist while the
    supervisor is healthy. The publisher retains its independent 250 ms lease.
    """
    def __init__(self,publisher,planner,task,output,*,max_rounds=4,wall_timeout=900.):
        if type(max_rounds) is not int or not 1<=max_rounds<=4:
            raise ValueError('max_rounds must be in [1,4]')
        if not 0<wall_timeout<=900:
            raise ValueError('wall timeout must be in (0,900] seconds')
        self.publisher=publisher
        self.task=task
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        self.session_id=uuid.uuid4().hex
        self.lock=threading.RLock()
        self.worker_lock=threading.Lock()
        self.stopped=threading.Event()
        self.thread=None
        self.initial=publisher.latest_state()
        self.raw=copy.deepcopy(self.initial)
        from .workstation_task import WorkstationTask, WorkstationEvaluator
        from .mobility_task import MobilityTask, MobilityEvaluator
        from .stationary_task import StationaryTask,StationaryEvaluator
        evaluator=(StationaryEvaluator if isinstance(task,StationaryTask) else
                   MobilityEvaluator if isinstance(task,MobilityTask) else
                   WorkstationEvaluator if isinstance(task,WorkstationTask) else SessionEvaluator)
        self.evaluator=evaluator(task,self.initial)
        self.sequence=self.initial['sequence']
        self.settled_since=None
        self.received_at=time.monotonic()
        self.desired=('idle',{})
        self.velocity_until=0.
        self.hold_anchor=None
        self.command_changed_at=time.monotonic()
        self.started_at=time.monotonic()
        self.wall_timeout=wall_timeout
        self.max_rounds=max_rounds
        self.rounds=[]
        self.current_tools=[]
        self.active_round=None
        self.events=(self.output/'trace.jsonl').open('w',buffering=1)
        from .session_tools import SessionTools
        self.tools=SessionTools(self,planner)
        write_json(self.output/'task.json',task.to_dict())
        write_json(self.output/'initial_state.json',self.initial)
        self._event('start',session_id=self.session_id,observation=self.initial)

    def _event(self,kind,**fields):
        with self.lock:
            self.events.write(json.dumps(dict(type=kind,wall_elapsed=time.monotonic()-self.started_at,
                                             sim_time=self.raw['sim_time'],**fields),allow_nan=False)+'\n')

    def start(self):
        if self.thread is not None:
            raise RuntimeError('session already started')
        self.thread=threading.Thread(target=self._supervise,name='g1-session-supervisor',daemon=True)
        self.thread.start()
        return self

    def _finish(self,reason):
        if self.evaluator.terminal_reason is None:
            self.evaluator.terminal_reason=reason
            self._event('terminal',reason=reason)
        self.desired=('idle',{})

    def finish(self,reason):
        with self.lock:
            self._finish(reason)

    def _set_command(self,kind,fields):
        with self.lock:
            if self.evaluator.terminal_reason:
                return
            if kind=='stationary' and (self.desired[0]!='stationary' or
                    (fields.get('positions') is not None and self.desired[1].get('positions') is None)):
                self.hold_anchor=copy.deepcopy(self.raw)
            self.desired=(kind,copy.deepcopy(fields))
            self.command_changed_at=time.monotonic()
            if kind in ('velocity','motion'):
                self.velocity_until=time.monotonic()+.15

    def command_velocity(self,vx,vy,yaw_rate):
        LeasedPlanner(0.,0.).command(vx,vy,yaw_rate,0.)
        self._set_command('velocity',dict(vx=vx,vy=vy,yaw_rate=yaw_rate))

    def command_motion(self,**fields):
        LeasedPlanner(0.,0.).motion(**fields,now=0.)
        self._set_command('motion',fields)

    def command_stationary(self,**fields):
        fields=dict(mode=0,height=-1.,positions=None,velocities=None,**fields) if not fields else {
            'mode':fields.get('mode',0),'height':fields.get('height',-1.),
            'positions':fields.get('positions'),'velocities':fields.get('velocities')}
        LeasedPlanner(0.,0.).stationary(**fields,now=0.)
        self._set_command('stationary',fields)

    def idle(self):
        self._set_command('idle',{})

    def _supervise(self):
        last_status=0.
        try:
            while not self.stopped.is_set():
                fresh=self.publisher.latest_state()
                samples=self.publisher.samples_after(self.sequence)
                with self.lock:
                    if fresh['state_age_s']>.25:
                        self._finish('state_stale')
                    for raw in samples:
                        self.raw=copy.deepcopy(raw)
                        self.received_at=time.monotonic()
                        self.sequence=raw['sequence']
                        settled=(all(raw.get('foot_normal_forces',{}).get(s,0)>=5 for s in ('left','right')) and
                                 math.hypot(*raw['planar_velocity'])<=.05 and abs(raw['yaw_rate'])<=.10 and
                                 abs(raw['tilt'])<=math.radians(15))
                        self.settled_since=(raw['sim_time'] if self.settled_since is None else self.settled_since) if settled else None
                        before=self.evaluator.terminal_reason
                        stationary=(self.desired[0]=='stationary' or
                                    (self.desired[0]=='motion' and self.desired[1]['mode'] in (0,4)))
                        reason=self.evaluator.update(raw,stationary=stationary)
                        self._event('state',observation=raw,metrics=dict(self.evaluator.metrics),
                                    command_mode=self.desired[0],stationary=stationary)
                        if reason and before is None:
                            self._event('terminal',reason=reason)
                    now=time.monotonic()
                    if now-self.started_at>=self.wall_timeout:
                        self._finish('wall_timeout')
                    kind,fields=self.desired
                    if self.evaluator.terminal_reason:
                        kind,fields='idle',{}
                    elif kind in ('velocity','motion') and now>=self.velocity_until:
                        self._event('fallback',reason=f'{kind}_lease_expired')
                        kind,fields='idle',{}
                    elif kind=='stationary' and fields.get('positions') is not None:
                        # A fixed-root arm path must not survive large base drift.
                        anchor=self.hold_anchor
                        yaw=math.atan2(math.sin(self.raw['pelvis_yaw']-anchor['pelvis_yaw']),
                                       math.cos(self.raw['pelvis_yaw']-anchor['pelvis_yaw']))
                        if (math.dist(self.raw['pelvis_position'],anchor['pelvis_position'])>.05 or
                                abs(yaw)>.15 or abs(self.raw['tilt']-anchor['tilt'])>.10):
                            self._event('fallback',reason='hold_configuration_changed')
                            kind,fields='idle',{}
                    self.desired=(kind,fields)
                    # Command selection and publish share this lock with terminal
                    # changes: no old command can be published after cancellation.
                    if kind=='velocity': self.publisher.set_velocity(**fields)
                    elif kind=='stationary': self.publisher.set_stationary(**fields)
                    elif kind=='motion': self.publisher.set_motion(**fields)
                    else: self.publisher.idle()
                    if now-last_status>=.2:
                        atomic_json(self.output/'status.json',self.status())
                        last_status=now
                self.stopped.wait(.01)
        except Exception as error:
            with self.lock:
                self._finish('backend_error')
                self._event('error',detail=f'{type(error).__name__}: {error}')
        finally:
            try: self.publisher.idle()
            except Exception: pass
            with self.lock:
                atomic_json(self.output/'status.json',self.status())

    def observe(self):
        with self.lock:
            raw=copy.deepcopy(self.raw)
            # Observation age includes time since capture, even if monitor failed.
            raw['state_age_s']+=max(0.,time.monotonic()-self.received_at)
            raw.update(frame='mujoco_world_z_up',session_id=self.session_id,
                       elapsed=raw['sim_time']-self.evaluator.start_time,
                       remaining=max(0.,self.task.deadline-(raw['sim_time']-self.evaluator.start_time)),
                       episode_status=self.evaluator.terminal_reason or 'running',
                       approached=self.evaluator.approached,
                       task_progress=dict(self.evaluator.metrics),
                       settled_for=0. if self.settled_since is None else raw['sim_time']-self.settled_since)
            return raw

    def status(self):
        with self.lock:
            return dict(session_id=self.session_id,terminal_reason=self.evaluator.terminal_reason,
                        success=self.evaluator.terminal_reason=='success',
                        active_round=self.active_round,rounds=len(self.rounds),max_rounds=self.max_rounds,
                        metrics=dict(self.evaluator.metrics),observation=self.observe(),
                        wall_elapsed=time.monotonic()-self.started_at,
                        task=self.task.to_dict(),
                        scene=self.tools.observe_scene() if getattr(self.task,'scene',None) else None)

    def dispatch(self,method,args,kwargs,round_id):
        with self.lock:
            if round_id!=self.active_round or round_id is None:
                return dict(status='rejected',reason='stale_round')
            if self.evaluator.terminal_reason and method!='observe':
                return dict(status='cancelled',reason=self.evaluator.terminal_reason)
        if method not in self.tools.METHODS:
            return dict(status='rejected',reason='unsupported_operation')
        try:
            function=getattr(self.tools,method)
            inspect.signature(function).bind(*args,**kwargs)
            result=function(*args,**kwargs)
        except (TypeError,ValueError) as error:
            result=dict(status='rejected',reason='invalid_arguments',detail=str(error))
        self._event('tool_result',round_id=round_id,method=method,result=result)
        if method!='observe':
            with self.lock:
                self.current_tools.append(dict(method=method,args=args,kwargs=kwargs,
                    **{k:result[k] for k in ('status','reason','detail','elapsed','wrist_error','hand_errors','replans','holding','planning') if k in result}))
        return result

    def execute(self,source):
        if not self.worker_lock.acquire(blocking=False):
            raise RuntimeError('another worker is active')
        try:
            with self.lock:
                if self.evaluator.terminal_reason:
                    raise RuntimeError('session is terminal')
                if len(self.rounds)>=self.max_rounds:
                    self._finish('revision_budget')
                    raise RuntimeError('revision budget exhausted')
                index=len(self.rounds)
                folder=self.output/f'round-{index:02d}'
                folder.mkdir(exist_ok=False)
                (folder/'policy.py').write_text(source)
                self.active_round=f'{self.session_id}:{index}'
                self.current_tools=[]
                round_id=self.active_round
                start=self.observe()
                self._event('round_start',round_id=round_id,code_sha256=digest(source.encode()))
            execution=execute_policy(source,self.task.to_dict(),round_id,self.dispatch,
                                     wall_timeout=90,max_requests=300)
            with self.lock:
                # Worker completion cancels leftover velocity, but keeps a valid
                # posture/arm hold under the continuously running supervisor.
                if self.desired[0] in ('velocity','motion'): self.desired=('idle',{})
                self.active_round=None
                result=dict(index=index,session_id=self.session_id,code_sha256=digest(source.encode()),
                            execution=execution,tools=copy.deepcopy(self.current_tools),
                            start_observation=start,end_observation=self.observe())
                self.rounds.append(result)
                self._event('round_end',result=result)
                atomic_json(folder/'result.json',result)
                if len(self.rounds)>=self.max_rounds: self._finish('revision_budget')
                return result
        finally:
            with self.lock:
                self.active_round=None
                if self.desired[0] in ('velocity','motion'): self.desired=('idle',{})
            self.worker_lock.release()

    def close(self):
        if self.stopped.is_set(): return
        self.finish('closed')
        self.stopped.set()
        if self.thread is not None: self.thread.join(timeout=3.)
        with self.lock:
            atomic_json(self.output/'summary.json',self.status())
            self.events.close()
