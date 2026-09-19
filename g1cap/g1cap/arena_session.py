"""Mailbox and Python-worker requests for one continuously stepped Arena world.

Only tick() touches the controller. Worker threads read copied observations and
wait on request events; they never call Isaac, Torch or the motion controller.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
from queue import Empty, Queue
import threading
import time
import uuid

from .arena_control import METHODS
from .execution import execute_policy
from .runner import digest


def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.part')
    temporary.write_text(json.dumps(value,allow_nan=False))
    temporary.replace(path)


def session_limits(recipe):
    """Finite wall budgets shared by launcher, physics owner and worker client."""
    limits={k:recipe.get(k,v) for k,v in {'wall_timeout_s':900.,'worker_timeout_s':600.}.items()}
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<v<=3600 for v in limits.values()):
        raise ValueError('Arena wall budgets must be finite seconds in(0,3600]')
    if limits['worker_timeout_s']>limits['wall_timeout_s']:raise ValueError('worker budget exceeds episode wall budget')
    return limits


class ArenaSession:
    def __init__(self,control,task,output,*,max_rounds=2,worker_timeout=600.,executor=execute_policy,sensor_observation=None,controller_observation=None):
        self.control,self.task=control,deepcopy(task)
        self.sensor_observation=sensor_observation
        self.controller_observation=controller_observation
        self.control_snapshot={}
        self.public_snapshot={}
        if sensor_observation is not None:
            from .arena_public import public_task
            self.task=public_task(task,sensor_control=controller_observation is not None)
        self.output=Path(output)
        self.output.mkdir(parents=True,exist_ok=True)
        (self.output/'inbox').mkdir(exist_ok=True)
        self.max_rounds,self.worker_timeout=max_rounds,worker_timeout
        self.executor=executor
        self.session_id=uuid.uuid4().hex
        self.queue=Queue()
        self.lock=threading.RLock()
        self.pending=None
        self.active_round=None
        self.round_deadline=None
        self.thread=None
        self.rounds=[]
        self.tool_results=[]
        self.snapshot={}
        self.received_at=time.monotonic()
        self.terminal_reason=None
        self.closed=threading.Event()
        self.trace=(self.output/'tool-trace.jsonl').open('w')

    def observe(self):
        with self.lock:
            keys=('time','step','physics_step','root_pos','root_quat','box_pos','box_quat','box_size_m','box_mass_kg',
                  'clearance','bilateral','supported','hand_forces_N','foot_upward_N',
                  'stance_clear','loaded_contacts','approach_clearance_m','hold_observation_v2','controller_phase',
                  'surfaces','box_bounds','turn_clearance_m','box_floor_contact_peak_N')
            obs=(deepcopy(self.public_snapshot) if self.sensor_observation is not None else
                 {k:deepcopy(self.snapshot[k]) for k in keys if k in self.snapshot})
            obs['state_age_s']=max(0.,time.monotonic()-self.received_at)
            obs['session_id']=self.session_id
            obs['episode_status']=self.public_terminal_reason() or 'running'
            return obs

    def public_terminal_reason(self):
        if self.sensor_observation is not None and self.terminal_reason:return 'episode_terminated'
        return self.terminal_reason

    def status(self):
        with self.lock:
            return dict(session_id=self.session_id,task=self.task,observation=self.observe(),
                        active_round=self.active_round,rounds=len(self.rounds),max_rounds=self.max_rounds,
                        terminal_reason=self.public_terminal_reason())

    def _resolve(self,ticket,result):
        if self.sensor_observation is not None:
            from .arena_public import public_reason
            result=dict(status=result['status'],reason=public_reason(result.get('reason')),
                        observation=self.observe())
        ticket['result']=deepcopy(result)
        self.tool_results.append(dict(round_id=ticket['round_id'],method=ticket['method'],result=deepcopy(result)))
        self.trace.write(json.dumps(dict(type='tool_result',round_id=ticket['round_id'],
            method=ticket['method'],result=result,sim_time=self.snapshot.get('time')),allow_nan=False)+'\n')
        self.trace.flush()
        ticket['done'].set()

    def dispatch(self,method,args,kwargs,round_id):
        with self.lock:
            if round_id!=self.active_round:return dict(status='rejected',reason='stale_round')
            if method=='observe':return self.observe()
            if self.terminal_reason:return dict(status='cancelled',reason=self.public_terminal_reason())
            deadline=self.round_deadline
            if time.monotonic()>=deadline:
                return dict(status='cancelled',reason='request_wall_timeout',observation=self.observe())
        if self.sensor_observation is not None:
            from .arena_public import METHODS as sensor_methods
            if method not in sensor_methods:
                return dict(status='rejected',reason='sensor_tool_unavailable')
        names={'wait':['duration'],'pickup_box':['object_id'],'lift_supported_box':[],'raise_held_box':['clearance_m'],
               'hold_box':['duration'],'return_box_to_source':[],'retreat_with_box':['distance_m'],
               'move_with_box':['distance_m'],'turn_with_box':['yaw_rad'],'place_box':['surface_id']}
        if method not in names or len(args)>len(names[method]):return dict(status='rejected',reason='invalid_request')
        parameters=dict(zip(names[method],args))
        if set(parameters)&set(kwargs):return dict(status='rejected',reason='duplicate_argument')
        parameters.update(kwargs)
        ticket=dict(method=method,args=parameters,round_id=round_id,done=threading.Event(),cancelled=False,result=None)
        self.queue.put(ticket)
        while not ticket['done'].wait(.05):
            if self.closed.is_set() or time.monotonic()>=deadline:
                ticket['cancelled']=True
                return dict(status='cancelled',reason='request_wall_timeout',observation=self.observe())
        return ticket['result']

    def tick(self,observation):
        """Main simulation thread: publish fresh state and advance queued requests."""
        control_obs=(self.controller_observation() if self.controller_observation is not None else observation)
        self.control_snapshot=deepcopy(control_obs)
        self.control.update(control_obs)
        with self.lock:
            self.snapshot=deepcopy(observation)
            self.snapshot['controller_phase']=self.control.phase
            if self.sensor_observation is not None:
                self.public_snapshot=deepcopy(self.sensor_observation(observation['time']))
                self.public_snapshot['controller_phase']=self.control.phase
            self.received_at=time.monotonic()
        if self.control.terminal_reason:
            self.finish(self.control.terminal_reason)
            return
        if self.pending:
            ticket=self.pending
            if ticket['cancelled'] or ticket['round_id']!=self.active_round:
                result=self.control.cancel('worker_request_cancelled',control_obs)
                self._resolve(ticket,result)
                self.pending=None
            elif self.control.result is not None:
                self._resolve(ticket,self.control.result)
                self.pending=None
        while self.pending is None:
            try:ticket=self.queue.get_nowait()
            except Empty:break
            if ticket['cancelled'] or ticket['round_id']!=self.active_round:
                self._resolve(ticket,dict(status='cancelled',reason='stale_round'))
                continue
            self.trace.write(json.dumps(dict(type='tool_request',round_id=ticket['round_id'],
                method=ticket['method'],arguments=ticket['args'],sim_time=observation['time']))+'\n');self.trace.flush()
            try:result=self.control.start(ticket['method'],ticket['args'],control_obs)
            except Exception as error:
                result=self.control.cancel(f'controller_error: {type(error).__name__}: {error}',control_obs)
            if result['status']=='running':self.pending=ticket
            else:self._resolve(ticket,result)

    def submit(self,source):
        """Start one isolated program without blocking the physics owner."""
        with self.lock:
            if self.terminal_reason or self.active_round or len(self.rounds)>=self.max_rounds:
                raise RuntimeError('session cannot accept another program')
            if not isinstance(source,str) or not 0<len(source.encode())<=65536:
                raise ValueError('source must contain at most 65536 UTF-8 bytes')
            index=len(self.rounds)
            folder=self.output/f'round-{index:02d}'
            folder.mkdir()
            (folder/'policy.py').write_text(source)
            round_id=f'{self.session_id}:{index}'
            self.active_round=round_id
            # Absolute wall seconds, shared by all calls and computation in this
            # revision. The physics owner's separate episode deadline still applies.
            deadline=time.monotonic()+self.worker_timeout
            self.round_deadline=deadline
            before=self.observe()
        def run():
            try:
                from .arena_public import METHODS as sensor_methods
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    execution=dict(status='wall_timeout')
                else:
                    execution=self.executor(source,self.task,round_id,self.dispatch,
                        wall_timeout=remaining,wall_deadline=deadline,max_requests=100,
                        tools=sensor_methods if self.sensor_observation is not None else METHODS)
            except Exception as error:execution=dict(status='worker_error',error=repr(error))
            with self.lock:
                result=dict(index=index,round_id=round_id,session_id=self.session_id,
                            source_sha256=digest(source.encode()),execution=execution,
                            tools=[deepcopy(r) for r in self.tool_results if r['round_id']==round_id],
                            start_observation=before,end_observation=self.observe())
                self.rounds.append(result)
                atomic_json(folder/'result.json',result)
                self.active_round=None
                self.round_deadline=None
        self.thread=threading.Thread(target=run,name='arena-python-worker',daemon=True)
        self.thread.start()

    def poll_submission(self):
        if self.active_round or self.terminal_reason or len(self.rounds)>=self.max_rounds:return
        path=self.output/'inbox'/f'round-{len(self.rounds):02d}.json'
        if not path.exists():return
        try:
            if path.stat().st_size>100000:raise ValueError('oversized submission')
            request=json.loads(path.read_text())
            if request['session_id']!=self.session_id:raise ValueError('stale session submission')
            self.submit(request['source'])
        except Exception as error:
            atomic_json(self.output/'request_error.json',dict(error=repr(error)))
            self.finish('submission_error')

    def finish(self,reason):
        """Called by the physics owner; resolve waiters while retaining state."""
        if self.terminal_reason:return
        with self.lock:self.terminal_reason=reason
        result=(self.control.result if self.control.terminal_reason and self.control.result
                else self.control.cancel(reason,self.control_snapshot))
        if self.pending:
            self._resolve(self.pending,result)
            self.pending=None
        while True:
            try:ticket=self.queue.get_nowait()
            except Empty:break
            self._resolve(ticket,dict(status='cancelled',reason=reason))

    def close(self):
        self.finish('closed')
        self.closed.set()
        if self.thread:self.thread.join(timeout=3.)
        atomic_json(self.output/'summary.json',self.status())
        self.trace.close()
