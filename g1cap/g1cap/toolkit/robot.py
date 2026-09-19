"""Serial bounded robot facade over explicit mock or measured SONIC backends.

MockBackend is not MuJoCo, SONIC, a balance controller, or robotics evidence.
The mock advances only inside bounded calls. Worker programs receive a remote client,
never this object or its evaluator/backend.
"""
from dataclasses import asdict, replace
import inspect
import math
import time
import uuid

from ..models import Evaluator, State, finite_number, wrap_yaw
from ..observations import ObservationStream


class StateUnavailable(RuntimeError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)

    def to_dict(self):
        return {'error': 'StateUnavailable', 'reason': self.reason}


def _json_safe(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)):
        return value if finite_number(value) else None
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k,v in value.items()}
    return repr(value)[:160]


class MockBackend:
    """Deterministic synthetic motion with instantaneous velocity response.

    Wrist movement is simple Cartesian interpolation. Nothing here represents
    humanoid balance, reach feasibility, inverse kinematics, or joint dynamics.
    """
    name = 'mock'

    def __init__(self, task):
        self.state = State(right_wrist_position=task.neutral_right_wrist)
        self.velocity = (0.,0.,0.)
        self.wrist_target = None

    def command_base(self, vx, vy, yaw_rate):
        self.velocity = (vx,vy,yaw_rate)

    def idle(self):
        self.velocity = (0.,0.,0.)

    def reach(self, position):
        self.idle()
        self.wrist_target = tuple(position)

    def hold_wrist(self):
        self.wrist_target = self.state.right_wrist_position if isinstance(self.state, State) else None

    def step(self, dt):
        old = self.state
        vx,vy,rate = self.velocity
        yaw = old.pelvis_yaw
        world_vx = vx*math.cos(yaw)-vy*math.sin(yaw)
        world_vy = vx*math.sin(yaw)+vy*math.cos(yaw)
        dx,dy = world_vx*dt, world_vy*dt
        x,y,z = old.pelvis_position
        wrist = old.right_wrist_position
        if self.wrist_target is not None:
            distance = math.dist(wrist,self.wrist_target)
            alpha = min(1., .20*dt/distance) if distance else 0.
            wrist = tuple(a+(b-a)*alpha for a,b in zip(wrist,self.wrist_target))
        else:
            wrist = (wrist[0]+dx,wrist[1]+dy,wrist[2])
        left = old.left_wrist_position
        self.state = replace(old, sim_time=old.sim_time+dt, sequence=old.sequence+1,
                             pelvis_position=(x+dx,y+dy,z), pelvis_yaw=wrap_yaw(yaw+rate*dt),
                             planar_velocity=(world_vx,world_vy), yaw_rate=rate,
                             right_wrist_position=wrist,
                             left_wrist_position=(left[0]+dx,left[1]+dy,left[2]))
        return self.state


class Robot:
    METHODS = frozenset(('observe','move_base','stop','hold','walk_to','turn_to','reach_right'))
    TICK = .02
    PROGRESS_WINDOW = 2.
    PROGRESS_THRESHOLD = .005
    TRACE_LIMIT = 12000

    def __init__(self, task, backend=None, on_event=None, observation_config=None):
        self.task = task
        self.backend = backend if backend is not None else MockBackend(task)
        self.observations = ObservationStream(observation_config) if observation_config is not None else None
        if self.backend.name not in ('mock', 'sonic'):
            raise ValueError('unknown robot backend')
        initial_state = getattr(self.backend, 'initial_state', None)
        if initial_state is None:
            initial_state = self.backend.state
        self.episode_id = str(uuid.uuid4())
        self.evaluator = Evaluator(task, start_height=initial_state.pelvis_position[2])
        self.ground_truth_evaluator = (Evaluator(task, start_height=initial_state.pelvis_position[2],
                                               stop_on_success=False) if self.observations else None)
        self.wall_deadline = None
        self.terminal_reason = None
        self.active_command_id = None
        self.trace = []
        self.trace_truncated = 0
        self._on_event = on_event
        self._score_sample(initial_state)
        # A delayed observer needs actual history. Physics idles during this
        # bounded warmup, and the episode clock continues; no data is invented.
        if self.observations:
            deadline = time.monotonic()+2.
            while not self.control_state.ready and not self.terminal_reason:
                if time.monotonic() >= deadline:
                    raise StateUnavailable('observation_warmup_timeout')
                self.backend.idle()
                self._tick(self.TICK)

    @property
    def control_state(self):
        """Latest causal estimated state, or backend truth in nominal mode."""
        truth = self.backend.state
        if self.observations is None:
            return truth
        return self.observations.read(truth)

    def _emit(self, event):
        event = _json_safe(event)
        if len(self.trace) >= self.TRACE_LIMIT:
            del self.trace[1]
            self.trace_truncated += 1
        self.trace.append(event)
        if self._on_event:
            self._on_event(event)

    def _snapshot(self, measured=None):
        measured = self.control_state if measured is None else measured
        if not isinstance(measured, State) or not measured.ready:
            return {'episode_id':self.episode_id,'backend':self.backend.name,
                    'episode_status':self.terminal_reason or 'running','state_available':False}
        state = asdict(measured)
        state.pop('backend_ok')
        state.update(episode_id=self.episode_id,backend=self.backend.name,
                     active_command_id=self.active_command_id,
                     episode_status=self.terminal_reason or 'running',
                     wrist_frames=({'right':'synthetic_right_wrist','left':'synthetic_left_wrist'}
                                   if self.backend.name == 'mock' else
                                   {'right':'right_wrist_yaw_link','left':'left_wrist_yaw_link'}))
        return _json_safe(state)

    def observe(self):
        self._drain_measured()
        state = self.control_state
        if not isinstance(state,State):
            self.finish('backend_failed')
            raise StateUnavailable('backend_failed')
        values = asdict(state)
        scalar_names = ('sim_time','pelvis_yaw','yaw_rate','tilt','state_age_s')
        vectors = {'pelvis_position':3,'planar_velocity':2,'right_wrist_position':3,
                   'left_wrist_position':3,'wrist_orientation':4}
        valid = all(finite_number(values[k]) for k in scalar_names)
        valid = valid and all(isinstance(values[k],(list,tuple)) and len(values[k]) == n
                              and all(finite_number(v) for v in values[k]) for k,n in vectors.items())
        if not valid or not state.backend_ok or not state.ready or state.sim_time < 0 or state.state_age_s < 0:
            self.finish('backend_failed')
            raise StateUnavailable('backend_failed')
        if state.state_age_s > .25:
            self.finish('state_stale')
            raise StateUnavailable('state_stale')
        return self._snapshot()

    def finish(self, reason):
        if self.terminal_reason is not None:
            return
        self.terminal_reason = str(reason)
        self.evaluator.terminal_reason = self.terminal_reason
        self.backend.idle()
        self.active_command_id = None
        self._emit({'type':'terminal','reason':reason,'observation':self._snapshot()})

    def summary(self):
        result = {'episode_id':self.episode_id,'backend':self.backend.name,
                'success':self.terminal_reason == 'success',
                'terminal_reason':self.terminal_reason,
                'trace_truncated':self.trace_truncated,
                'metrics':dict(self.evaluator.metrics),
                'tool_configuration':{'tick_seconds':self.TICK,
                    'progress_window_seconds':self.PROGRESS_WINDOW,
                    'progress_threshold':self.PROGRESS_THRESHOLD,
                    'qualification':('synthetic_only_unqualified' if self.backend.name == 'mock'
                                     else 'experimental_sonic_unqualified')}}
        if self.ground_truth_evaluator:
            truth = self.ground_truth_evaluator
            success = truth.success_reached and truth.terminal_reason is None
            reason = truth.terminal_reason or ('success' if success else
                     'estimated_success_only' if self.terminal_reason == 'success' else self.terminal_reason)
            result.update(success=success, terminal_reason=reason, metrics=dict(truth.metrics),
                          estimated_success=self.terminal_reason == 'success',
                          control_terminal_reason=self.terminal_reason,
                          observation_model=self.observations.metadata_label)
        return result

    def _result(self, operation, status, reason, start, command_id=None, errors=None, measurements=None):
        result = dict(episode_id=self.episode_id,command_id=command_id,
                      operation=operation,status=status,reason=reason,start_time=start,
                      end_time=getattr(self.control_state,'sim_time',start),
                      observation=self._snapshot(),measurements={**self.evaluator.metrics, **(measurements or {})},
                      errors=errors or {})
        result = _json_safe(result)
        self._emit({'type':'result','result':result})
        return result

    def dispatch(self, method, args, kwargs, episode_id):
        start = getattr(self.control_state, 'sim_time', self.evaluator.last_time or 0.)
        if episode_id != self.episode_id:
            return self._result(method,'rejected','stale_episode',start,
                                errors={'episode_id':'does not match current episode'})
        if self.terminal_reason and method != 'observe':
            return self._result(method,'cancelled',self.terminal_reason,start)
        if not isinstance(method,str) or method not in self.METHODS:
            return self._result(str(method),'rejected','invalid_arguments',start,
                                errors={'method':'operation is not public'})
        if not isinstance(args,list) or not isinstance(kwargs,dict) or not all(isinstance(k,str) for k in kwargs):
            return self._result(method,'rejected','invalid_arguments',start,
                                errors={'arguments':'expected a list and string-keyed dictionary'})
        try:
            inspect.signature(getattr(self,method)).bind(*args,**kwargs)
        except TypeError as exc:
            return self._result(method,'rejected','invalid_arguments',start,
                                errors={'arguments':str(exc)})
        return getattr(self,method)(*args,**kwargs)

    def move_base(self, vx, vy, yaw_rate, duration):
        return self._motion('move_base',dict(vx=vx,vy=vy,yaw_rate=yaw_rate,duration=duration))

    def stop(self, timeout=None):
        return self._motion('stop',dict(timeout=timeout))

    def hold(self, duration):
        return self._motion('hold',dict(duration=duration))

    def walk_to(self, position_xy, timeout=None):
        return self._motion('walk_to',dict(position_xy=position_xy,timeout=timeout))

    def turn_to(self, yaw, timeout=None):
        return self._motion('turn_to',dict(yaw=yaw,timeout=timeout))

    def reach_right(self, position, timeout=None):
        return self._motion('reach_right',dict(position=position,timeout=timeout))

    def _validate(self, op, data):
        errors = {}
        remaining = max(0.,self.task.deadline-self.control_state.sim_time)
        for field in ('vx','vy','yaw_rate','duration','yaw'):
            if field in data and not finite_number(data[field]):
                errors[field] = 'must be a finite number (not bool)'
        if 'duration' in data and 'duration' not in errors and not 0 < data['duration'] <= 2.:
            errors['duration'] = 'must be in (0, 2] seconds'
        if op == 'move_base' and not errors:
            if math.hypot(data['vx'],data['vy']) > .30+1e-12:
                errors['vx,vy'] = 'planar speed norm must not exceed 0.30 m/s'
            if abs(data['vy']) > .20:
                errors['vy'] = 'absolute lateral speed must not exceed 0.20 m/s'
            if abs(data['yaw_rate']) > .40:
                errors['yaw_rate'] = 'absolute yaw rate must not exceed 0.40 rad/s'
        if 'timeout' in data:
            timeout = data['timeout']
            if timeout is None:
                data['timeout'] = min(remaining,{'stop':3.,'walk_to':15.,'turn_to':8.,'reach_right':5.}[op])
            elif not finite_number(timeout) or not 0 < timeout <= remaining+1e-9:
                errors['timeout'] = 'must be positive, finite and no greater than remaining task time'
        for field,dimension in (('position_xy',2),('position',3)):
            if field not in data:
                continue
            value = data[field]
            if not isinstance(value,(list,tuple)) or len(value) != dimension or not all(finite_number(v) for v in value):
                errors[field] = 'must be a finite %s-vector' % dimension
            elif field == 'position_xy' and math.dist(value,self.control_state.pelvis_position[:2]) > 1.2+1e-12:
                errors[field] = 'target must be within 1.2 m of current base'
            elif field == 'position':
                constraints = self.task.to_dict()['constraints']
                if not all(lo <= v <= hi for v,lo,hi in zip(value,constraints['reach_envelope_min'],constraints['reach_envelope_max'])):
                    errors[field] = 'outside frozen synthetic reach envelope'
        if op == 'turn_to' and 'yaw' not in errors:
            if abs(wrap_yaw(data['yaw']-self.control_state.pelvis_yaw)) > math.pi/2+1e-12:
                errors['yaw'] = 'shortest-angle change must be at most pi/2'
        return errors

    def _tick(self, dt):
        state = self.backend.step(dt)
        if hasattr(self.backend, 'consume_samples'):
            self._drain_measured()
            return
        self._score_sample(state)

    def _score_sample(self, truth):
        """Keep privileged evaluation separate from online estimated completion."""
        state = truth
        if self.observations:
            reason = self.ground_truth_evaluator.update(truth)
            self._emit({'type':'ground_truth','observation':_json_safe(asdict(truth))})
            self.observations.push(truth)
            state = self.observations.read(truth)
            if reason:
                # A global clock / physical guard may stop the simulation, but
                # privileged goal attainment never controls the policy's flow.
                self.finish('timeout' if reason == 'timeout' else 'external_stop')
                return
            if not state.ready:
                return
        if self.evaluator.last_time is not None and state.sim_time <= self.evaluator.last_time:
            return  # The same delayed capture cannot earn additional dwell.
        reason = self.evaluator.update(state)
        self._emit({'type':'state','observation':self._snapshot(state)})
        if reason:
            self.finish(reason)

    def _drain_measured(self):
        """Score the ordered publisher history, including policy computation gaps."""
        if self.terminal_reason or not hasattr(self.backend, 'consume_samples'):
            return
        try:
            # Health is checked before accepting queued samples or awarding success.
            if not self.backend.state.backend_ok:
                raise RuntimeError('controller/simulation backend is unhealthy')
            for state in self.backend.consume_samples():
                scorer = self.ground_truth_evaluator or self.evaluator
                if scorer.last_time is not None and state.sim_time <= scorer.last_time:
                    continue
                self._score_sample(state)
                if self.terminal_reason:
                    break
        except Exception as exc:
            self._emit({'type':'backend_error','detail':str(exc)[:500]})
            self.finish('backend_failed')

    def _motion(self, op, data):
        start = getattr(self.control_state, 'sim_time', self.evaluator.last_time or 0.)
        if self.terminal_reason:
            return self._result(op,'cancelled',self.terminal_reason,start)
        if op not in getattr(self.backend, 'supported_operations', self.METHODS):
            return self._result(op,'rejected','unsupported_operation',start)
        try:
            self.observe()
        except StateUnavailable as exc:
            return self._result(op,'backend_error',exc.reason,start)
        errors = self._validate(op,data)
        if errors:
            return self._result(op,'rejected','invalid_arguments',start,errors=errors)
        if self.active_command_id is not None:
            return self._result(op,'rejected','command_active',start,errors={'command':'another motion is active'})
        command_id = str(uuid.uuid4())
        self.active_command_id = command_id
        self._emit({'type':'request','episode_id':self.episode_id,'command_id':command_id,
                    'operation':op,'arguments':data,'sim_time':start})
        end = min(self.task.deadline,start+data.get('duration',data.get('timeout',0.)))
        call_start_xy = self.control_state.pelvis_position[:2]
        dwell_since = None
        progress_at = start
        progress_error = None
        status, reason = 'completed','duration_elapsed'
        if op == 'reach_right':
            self.backend.reach(data['position'])
        try:
            while self.control_state.sim_time < end-1e-9 and not self.terminal_reason:
                if self.wall_deadline is not None and time.monotonic() >= self.wall_deadline:
                    self.finish('wall_timeout')
                    status,reason = 'cancelled','wall_timeout'
                    break
                state = self.control_state
                error = None
                if op == 'move_base':
                    self.backend.command_base(data['vx'],data['vy'],data['yaw_rate'])
                elif op == 'walk_to':
                    dx = data['position_xy'][0]-state.pelvis_position[0]
                    dy = data['position_xy'][1]-state.pelvis_position[1]
                    error = math.hypot(dx,dy)
                    if error <= .06:
                        self.backend.idle()
                    else:
                        wx,wy = dx*1.5,dy*1.5
                        c,s = math.cos(state.pelvis_yaw),math.sin(state.pelvis_yaw)
                        vx,vy = c*wx+s*wy,-s*wx+c*wy
                        scale = max(1.,math.hypot(vx,vy)/.25,abs(vy)/.20)
                        self.backend.command_base(vx/scale,vy/scale,0.)
                elif op == 'turn_to':
                    delta = wrap_yaw(data['yaw']-state.pelvis_yaw)
                    error = abs(delta)
                    self.backend.command_base(0.,0.,0. if error <= .08 else max(-.4,min(.4,delta*1.5)))
                else:
                    self.backend.idle()
                if op == 'reach_right':
                    error = math.dist(state.right_wrist_position,data['position'])
                if error is not None:
                    if progress_error is None:
                        progress_error = error
                    goal_tolerance = {'walk_to':.10,'turn_to':math.radians(10),'reach_right':.05}[op]
                    if state.sim_time-progress_at >= self.PROGRESS_WINDOW-1e-9:
                        if error > goal_tolerance and progress_error-error < self.PROGRESS_THRESHOLD:
                            status,reason = 'timed_out','no_progress'
                            break
                        progress_at,progress_error = state.sim_time,error
                self._tick(min(self.TICK,end-state.sim_time))
                if self.terminal_reason:
                    status,reason = 'cancelled',self.terminal_reason
                    break
                state = self.control_state
                settled = math.hypot(*state.planar_velocity) <= .05 and abs(state.yaw_rate) <= .10
                good = False
                dwell = .5
                if op == 'stop':
                    good = settled
                elif op == 'walk_to':
                    good = settled and math.dist(state.pelvis_position[:2],data['position_xy']) <= .10 and abs(state.tilt) <= math.radians(15)
                elif op == 'turn_to':
                    good = settled and abs(wrap_yaw(state.pelvis_yaw-data['yaw'])) <= math.radians(10) and math.dist(state.pelvis_position[:2],call_start_xy) <= .10 and abs(state.tilt) <= math.radians(15)
                elif op == 'reach_right':
                    good = math.dist(state.right_wrist_position,data['position']) <= .05
                    dwell = .25
                dwell_since = (state.sim_time if dwell_since is None else dwell_since) if good else None
                if dwell_since is not None and state.sim_time-dwell_since >= dwell-1e-9:
                    status,reason = 'completed','goal_reached'
                    break
            else:
                if self.terminal_reason:
                    status,reason = 'cancelled',self.terminal_reason
                elif op not in ('move_base','hold'):
                    status,reason = 'timed_out','command_timeout'
        except Exception as exc:
            self.finish('backend_failed')
            status,reason = 'backend_error','backend_failed'
            self._emit({'type':'backend_error','detail':str(exc)[:500]})
        finally:
            self.backend.idle()
            if op == 'reach_right' and status != 'completed':
                self.backend.hold_wrist()
            self.active_command_id = None
        measurements = {}
        state = self.control_state
        if isinstance(state, State):
            measurements['command_base_displacement'] = math.dist(state.pelvis_position[:2],call_start_xy)
            if op == 'walk_to':
                measurements['goal_position_error'] = math.dist(state.pelvis_position[:2],data['position_xy'])
            elif op == 'turn_to':
                measurements['goal_heading_error'] = abs(wrap_yaw(state.pelvis_yaw-data['yaw']))
            elif op == 'reach_right':
                measurements['goal_wrist_error'] = math.dist(state.right_wrist_position,data['position'])
        return self._result(op,status,reason,start,command_id,measurements=measurements)

    def policy_exit(self):
        self._drain_measured()
        if self.terminal_reason:
            return
        duration = min(1.,self.task.deadline-self.control_state.sim_time)
        if duration > 0:
            self.hold(duration)
        if not self.terminal_reason:
            self.finish('policy_exited_early')
