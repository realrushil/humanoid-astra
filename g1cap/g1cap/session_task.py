"""Fixed world-frame approach/reach task, scored continuously across code revisions."""
from dataclasses import asdict, dataclass
import math
from .models import finite_number
from .stationary_task import state_fault


@dataclass(frozen=True)
class SessionTask:
    approach_world_xy: list
    wrist_target_world: list
    height: float = .70
    deadline: float = 600.
    approach_tolerance: float = .08
    final_base_tolerance: float = .15
    wrist_tolerance: float = .05
    dwell: float = .5

    def __post_init__(self):
        for value, size in ((self.approach_world_xy,2),(self.wrist_target_world,3)):
            if len(value)!=size or not all(finite_number(v) for v in value):
                raise ValueError('finite world-frame targets required')
        if not finite_number(self.height) or not .60<=self.height<=.85:
            raise ValueError('height must be in [0.60,0.85] metres')
        if not finite_number(self.deadline) or not 0<self.deadline<=600:
            raise ValueError('deadline must be in (0,600] simulation seconds')
        if (self.approach_tolerance,self.final_base_tolerance,self.wrist_tolerance,self.dwell)!=(.08,.15,.05,.5):
            raise ValueError('first-demo scoring tolerances are fixed')

    def to_dict(self):
        return dict(asdict(self),name='approach_reach',frame='mujoco_world_z_up',
                    observation='privileged_simulator_state',endpoint='right_wrist_yaw_link')


class SessionEvaluator:
    def __init__(self,task,initial):
        self.task=task
        self.start_time=initial['sim_time']
        self.initial_xy=tuple(initial['pelvis_position'][:2])
        self.previous=None
        self.approached=False
        self.approach_since=self.goal_since=self.unsupported_since=None
        self.terminal_reason=None
        self.path_length=0.
        self.metrics={}
        self.update(initial)

    def update(self,raw,stationary=False):
        if self.terminal_reason:
            return self.terminal_reason
        reason=state_fault(raw)
        if reason:
            self.terminal_reason=reason
            return reason
        t=raw['sim_time']
        if self.previous:
            if t<=self.previous['sim_time'] or raw['sequence']<=self.previous['sequence']:
                self.terminal_reason='state_order'
            elif t-self.previous['sim_time']>.100000001:
                self.terminal_reason='state_gap'
            if self.terminal_reason:
                return self.terminal_reason
            self.path_length+=math.dist(raw['pelvis_position'][:2],self.previous['pelvis_position'][:2])
        self.previous=raw
        elapsed=t-self.start_time
        supported=all(raw['foot_normal_forces'][s]>=5 for s in ('left','right'))
        # A walking phase permits single-foot support. Stationary commands do not.
        self.unsupported_since=(None if supported or not stationary else
                                t if self.unsupported_since is None else self.unsupported_since)
        if self.unsupported_since is not None and t-self.unsupported_since>=.25:
            self.terminal_reason='support_lost'
        if math.dist(raw['pelvis_position'][:2],self.initial_xy)>getattr(self.task,'workspace_radius',1.5):
            self.terminal_reason='workspace_exceeded'
        if elapsed>=self.task.deadline:
            self.terminal_reason=self.terminal_reason or 'timeout'
        settled=(supported and math.hypot(*raw['planar_velocity'])<=.05 and
                 abs(raw['yaw_rate'])<=.10 and abs(raw['tilt'])<=math.radians(15))
        self.metrics=dict(elapsed=elapsed,path_length_m=self.path_length,**self._goal(raw,settled))
        return self.terminal_reason

    def _goal(self,raw,settled):
        """Legacy approach/height goal; scene tasks override only this predicate."""
        t=raw['sim_time']
        base_error=math.dist(raw['pelvis_position'][:2],self.task.approach_world_xy)
        wrist_error=math.dist(raw['right_wrist_position'],self.task.wrist_target_world)
        approach_good=base_error<=self.task.approach_tolerance and settled
        self.approach_since=(t if self.approach_since is None else self.approach_since) if approach_good else None
        if self.approach_since is not None and t-self.approach_since>=self.task.dwell-1e-9:
            self.approached=True
        good=(self.approached and settled and base_error<=self.task.final_base_tolerance and
              wrist_error<=self.task.wrist_tolerance and
              abs(raw['pelvis_position'][2]-self.task.height)<=.03 and
              math.hypot(*raw['right_wrist_velocity_world'])<=.05)
        self.goal_since=(t if self.goal_since is None else self.goal_since) if good else None
        if self.goal_since is not None and t-self.goal_since>=self.task.dwell-1e-9 and not self.terminal_reason:
            self.terminal_reason='success'
        return dict(approached=self.approached,base_error_m=base_error,wrist_error_m=wrist_error,
                    height_error_m=abs(raw['pelvis_position'][2]-self.task.height))
