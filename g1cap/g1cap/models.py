"""Public immutable contracts and measured-state task evaluator."""
from dataclasses import asdict, dataclass
import math


def wrap_yaw(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


@dataclass(frozen=True)
class Task:
    name: str
    task_id: str
    seed: int
    deadline: float
    target_position: tuple
    target_yaw: float
    spawn_yaw: float
    neutral_right_wrist: tuple = (.25, -.20, .90)
    waypoints: tuple = ()  # Ordered (x, y) targets; empty for a single-target task.

    def to_dict(self):
        value = asdict(self)
        value['target_position'] = list(self.target_position)
        value['neutral_right_wrist'] = list(self.neutral_right_wrist)
        value['constraints'] = {
            'position_tolerance': .05 if self.name == 'reach' else .10,
            'heading_tolerance': math.radians(10),
            'success_dwell': .5, 'max_settled_speed': .05,
            'max_settled_yaw_rate': .10, 'max_success_tilt': math.radians(15),
            'fall_tilt': math.radians(45), 'fall_height_fraction': .65,
            'fall_dwell': .10, 'workspace_radius': 2.,
            'reach_base_success_radius': .10, 'reach_base_failure_radius': .20,
            'reach_envelope_min': [.20, -.30, .80],
            'reach_envelope_max': [.45, -.10, 1.00],
        }
        return value


@dataclass(frozen=True)
class State:
    sim_time: float = 0.
    sequence: int = 0
    pelvis_position: tuple = (0., 0., .75)
    pelvis_yaw: float = 0.
    planar_velocity: tuple = (0., 0.)
    yaw_rate: float = 0.
    tilt: float = 0.
    right_wrist_position: tuple = (.25, -.20, .90)
    left_wrist_position: tuple = (.25, .20, .90)
    wrist_orientation: tuple = (1., 0., 0., 0.)
    state_age_s: float = 0.
    ready: bool = True
    backend_ok: bool = True


class Evaluator:
    """Scores measured state, independently of reported command status.

    Tick timestamps must increase. Dwell starts with the first observed qualifying
    sample; unobserved intervals before that sample earn no dwell credit.
    """
    def __init__(self, task, start_height=.75, stop_on_success=True):
        self.task = task
        self.start_height = start_height
        self.terminal_reason = None
        self.last_time = None
        self.success_since = None
        self.fall_since = None
        self.max_tilt = 0.
        self.metrics = {}
        self.waypoint_index = 0
        self.stop_on_success = stop_on_success
        self.success_reached = False

    def update(self, state):
        if self.terminal_reason:
            return self.terminal_reason
        if not isinstance(state, State):
            self.terminal_reason = 'backend_failed'
            return self.terminal_reason
        vectors = [('pelvis_position', 3), ('planar_velocity', 2),
                   ('right_wrist_position', 3), ('left_wrist_position', 3),
                   ('wrist_orientation', 4)]
        valid = all(isinstance(getattr(state, key), (tuple, list))
                    and len(getattr(state, key)) == size
                    and all(finite_number(x) for x in getattr(state, key))
                    for key, size in vectors)
        valid = valid and all(finite_number(getattr(state, key)) for key in
                              ('sim_time', 'pelvis_yaw', 'yaw_rate', 'tilt', 'state_age_s'))
        if (not valid or not state.ready or not state.backend_ok or
                state.sim_time < 0 or state.state_age_s < 0 or
                (self.last_time is not None and state.sim_time <= self.last_time)):
            self.terminal_reason = 'backend_failed'
            return self.terminal_reason
        if state.state_age_s > .25:
            self.terminal_reason = 'state_stale'
            return self.terminal_reason
        self.last_time = state.sim_time
        self.max_tilt = max(self.max_tilt, abs(state.tilt))
        x, y, height = state.pelvis_position
        targets = self.task.waypoints or (self.task.target_position[:2],)
        tx, ty = targets[min(self.waypoint_index, len(targets)-1)]
        position_error = math.hypot(x-tx, y-ty)
        heading_error = abs(wrap_yaw(state.pelvis_yaw-self.task.target_yaw))
        wrist_error = math.dist(state.right_wrist_position, self.task.target_position)
        displacement = math.hypot(x, y)
        speed = math.hypot(*state.planar_velocity)
        self.metrics = dict(position_error=position_error, heading_error=heading_error,
                            wrist_error=wrist_error, base_displacement=displacement,
                            planar_speed=speed, yaw_rate=abs(state.yaw_rate),
                            max_tilt=self.max_tilt, elapsed_time=state.sim_time)
        if self.task.name == 'route':
            self.metrics.update(waypoints_completed=self.waypoint_index,
                                waypoints_total=len(targets))
        falling = abs(state.tilt) > math.pi/4 or height < .65*self.start_height
        self.fall_since = (state.sim_time if self.fall_since is None else self.fall_since) if falling else None
        reason = None
        if self.fall_since is not None and state.sim_time-self.fall_since >= .1-1e-9:
            reason = 'fell'
        elif self.task.name in ('waypoint', 'route') and displacement > 2.:
            reason = 'left_workspace'
        elif self.task.name == 'reach' and displacement > .20:
            reason = 'base_displacement_exceeded'
        settled = speed <= .05 and abs(state.yaw_rate) <= .10 and abs(state.tilt) <= math.radians(15)
        # A pending height-based fall also prevents a success dwell.
        good = settled and not falling
        if self.task.name in ('waypoint', 'route'):
            final_target = self.waypoint_index >= len(targets)-1
            good = good and position_error <= .10
            if final_target:
                good = good and heading_error <= math.radians(10)
        else:
            good = good and wrist_error <= .05 and displacement <= .10
        self.success_since = (state.sim_time if self.success_since is None else self.success_since) if good else None
        if reason is None and self.success_since is not None and state.sim_time-self.success_since >= .5-1e-9:
            if self.task.name == 'route':
                self.waypoint_index = min(self.waypoint_index+1, len(targets))
                self.metrics['waypoints_completed'] = self.waypoint_index
            if self.task.name != 'route' or self.waypoint_index == len(targets):
                self.success_reached = True
                if self.stop_on_success:
                    reason = 'success'
            self.success_since = None
        if reason is None and state.sim_time >= self.task.deadline-1e-9:
            reason = 'timeout'
        self.terminal_reason = reason
        return reason
