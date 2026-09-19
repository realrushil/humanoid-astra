"""Independent truth verifier for the open-floor stationary qualification scene.

This is a component task, not the proposed R1/P1 benchmark. World-frame targets
are specified before commands; no tool return value is used for scoring.
"""
from dataclasses import asdict, dataclass
import math
from .models import finite_number


@dataclass(frozen=True)
class StationaryTask:
    stages: tuple  # Ordered dicts: target_world (3 m or None), height_band [low, high] m.
    deadline: float = 20.
    dwell: float = .5
    scene: object = None

    def __post_init__(self):
        if not self.stages or not finite_number(self.deadline) or not 0 < self.deadline <= 60.:
            raise ValueError('nonempty stages and deadline in (0,60] s required')
        if not finite_number(self.dwell) or not .5 <= self.dwell <= 2.:
            raise ValueError('dwell must be in [0.5,2] s')
        for stage in self.stages:
            low, high = stage['height_band']
            if not all(finite_number(v) for v in (low, high)) or not .55 <= low < high <= .9:
                raise ValueError('height band must lie in [0.55,0.90] m')
            if 'hand_goals' in stage:
                from .toolkit.bimanual import validate_goals
                validate_goals(stage['hand_goals'])
            target = stage['target_world']
            if target is not None and (len(target) != 3 or not all(finite_number(v) for v in target)):
                raise ValueError('target must be three finite world coordinates or None')

    def to_dict(self):
        return dict(asdict(self), frame='mujoco_world_z_up', scene=self.scene.to_dict() if self.scene else 'open_floor_target_markers',
                    observation='privileged_simulator_state', qualification='development_only')


def state_fault(raw):
    """Validate stationary evidence. Missing fields fail closed; truth is labeled."""
    if (raw.get('dds_domain') != 1 or raw.get('dds_interface') != 'lo'
            or raw.get('ready') is not True or raw.get('backend_ok') is not True
            or raw.get('no_support') is not True):
        return 'invalid_state'
    scalars = ('sim_time', 'state_age_s', 'yaw_rate', 'tilt')
    if not all(finite_number(raw.get(key)) for key in scalars):
        return 'invalid_state'
    for key, size in (('pelvis_position',3), ('planar_velocity',2),
                      ('right_wrist_position',3), ('right_wrist_velocity_world',3)):
        vector = raw.get(key)
        if not isinstance(vector,(list,tuple)) or len(vector) != size or not all(finite_number(v) for v in vector):
            return 'invalid_state'
    if type(raw.get('sequence')) is not int or raw['sequence'] < 0 or raw['sim_time'] < 0:
        return 'invalid_state'
    if not 0 <= raw['state_age_s'] <= .25:
        return 'state_stale'
    forces = raw.get('foot_normal_forces', {})
    if not all(finite_number(forces.get(side)) and forces[side] >= 0 for side in ('left','right')):
        return 'invalid_state'
    if not isinstance(raw.get('forbidden_contacts'), list):
        return 'invalid_state'
    if raw['forbidden_contacts']:
        return 'forbidden_contact'
    # Absolute development floor plus contact and tilt checks allow intended
    # lowering without disabling collapse detection or using initial-height ratio.
    if raw['pelvis_position'][2] < .5 or abs(raw['tilt']) > math.radians(30):
        return 'unsafe_posture'
    return None


class StationaryEvaluator:
    def __init__(self, task, initial):
        self.task = task
        self.start_time = initial['sim_time']
        self.origin = tuple(initial['pelvis_position'][:2])
        self.last_time = self.last_sequence = None
        self.stage_index = 0
        self.approached = False
        self.since = self.unsupported_since = None
        self.terminal_reason = None
        self.events, self.metrics = [], {}
        self.update(initial)

    def update(self, raw, stationary=True):
        if self.terminal_reason and self.terminal_reason != 'success':
            return self.terminal_reason
        already_successful = self.terminal_reason == 'success'
        self.terminal_reason = None
        fault = state_fault(raw)
        if fault:
            self.terminal_reason = fault
            return fault
        now = raw['sim_time']
        if self.last_time is not None:
            if now <= self.last_time or raw['sequence'] <= self.last_sequence:
                self.terminal_reason = 'state_order'
            elif now-self.last_time > .1+1e-9:
                self.terminal_reason = 'state_gap'
            if self.terminal_reason:
                return self.terminal_reason
        self.last_time, self.last_sequence = now, raw['sequence']
        supported = all(raw['foot_normal_forces'][s] >= 5. for s in ('left','right'))
        self.unsupported_since = (None if supported else
                                  now if self.unsupported_since is None else self.unsupported_since)
        displacement = math.dist(raw['pelvis_position'][:2], self.origin)
        if displacement > .15:
            self.terminal_reason = 'base_displacement'
        elif self.unsupported_since is not None and now-self.unsupported_since >= .25-1e-9:
            self.terminal_reason = 'support_lost'
        elif now-self.start_time > self.task.deadline+1e-9:
            self.terminal_reason = 'timeout'
        if self.terminal_reason:
            return self.terminal_reason
        if already_successful:
            self.terminal_reason = 'success'
            return 'success'
        stage = self.task.stages[self.stage_index]
        error = (math.dist(raw['right_wrist_position'],stage['target_world'])
                 if stage['target_world'] is not None else 0.)
        speed = math.hypot(*raw['right_wrist_velocity_world'])
        low, high = stage['height_band']
        good = (low <= raw['pelvis_position'][2] <= high and error <= .05 and speed <= .05
                and displacement <= .10 and supported and abs(raw['tilt']) <= math.radians(15)
                and math.hypot(*raw['planar_velocity']) <= .05 and abs(raw['yaw_rate']) <= .10)
        hand_goals=stage.get('hand_goals')
        paired_errors=None
        if hand_goals is not None:
            from .toolkit.bimanual import hand_errors
            try:paired_errors=hand_errors(raw,hand_goals)
            except ValueError:
                self.terminal_reason='invalid_hand_state'
                return self.terminal_reason
            good=good and all(e['position_m']<=.02 and e['orientation_rad']<=.15 and e['speed_m_s']<=.05
                              for e in paired_errors.values())
        self.metrics = dict(hand_errors=paired_errors,wrist_error=error, wrist_speed=speed, base_displacement=displacement,
                            height=raw['pelvis_position'][2], stage_index=self.stage_index,
                            elapsed=now-self.start_time)
        self.since = (now if self.since is None else self.since) if good else None
        if self.since is not None and now-self.since >= self.task.dwell-1e-9:
            self.events.append(dict(stage=self.stage_index, start=self.since, end=now))
            self.stage_index += 1
            self.since = None
            if self.stage_index == len(self.task.stages):
                self.terminal_reason = 'success'
        if self.terminal_reason is None and now-self.start_time >= self.task.deadline-1e-9:
            self.terminal_reason = 'timeout'
        return self.terminal_reason
