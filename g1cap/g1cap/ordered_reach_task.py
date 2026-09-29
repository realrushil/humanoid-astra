"""Two ordered workstation wrist goals followed by measured neutral standing."""
from dataclasses import dataclass
import math

from .models import finite_number
from .scene import Scene
from .session_task import SessionEvaluator
from .toolkit.arm_planner import UPPER_BODY


def upper_state(raw):
    """Read the named waist/arm encoders in radians; reject missing samples."""
    names = raw.get('body_joint_names')
    positions = raw.get('body_joint_positions')
    speeds = raw.get('body_joint_velocities')
    if (not isinstance(names, list) or not isinstance(positions, list)
            or not isinstance(speeds, list) or len(names) != len(positions)
            or len(names) != len(speeds) or len(names) != len(set(names))):
        raise ValueError('missing or malformed named body joint state')
    by_name = dict(zip(names, zip(positions, speeds)))
    if any(name not in by_name or not all(finite_number(v) for v in by_name[name])
           for name in UPPER_BODY):
        raise ValueError('missing or invalid waist/arm encoder state')
    return tuple(float(by_name[name][0]) for name in UPPER_BODY), tuple(
        float(by_name[name][1]) for name in UPPER_BODY)


@dataclass(frozen=True)
class OrderedReachTask:
    scene: Scene
    target_ids: tuple = ('blue_lower', 'blue_upper')
    deadline: float = 180.

    def __post_init__(self):
        if (not isinstance(self.target_ids, tuple) or len(self.target_ids) != 2
                or len(set(self.target_ids)) != 2):
            raise ValueError('exactly two distinct target ids are required')
        for name in self.target_ids:
            self.scene.marker(name)
        if not finite_number(self.deadline) or not 0 < self.deadline <= 600:
            raise ValueError('deadline must be in (0,600] simulation seconds')

    def to_dict(self):
        labels = [self.scene.marker(name).label for name in self.target_ids]
        return dict(name='ordered_reach', scene=self.scene.name,
                    target_ids=list(self.target_ids),
                    instruction=(f'Reach the {labels[0]}, then the {labels[1]}, '
                                 'holding each with a stable stance. Finally return to the '
                                 'measured startup upper-body posture and pelvis height.'),
                    frame='mujoco_world_z_up', observation='privileged_simulator_state',
                    endpoint='right_wrist_yaw_link', deadline=self.deadline,
                    wrist_tolerance=.05, dwell=.5,
                    neutral_joint_tolerance_rad=.12, neutral_height_tolerance_m=.04,
                    neutral_joint_speed_limit_rad_s=.10,
                    neutral_reference_source='measured_unsupported_startup_state',
                    max_base_distance_from_start=1.5)


class OrderedReachEvaluator(SessionEvaluator):
    """Keep historical target stages; score neutral only at the current state."""
    def __init__(self, task, initial):
        self.neutral_joints, _ = upper_state(initial)
        self.neutral_height = initial['pelvis_position'][2]
        if not finite_number(self.neutral_height):
            raise ValueError('finite startup pelvis height required')
        self.stage_index = 0
        self.current_joints, self.current_speeds = self.neutral_joints, (0.,) * len(UPPER_BODY)
        super().__init__(task, initial)

    def update(self, raw, stationary=False):
        if self.terminal_reason:
            return self.terminal_reason
        try:
            self.current_joints, self.current_speeds = upper_state(raw)
        except ValueError:
            self.terminal_reason = 'invalid_state'
            return self.terminal_reason
        return super().update(raw, stationary=stationary)

    def _goal(self, raw, settled):
        t = raw['sim_time']
        joint_error = max(abs(a - b) for a, b in zip(self.current_joints, self.neutral_joints))
        joint_speed = max(abs(v) for v in self.current_speeds)
        height_error = abs(raw['pelvis_position'][2] - self.neutral_height)
        wrist_speed = math.hypot(*raw['right_wrist_velocity_world'])
        target_id = self.task.target_ids[self.stage_index] if self.stage_index < 2 else None
        wrist_error = (math.dist(raw['right_wrist_position'],
                       self.task.scene.marker(target_id).position_world)
                       if target_id else None)
        good = (settled and wrist_speed <= .05 and
                (wrist_error <= .05 if target_id else
                 joint_error <= .12 and height_error <= .04 and joint_speed <= .10))
        self.goal_since = (t if self.goal_since is None else self.goal_since) if good else None
        dwell = 0. if self.goal_since is None else t - self.goal_since
        if dwell >= .5 - 1e-9 and not self.terminal_reason:
            self.stage_index += 1
            self.goal_since = None
            if self.stage_index == 3:
                self.terminal_reason = 'success'
        active_target = self.task.target_ids[self.stage_index] if self.stage_index < 2 else None
        active_error = (math.dist(raw['right_wrist_position'],
                        self.task.scene.marker(active_target).position_world)
                        if active_target else None)
        return dict(stage_index=self.stage_index, stage_count=3,
                    target_id=active_target, wrist_error_m=active_error,
                    neutral_height_error_m=height_error,
                    neutral_max_joint_error_rad=joint_error,
                    neutral_max_joint_speed_rad_s=joint_speed,
                    neutral_reference=dict(pelvis_height_m=self.neutral_height,
                        upper_body_joint_positions_rad=dict(zip(UPPER_BODY, self.neutral_joints))),
                    goal_dwell_s=dwell)
