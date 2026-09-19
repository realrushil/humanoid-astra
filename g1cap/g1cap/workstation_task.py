"""Workstation goal predicates, independent of tool return values and code."""
from dataclasses import dataclass
import math
from .models import finite_number
from .scene import Scene
from .session_task import SessionEvaluator


@dataclass(frozen=True)
class WorkstationTask:
    scene: Scene
    target_id: str = 'blue_lower'
    deadline: float = 180.

    def __post_init__(self):
        self.scene.marker(self.target_id)
        if not finite_number(self.deadline) or not 0 < self.deadline <= 600:
            raise ValueError('deadline must be in (0,600] simulation seconds')

    def to_dict(self):
        marker = self.scene.marker(self.target_id)
        return dict(name='workstation_reach', scene=self.scene.name, target_id=self.target_id,
                    instruction=f'Go to the {marker.label} and hold your right wrist at the marker while standing stably. Choose your standing position and posture. Avoid contact with the workstations.',
                    frame='mujoco_world_z_up', observation='privileged_simulator_state',
                    endpoint='right_wrist_yaw_link', deadline=self.deadline,
                    wrist_tolerance=.05, dwell=.5, max_base_distance_from_start=1.5)


class WorkstationEvaluator(SessionEvaluator):
    """Share continuous fault/time accounting; replace only the goal predicate."""
    def _goal(self, raw, settled):
        target = self.task.scene.marker(self.task.target_id).position_world
        error = math.dist(raw['right_wrist_position'], target)
        good = settled and error <= .05 and math.hypot(*raw['right_wrist_velocity_world']) <= .05
        t = raw['sim_time']
        self.goal_since = (t if self.goal_since is None else self.goal_since) if good else None
        dwell = 0. if self.goal_since is None else t-self.goal_since
        if dwell >= .5-1e-9 and not self.terminal_reason:
            self.terminal_reason = 'success'
        return dict(wrist_error_m=error, goal_dwell_s=dwell, target_id=self.task.target_id)
