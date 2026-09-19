"""Ordered world-frame walking goals, independent of navigation return values."""
from dataclasses import dataclass
import math
from .models import finite_number, wrap_yaw
from .scene import Scene
from .session_task import SessionEvaluator


@dataclass(frozen=True)
class MobilityTask:
    waypoints: tuple
    final_yaw: float | None = None
    deadline: float = 180.
    corridor_half_width: float = .6
    start_world_xy: tuple = (0., 0.)

    def __post_init__(self):
        if not isinstance(self.waypoints,tuple) or not 1 <= len(self.waypoints) <= 8:
            raise ValueError('one to eight immutable world-XY waypoints required')
        for p in (self.start_world_xy, *self.waypoints):
            if not isinstance(p,tuple) or len(p)!=2 or not all(finite_number(v) for v in p):
                raise ValueError('finite immutable world-XY pairs required')
        if self.final_yaw is not None and not finite_number(self.final_yaw):
            raise ValueError('finite final yaw in world radians required')
        if not finite_number(self.deadline) or not 0 < self.deadline <= 600:
            raise ValueError('deadline must be in (0,600] simulation seconds')
        if not finite_number(self.corridor_half_width) or not .2 <= self.corridor_half_width <= 1.:
            raise ValueError('corridor half width must be in [.2,1] metres')
        if any(math.dist(a,b)<.3 for a,b in zip((self.start_world_xy,*self.waypoints),self.waypoints)):
            raise ValueError('successive waypoint centres must be at least .3 m apart')
        if self.workspace_radius > 12.:
            raise ValueError('route exceeds 12 m workspace')

    @property
    def scene(self):
        return Scene('open_floor_mobility_v1', (), ())

    @property
    def workspace_radius(self):
        return max(math.dist(self.start_world_xy,p) for p in self.waypoints)+self.corridor_half_width+.3

    def corridor_distance(self, xy):
        # Distance to the fixed route polyline, metres in world XY. Rounded ends.
        distances=[]
        for a,b in zip((self.start_world_xy,*self.waypoints),self.waypoints):
            v=[b[i]-a[i] for i in range(2)]
            u=max(0.,min(1.,sum((xy[i]-a[i])*v[i] for i in range(2))/sum(x*x for x in v)))
            distances.append(math.dist(xy,[a[i]+u*v[i] for i in range(2)]))
        return min(distances)

    def to_dict(self):
        return dict(name='mobility', instruction='Visit the numbered floor regions in order and stop at each one. Finish with the requested heading.',
                    waypoints=[list(p) for p in self.waypoints], start_world_xy=list(self.start_world_xy),
                    final_yaw=self.final_yaw, deadline=self.deadline,
                    corridor_half_width=self.corridor_half_width, workspace_radius=self.workspace_radius,
                    waypoint_tolerance=.12, yaw_tolerance=.12, dwell=.5,
                    frame='mujoco_world_z_up', observation='privileged_simulator_state')


class MobilityEvaluator(SessionEvaluator):
    def __init__(self,task,initial):
        self.waypoint_index=0
        super().__init__(task,initial)

    def update(self,raw,stationary=False):
        if not self.terminal_reason and not finite_number(raw.get('pelvis_yaw')):
            self.terminal_reason='invalid_state'
        return self.terminal_reason or super().update(raw,stationary)

    def _goal(self,raw,settled):
        xy=raw['pelvis_position'][:2]
        corridor=self.task.corridor_distance(xy)
        if corridor > self.task.corridor_half_width:
            self.terminal_reason=self.terminal_reason or 'corridor_exceeded'
        index=min(self.waypoint_index,len(self.task.waypoints)-1)
        error=math.dist(xy,self.task.waypoints[index])
        yaw_error=0. if self.task.final_yaw is None else abs(wrap_yaw(raw['pelvis_yaw']-self.task.final_yaw))
        good=(settled and error<=.12 and (index<len(self.task.waypoints)-1 or yaw_error<=.12))
        t=raw['sim_time']
        self.goal_since=(t if self.goal_since is None else self.goal_since) if good else None
        dwell=0. if self.goal_since is None else t-self.goal_since
        if dwell>=.5-1e-9 and not self.terminal_reason:
            self.waypoint_index+=1
            self.goal_since=None
            if self.waypoint_index==len(self.task.waypoints): self.terminal_reason='success'
        return dict(waypoint_index=self.waypoint_index,waypoint_count=len(self.task.waypoints),
                    waypoint_error_m=error,final_yaw_error_rad=yaw_error,
                    corridor_distance_m=corridor,goal_dwell_s=dwell)
