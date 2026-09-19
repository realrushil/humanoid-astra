"""Offline SONIC planner field conversion, NOT a transport or verified controller.

Upstream revision 087f9ac01d46f6d8e4d0b73c01ae64799f292a38:
gear_sonic_deploy/.../include/localmotion_kplanner.hpp defines IDLE=0,
SLOW_WALK=1, WALK=2; the supported low speed range uses SLOW_WALK.
gear_sonic/utils/teleop/zmq/zmq_planner_sender.py owns build_planner_message.
This helper does not publish messages. The live adapter in sonic_backend.py
uses LeasedPlanner and the upstream builder directly. No wire bytes are reimplemented.
Heading must already be in the planner/world frame. Retain and integrate the
returned heading across a leased command, rather than resetting from noisy state.
"""
import math
from ..models import finite_number, wrap_yaw


def planner_command(vx, vy, yaw_rate, heading, dt):
    if not all(finite_number(v) for v in (vx,vy,yaw_rate,heading,dt)):
        raise ValueError('command fields must be finite numbers')
    speed = math.hypot(vx,vy)
    if speed > .30+1e-12 or abs(vy) > .20 or abs(yaw_rate) > .40 or not 0 < dt <= 2.:
        raise ValueError('command exceeds bounded velocity or timestep limits')
    movement = [0.,0.,0.]
    if speed:
        movement = [(vx*math.cos(heading)-vy*math.sin(heading))/speed,
                    (vx*math.sin(heading)+vy*math.cos(heading))/speed, 0.]
    updated = wrap_yaw(heading+yaw_rate*dt)
    return dict(mode=1 if speed else 0, movement=movement,
                facing=[math.cos(updated),math.sin(updated),0.],
                speed=speed if speed else -1., height=-1., heading=updated)
