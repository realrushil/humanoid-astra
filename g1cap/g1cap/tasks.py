"""Small task samplers and JSON loading; positions use the episode frame."""
from dataclasses import fields
import math
import random
from .models import Task, finite_number


def make_task(name='waypoint', seed=0):
    if name not in ('waypoint', 'route', 'reach'):
        raise ValueError('name must be waypoint, route or reach')
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError('seed must be an integer')
    rng = random.Random(seed)
    if name == 'route':
        points = ((.45, rng.uniform(-.2, .2)), (.75, rng.uniform(.25, .4)))
        return Task(name, 'g1_ordered_waypoints_v0', seed, 30., (*points[-1], 0.),
                    rng.uniform(-.8, .8), 0., waypoints=points)
    if name == 'waypoint':
        position = (rng.uniform(.5,1.), rng.uniform(-.3,.3), 0.)
        yaw = rng.uniform(-math.pi/2, math.pi/2)
        deadline, task_id = 20., 'g1_waypoint_turn_v0'
    else:
        position = (.25+rng.uniform(.05,.15), -.20+rng.uniform(-.05,.05), .90+rng.uniform(-.05,.05))
        yaw = 0.
        deadline, task_id = 10., 'g1_standing_reach_v0'
    return Task(name, task_id, seed, deadline, position, yaw, rng.uniform(-math.pi,math.pi))


def task_from_dict(data):
    """Load trusted experiment JSON, ignoring derived public constraints."""
    values = {field.name: data[field.name] for field in fields(Task) if field.name in data}
    for name in ('target_position', 'neutral_right_wrist'):
        if name in values:
            values[name] = tuple(values[name])
    values['waypoints'] = tuple(tuple(point) for point in values.get('waypoints', ()))
    task = Task(**values)
    if task.name not in ('waypoint', 'route', 'reach'):
        raise ValueError('unknown task name')
    if not finite_number(task.deadline) or not 0 < task.deadline <= 60.:
        raise ValueError('task deadline must be in (0,60] seconds')
    positions = [(task.target_position, 3), (task.neutral_right_wrist, 3)]
    positions.extend((point, 2) for point in task.waypoints)
    for vector, size in positions:
        if len(vector) != size or not all(finite_number(v) for v in vector):
            raise ValueError('task positions must have finite coordinates')
    if not finite_number(task.target_yaw):
        raise ValueError('target yaw must be finite')
    if task.name == 'route' and (len(task.waypoints) < 2 or task.waypoints[-1] != task.target_position[:2]):
        raise ValueError('route needs at least two waypoints ending at target_position')
    if task.name != 'route' and task.waypoints:
        raise ValueError('ordered waypoints require a route task')
    return task
