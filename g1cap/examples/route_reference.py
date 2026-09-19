"""A readable baseline: visit each target, settle, then face the final heading."""


def run(robot, task):
    waypoints = task.get('waypoints') or [task['target_position'][:2]]
    for position in waypoints:
        robot.walk_to(position)
        robot.hold(.6)
    robot.turn_to(task['target_yaw'])
    robot.hold(1.)
