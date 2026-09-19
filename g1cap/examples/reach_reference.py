def run(robot, task):
    robot.reach_right(task['target_position'])
    robot.hold(1.0)
