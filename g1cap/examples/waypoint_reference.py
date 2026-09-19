# Hand-written integration reference; no claim of G1 physics performance.
def run(robot, task):
    result = robot.walk_to(task['target_position'][:2])
    if result['status'] not in ('completed', 'cancelled'):
        print('walk result:', result['reason'])
        return
    robot.turn_to(task['target_yaw'])
    robot.hold(1.0)
