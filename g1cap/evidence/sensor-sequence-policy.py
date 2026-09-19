"""Developer qualification of the ordinary tool sequence; no recorded actions."""
def run(robot, task):
    result = robot.wait(duration=0.3)
    print('initialize', result['status'], result['reason'])
    result = robot.pickup_box('brown_box')
    print('pickup', result['status'], result['reason'])
    if result['status'] == 'completed':
        result = robot.raise_held_box(clearance_m=0.08)
        print('clearance', result['status'], result['reason'])
    if result['status'] == 'completed':
        result = robot.hold_box(duration=3.0)
        print('initial hold', result['status'], result['reason'])
    if result['status'] == 'completed':
        result = robot.retreat_with_box(distance_m=0.20)
        print('retreat', result['status'], result['reason'])
    if result['status'] == 'completed':
        result = robot.hold_box(duration=2.0)
        print('pre-turn hold', result['status'], result['reason'])
    if result['status'] == 'completed':
        result = robot.turn_with_box(yaw_rad=-0.5235987755982988)
        print('relative turn', result['status'], result['reason'])
    if result['status'] == 'completed':
        result = robot.hold_box(duration=2.0)
        print('post-turn hold', result['status'], result['reason'])
    # Record a bounded post-outcome interval; elapsed time is not a success test.
    result = robot.wait(duration=5.0)
    print('post-outcome observation', result['status'], result['reason'])
    print('final permitted observations', robot.observe())
