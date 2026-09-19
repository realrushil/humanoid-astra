"""Handwritten nominal witness for the upper-marker workstation task.

NOT supplied to the coding agent. The standing-point heuristic comes from the
18-case offline scan in runs/task-first-001; no traversability claim is made.
Coordinates are MuJoCo world metres, +X toward this fixed workstation layout.
"""


def run(robot, task):
    scene = robot.observe_scene()
    marker = next(m for m in scene['markers'] if m['name'] == task['target_id'])
    target = marker['position_world']
    # An upright hypothetical stance planned successfully in the offline model.
    # Actual walking, settling and arm tracking must still be demonstrated.
    stance = [target[0] - .22, target[1] + .24]
    result = robot.walk_to(stance, timeout=15.)
    print('walk_to:', result['status'], result['reason'])
    if robot.observe()['episode_status'] != 'running':
        return
    result = robot.stop(timeout=8.)
    print('stop:', result['status'], result['reason'])
    state = robot.observe()
    if state['episode_status'] != 'running' or state['settled_for'] < .5:
        print('Cannot begin reaching: episode ended or base not settled.')
        return
    # Recheck the actual stance even after a walk timeout; no assumed arrival.
    plan = robot.check_reach(target)
    print('check_reach:', plan['status'], plan['reason'])
    if plan['status'] != 'planned':
        return
    result = robot.reach_right(target, timeout=12.)
    print('reach_right:', result['status'], result['reason'])
    if result['status'] == 'completed' and robot.observe()['episode_status'] == 'running':
        robot.hold(1.)
