"""Handwritten development reference, not supplied to benchmark agents.

Assumes the fixed workstation layout is approached approximately along world +X,
with room for an upright G1 to its front. No object contact or grasping.
"""
import math


def run(robot, task):
    scene = robot.observe_scene()
    target = next(m['position_world'] for m in scene['markers']
                  if m['name'] == task['target_id'])
    state = robot.observe()
    # Retain 15 cm of forward wrist travel for the local reach. This is a
    # reference-policy heuristic, not an ideal stance provided by the task.
    base = [state['pelvis_position'][0] + target[0] - state['right_wrist_position'][0] - .15,
            state['pelvis_position'][1] + target[1] - state['right_wrist_position'][1]]
    for _ in range(2):
        if robot.observe()['episode_status'] != 'running':
            return
        result = robot.walk_to(base, timeout=15.)
        print('walk', result['status'], result['reason'])
        robot.stop(timeout=8.)
        state = robot.observe()
        if state['episode_status'] != 'running':
            return
        if state['settled_for'] >= .5 and math.dist(state['right_wrist_position'], target) <= .25:
            break
    state = robot.observe()
    if state['settled_for'] < .5:
        print('Base did not settle; return for a revision without erasing motion.')
        return
    plan = robot.check_reach(target)
    print('reach check', plan['status'], plan['reason'])
    if plan['status'] != 'planned':
        return
    result = robot.reach_right(target, timeout=12.)
    print('reach', result['status'], result['reason'])
    if robot.observe()['episode_status'] == 'running':
        robot.hold(1.)
