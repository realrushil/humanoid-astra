"""Single-goal reference with one bounded recovery for walk_to / no_progress.

Decisions use public observations, never privileged scoring. A stop can change
position, so reobserve before deciding whether another walk is needed.
"""
import math


def run(robot, task):
    goal = task['target_position'][:2]
    result = robot.walk_to(goal)
    if result['status'] != 'completed':
        if result['status'] != 'timed_out' or result['reason'] != 'no_progress':
            return
        print('recovery: stopping after no_progress')
        result = robot.stop()
        if result['status'] != 'completed':
            return
        state = robot.observe()
        if state['episode_status'] != 'running':
            return
        error = math.dist(state['pelvis_position'][:2], goal)
        print('recovery: observed position error (m)', error)
        if error > task['constraints']['position_tolerance']:
            print('recovery: retrying walk once')
            result = robot.walk_to(goal)
            if result['status'] != 'completed':
                return
        else:
            print('recovery: position within tolerance; setting final heading')

    result = robot.turn_to(task['target_yaw'])
    if result['status'] != 'completed':
        return
    robot.hold(1.)
