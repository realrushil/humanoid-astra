"""Experimental single-goal navigation using only turning and forward segments.

Positions/headings are observations in the fixed episode frame. Distances are
metres, angles radians. The 7 cm stopping allowance is a heuristic from short
SONIC probes, not a hardware bound. Reobserve after every stop and final turn.
"""
import math


def face(robot, target_yaw):
    """Split large heading changes to respect turn_to's 90-degree limit."""
    for _ in range(5):
        state = robot.observe()
        if state['episode_status'] != 'running':
            return False
        yaw = state['pelvis_yaw']
        error = (target_yaw-yaw+math.pi) % (2*math.pi)-math.pi
        if abs(error) <= .08:
            return True
        # 60 degrees leaves room for motion between observing and calling.
        step = max(-math.pi/3, min(math.pi/3, error))
        result = robot.turn_to(yaw+step)
        if result['status'] != 'completed':
            return False
    return False


def run(robot, task):
    goal = task['target_position'][:2]
    for _ in range(12):
        state = robot.observe()
        if state['episode_status'] != 'running':
            return
        x, y = state['pelvis_position'][:2]
        dx, dy = goal[0]-x, goal[1]-y
        distance = math.hypot(dx, dy)
        if distance <= .07:
            if not face(robot, task['target_yaw']):
                return
            result = robot.hold(.6)
            if result['status'] != 'completed':
                return
            # Turning may move the base: check position again next iteration.
            continue

        if not face(robot, math.atan2(dy, dx)):
            return
        state = robot.observe()
        distance = math.dist(state['pelvis_position'][:2], goal)
        if distance <= .07:
            continue
        duration = max(.5, min(1.5, (distance-.07)/.20))
        result = robot.move_base(.20, 0., 0., duration)
        if result['status'] != 'completed':
            return
        result = robot.stop()
        if result['status'] != 'completed':
            return
    # A bounded attempt is not a success claim; the evaluator owns that result.
    robot.stop()
