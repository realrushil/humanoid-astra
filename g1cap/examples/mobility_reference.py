"""Handwritten capability witness, not agent-generated code. World metres/radians."""
import math


def run(robot, task):
    # Bounded actions, current progress: safe to submit again in the same episode.
    for _ in range(6):
        state=robot.observe()
        if state['episode_status']!='running': return
        index=state['task_progress']['waypoint_index']
        goal=task['waypoints'][index]
        if math.dist(state['pelvis_position'][:2],goal)>.07:
            print('walk',index,robot.walk_to(goal,timeout=20.))
        else:
            yaw=task.get('final_yaw')
            if index==len(task['waypoints'])-1 and yaw is not None:
                error=(yaw-state['pelvis_yaw']+math.pi)%(2*math.pi)-math.pi
                if abs(error)>.1:
                    step=max(-1.4,min(1.4,error))
                    print('turn',robot.turn_to(state['pelvis_yaw']+step,timeout=12.))
            if robot.observe()['episode_status']=='running':
                print('settle',robot.stop(timeout=5.))
