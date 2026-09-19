"""The public robot API contract supplied to coding agents."""

def api_documentation(backend='mock'):
    document = '''Write Python defining run(robot, task). The task is a dictionary with
name, target_position [x,y,z], target_yaw (radians), and public constraints.
For name='route', task['waypoints'] contains ordered [x,y] targets. Visit every
target in order and settle there for 0.5s; only the final target requires target_yaw.
For single-waypoint tasks, waypoints is empty: use target_position[:2].
Use math if needed. Robot tools return dictionaries, never an automatic task reward.
All positions are in a fixed episode frame. observe() includes pelvis_position,
pelvis_yaw, planar_velocity, yaw_rate, tilt, right_wrist_position, sim_time,
episode_status. Read task["target_position"][:2] for the walking goal.
Tools: observe(); move_base(vx,vy,yaw_rate,duration) with duration (0,2], speed norm
<=0.30, |vy|<=0.20, |yaw_rate|<=0.40; stop(timeout=None); hold(duration) in (0,2];
walk_to(xy,timeout=None); turn_to(yaw,timeout=None); reach_right(position,timeout=None).
Velocities use body coordinates: +vx forward, +vy left, yaw rate in radians/s.
turn_to uses an absolute episode heading; shortest-angle change must be <=pi/2.
move_base completed means its duration elapsed, not that a distance or speed was
achieved. stop requests idle and waits for settling; hold does not fix base pose.
Omit timeout to use a bounded default capped by remaining task time. Explicit
timeouts must not exceed task['deadline'] minus observe()['sim_time'].
Motion result status is one of: completed, rejected, timed_out, cancelled,
backend_error. The reason field explains the outcome. A cancelled result with
reason='success' means the online evaluator considers the task complete; return
without further motion. With estimated observations, independent true scoring can disagree.
observe() returns the state dictionary directly, not a motion result.
walk_to drives near a goal and settles, turn_to rotates to an absolute episode
yaw, reach_right moves the wrist toward a nearby position while standing.
Use tools sequentially. Inspect status and observation if needed. Task A requires
position, final heading and settling; Task B requires wrist arrival and stance.
Task completion requires holding the goal briefly; hold(1.0) can supply dwell.
No files, network, reset, arbitrary simulator access or extra libraries are available.
This run uses a synthetic mock: write meaningful task logic, but do not claim physical performance.
Return source code as the source field of the requested JSON schema.'''
    if backend == 'sonic':
        document = document.replace('This run uses a synthetic mock: write meaningful task logic, but do not claim physical performance.',
            'This run uses a free-standing MuJoCo G1 29 DOF with Dex3 and SONIC neural locomotion. '
            'Tools are experimental: goals can fail, drift or time out. Observe measured state and inspect results. '
            'Nominal mode observes simulator ground truth. With an explicitly configured observation adapter, '
            'tools and online completion use its estimates while separate offline scoring retains truth. '
            'Read the supplied observation configuration; do not assume true scoring state is available to the policy. '
            'These simulations do not establish hardware qualification. '
            'reach_right is unavailable and will return rejected/unsupported_operation. '
            'The task is waypoint arrival plus heading and settling. No teleportation or direct joint control exists.')
        document = document.replace('reach_right moves the wrist toward a nearby position while standing.',
                                    'reach_right is unsupported in this locomotion-only backend.')
    return document
