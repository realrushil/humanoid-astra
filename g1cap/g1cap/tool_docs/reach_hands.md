# reach_hands(goals, timeout=20)

Attempt two wrist poses together while standing. This is an ordinary Python tool using SONIC, not a grasp controller. Development capability: physical coverage is linked from the package guide; accepted input bounds are not a reliability guarantee.

`goals` is a dictionary with exactly `left` and `right`. Each value contains `position: [x,y,z]` in MuJoCo world metres (Z up), and `quaternion: [w,x,y,z]`, a unit body-to-world quaternion. The controlled frame is each `wrist_yaw_link` body origin, **not the palm or fingertip**. Transform object-relative offsets using the object's current orientation before making world goals. Read current free-object poses from `observe_scene()['objects']`; scene box centres are initial poses.

Requires at least 0.5 s measured two-foot settling. Each target must be within 0.4 m of its measured wrist. Timeout is simulation seconds in (0,20], with a separate execution wall guard. Local IK freezes measured root, waist, fingers and free-object poses, checks joint limits and sampled declared collisions, then requests SONIC tracking with each reference joint bounded to 0.25 rad/s. No intentional hand–object contact is permitted.

Up to two feedback replans are attempted after the commanded path has finished and 3 s of tracking time has not achieved the goal. Planning runs in one worker while observation and command supervision continue. Large root/finger changes, object motion (>1 cm or 0.10 rad), stale/gapped evidence, contact or support faults stop execution. Collision checks are local and sampled; a plan does not establish dynamic feasibility.

`completed` requires both wrists within 2 cm and 0.15 rad, both wrist speeds ≤0.05 m/s, stable two-foot stance, base speed ≤0.05 m/s, yaw rate ≤0.10 rad/s and tilt ≤15°, together for 0.5 s. Result includes `status`, `reason`, `elapsed`, `observation`, `hand_errors`, `replans` and `holding` for executed actions. This tool result is not benchmark success.

Successful execution leaves a supervised joint reference during editing while the episode remains active. Episode termination cancels an in-flight call and releases references, including when the independent evaluator reaches success. `hold()` checks the retained paired pose as well as stance; the supervisor maintains references subject to fault/configuration checks, but does not independently repair pose error after this tool ends. `stop()` preserves a valid stationary hold. Navigation and posture changes release arm references. Failed/timed-out execution returns toward nominal idle and preserves all earlier physical consequences. A pre-execution planning rejection applies no new arm reference and leaves physics running. A rejected replan after motion is an execution failure, not an untouched world.

`holding='paired_pose_reference'` does not mean holding an object. Fingers retain their existing posture profile. Do not use this tool to claim contact, grasping, lift, release or carrying.
