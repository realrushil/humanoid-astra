# Robot tools

These contracts describe the persistent `run(robot, task)` session. All tools share one world and clock. Code editing and computation consume physical time. Do not reset or replay earlier actions to repair a program. Simulator-derived observations are privileged; they are not camera measurements or calibrated hardware estimates. `completed` is the tool's own predicate, not task success. Consult `episode_status` after actions. Model capability and tracking are state-dependent.

Read [the programming guide](programming.md) and [tool selection and composition](choosing_tools.md), then the document for each tool you intend to use. The controller backend is SONIC; interface availability is distinct from physical qualification. These are ordinary Python methods, including `sonic_motion`. There is no special execution tier.

| Tool | Purpose | Completion meaning |
| --- | --- | --- |
| [observe](observe.md) | Read the latest robot state | Snapshot, not an action |
| [observe_scene](observe_scene.md) | Read named geometry/current free objects | Simulator snapshot, not perception |
| [walk_to](walk_to.md) | Approach a world XY destination | Position tolerance and settling |
| [turn_to](turn_to.md) | Change facing | Yaw tolerance and settling |
| [move_base](move_base.md) | Request body-frame velocity | Requested interval elapsed |
| [stop](stop.md) | Request idle or maintain stationary reference | Measured settling; context-dependent |
| [hold](hold.md) | Maintain stationary references | Duration plus settling |
| [set_posture](set_posture.md) | Request SONIC height-conditioned posture | Measured height and settling |
| [check_reach](check_reach.md) | Assess local right-arm IK | Plan found, not physical execution |
| [reach_right](reach_right.md) | Plan and attempt a stationary wrist reach | Measured wrist position/speed and settling |
| [check_hands](check_hands.md) | Read-only paired wrist pose IK | Geometry plan, not execution |
| [reach_hands](reach_hands.md) | Coordinated position/orientation tracking with feedback | Both measured wrist poses and stance dwell; no contact |
| [sonic_motion](sonic_motion.md) | Supply a short coordinated SONIC command sequence | Command duration only |

## Controller assumptions and interpreting results

Named `hand_posture`/`arm_posture` observations are command profiles, not measurements. `upstream_default` omits targets; upstream can supply curled fingers and its own arm motion. `tucked_thumb` supplies fixed seven-joint hand targets; `travel` supplies nominal outward arms/bent elbows. Consult measured joint arrays to assess tracking. A stationary or motion tool can override nominal upper-body references; idle restores the selected nominal profile. None of these profiles is a grasp controller.

Common action statuses: `completed`, `timed_out`, `rejected`, `failed`, `cancelled`. `check_reach` additionally returns `planned`. Invalid arguments are rejected before the request is applied. A rejected planning request leaves physics running. A timed-out/failed action may already have moved the robot and can release its references. Terminal episode faults cannot be repaired in that episode. `observe()` remains available.

Success signals for deployment would need real sensing. This baseline exposes simulator state and simulator terminal outcomes explicitly. No hardware or reliable whole-task capability follows from an accepted input range. No arbitrary task objective, obstacle avoidance, grasping or loaded carrying is supplied by `sonic_motion`.

API/module/document snapshots are hashed for each agent round. This catalog contains general contract knowledge, not task-specific successful base poses or benchmark solutions. Physical evidence and videos are kept in the human-facing project report rather than made available as hidden benchmark answers.
