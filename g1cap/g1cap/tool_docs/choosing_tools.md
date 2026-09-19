# Choose and compose tools

Use this guide before writing a tool sequence, then read the selected tools' contracts for exact arguments and limits. The available Python methods are listed in [the catalog](README.md). A controller accepting a command does not establish that the robot can achieve it from the current state.

## Choose by the physical outcome

| Intended outcome | Current tools | What the program must supply or check |
| --- | --- | --- |
| Arrive at a base location | `walk_to`, optionally `turn_to` | A clear route inside the task workspace; current position, heading and task progress. The tool does not plan around furniture. |
| Stand at a different height | `set_posture`, `hold` | Reachable height and measured settling; this changes the arm reference context. |
| Put one wrist at a point | `check_reach`, `reach_right` | A nearby collision-free world point. This interface does not control wrist orientation. |
| Put both wrists at oriented poses | `check_hands`, `reach_hands` | Two nearby world wrist poses, current object geometry and a settled stance. Contact is forbidden. |
| Request coordinated body/arm motion over time | `sonic_motion` | Its documented conditioning sequence and an independently checked physical objective. Completion means the command interval ended. |
| Pick up, carry or release an object | No qualified public tool | Report the missing capability. Reaching, finger posture and motion duration do not establish a grasp or load support. |

`check_reach` and `check_hands` can reject geometry before an action, but a `planned` result is neither execution nor a reusable trajectory handle. The corresponding reach tool plans again from measured state. During both calls, the physical world keeps evolving.

## Check whether a sequence preserves its assumptions

| After this operation | Reference behavior | Consequence for the next operation |
| --- | --- | --- |
| Successful stationary reach | Supervised arm reference remains active | Use `hold` to assess continued settling; reobserve before changing the goal. |
| `walk_to`, `turn_to`, `move_base` or a posture change | Previous stationary arm targets are released | Obtain fresh wrist/object measurements and plan the next reach. Do not assume a carried pose persists. |
| `stop` with a valid stationary reference | Checks settling while preserving that reference on success | It is not a command to open fingers or a hardware emergency stop. |
| `sonic_motion` exits | Its motion references are released | A later `hold` cannot preserve the trajectory's final pose automatically. |
| Executed action fails or times out | Earlier motion remains; references may be released | Read current state before deciding on a different bounded attempt. |

Choose an approach stance before a stationary reach. Calling a navigation tool after reaching invalidates the arm-hold assumption. If the task requires simultaneous walking and object support, this toolkit currently lacks the necessary controller contract.

## Context to resolve before issuing a command

1. **Live state:** read `observe()` and, when objects matter, `observe_scene()`. Check `episode_status`; stop issuing actions when it is no longer `running`. Initial scene box coordinates are not current free-object poses.
2. **Frames:** distinguish body velocity from world position, and wrist origins from palm/fingertip contact surfaces. Use metres, radians and WXYZ quaternions as each contract specifies.
3. **Entry conditions:** check workspace, local reach bounds, settling and clearance. Accepted input limits describe interface bounds, not proven reliability throughout that range.
4. **Exit conditions:** decide what measurement would demonstrate progress. Tool completion and independent task completion are different predicates.
5. **Continuation:** account for elapsed time, changed object poses and released references. A revised program inherits all previous consequences.

## Common reasoning errors

- **“IK succeeded, so the reach will succeed.”** IK checks a local geometric model; neural tracking and balance must still succeed during execution.
- **“The wrists touched the box, so it is held.”** Contact and bilateral wrist pose do not establish sufficient support, retention or controlled release. `holding='paired_pose_reference'` refers to a controller reference.
- **“The motion returned completed, so the robot arrived.”** For a timed command, inspect measured displacement and task progress; do not infer a spatial outcome from elapsed time.
- **“Retry the same request after a timeout.”** First inspect remaining error and episode state. Bound retries and change the plan when measurements show no progress.

Keep logs brief: intended physical outcome, tool arguments, returned status/reason and the relevant measured change. Do not invent callable tools from papers, task wording or examples. Additional libraries must first be integrated into the backend and documented as an available public API.
