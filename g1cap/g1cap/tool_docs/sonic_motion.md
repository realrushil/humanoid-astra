# sonic_motion

`robot.sonic_motion(segments)` is an ordinary toolkit method. It executes a short SONIC command sequence in the current physical episode. It does not accept an arbitrary cost function, generate code, solve a route, or establish task success.

## Inputs and meaning

`segments` is a list of 1–8 dictionaries, each lasting 0.2–2 simulation seconds, total at most 8 seconds. Only these fields are accepted:

| Field | Meaning and bound |
| --- | --- |
| `mode` | Required: `idle`, `walk`, or `squat`; upstream modes 0, 1, 4 respectively |
| `duration` | Required simulation seconds; physics continues throughout |
| `velocity_world` | Optional world XY m/s, default `[0,0]`; norm <=0.20; nonzero only in walk mode. This differs from move_base's BODY velocity |
| `facing_world` | Optional world yaw radians; defaults to the preceding requested facing or measured entry yaw. Within +/-0.4 rad of entry, interpolated reference rate <=0.4 rad/s |
| `height` | Required in squat mode: [0.60,0.85] m height conditioning. Otherwise absent or -1. Not a guarantee of measured pelvis height |
| `right_arm` | Optional nonempty dictionary of absolute named joint radians. Any subset of `right_shoulder_pitch_joint`, `right_shoulder_roll_joint`, `right_shoulder_yaw_joint`, `right_elbow_joint`, `right_wrist_roll_joint`, `right_wrist_pitch_joint`, `right_wrist_yaw_joint` |

Arm references must satisfy model joint limits, stay within +/-0.20 rad of the measured entry joints, and interpolate no faster than 0.35 rad/s. Unspecified joints retain their preceding reference; before the first arm request they start from the measured entry state. Once an arm request is present, measured-entry waist/left-arm references fill the 17-joint SONIC message and remain fixed through the sequence. Before that, nominal profile/upstream arm behavior applies. These are reference constraints, not measured joint-rate guarantees. The adapter interpolates position references and sends zero joint-velocity reference fields, matching the existing stationary adapter. Fingers keep the trusted hand profile. No finger target field is exposed.

## Preconditions and execution

The whole sequence is validated before any old reference is released. It requires a running, healthy episode and at least 0.5 s of measured base settling. Arm requests additionally require model joint limits. Invalid arguments are rejected/invalid_arguments; unsettled entry is rejected/base_not_settled. A rejected request does not release an existing hold.

Accepted execution releases previous stationary posture/arm targets, then refreshes timed motion fields. The same session supervisor publishes them; motion refresh expires after 150 ms, with the publisher's independent 250 ms lease as a second boundary. Reference loss ends the attempt; it is not silently resumed. Global contact, posture, evidence and time guards remain active. There is no destination controller, IK, collision-path prediction or automatic grasp maintenance. In particular, initial joint validity does not establish future body/scene clearance while walking.

## Result and physical consequences

`completed` with `reason="duration_elapsed"` and `completion="command_duration_only"` means the command interval finished. It does **not** mean the robot tracked its velocity, reached its facing/height/arm reference, settled, or accomplished the task. Inspect the returned observation and fresh state; task scoring is independent.

Terminal faults return cancelled with the episode reason. Lost reference returns failed/motion_reference_lost. Wall overrun returns timed_out/wall_timeout; a sequence whose interval passed before its first command returns timed_out/sequence_not_started. All accepted exits request nominal idle and release motion arm/posture targets. The robot can drift, rise, or return its arms toward its configured nominal pose. Physics, contact consequences and elapsed time remain. No motion reference persists during subsequent code editing. Use stop/observe before choosing another action; do not assume hold() preserves this sequence's last pose.

## Example and evidence scope

A minimal command-composition example, not a task solution:

```python
state = robot.observe()
if state['settled_for'] >= .5:
    result = robot.sonic_motion([
        {'mode': 'walk', 'duration': 1., 'velocity_world': [.08, 0.],
         'facing_world': state['pelvis_yaw'] + .10},
        {'mode': 'idle', 'duration': 1.},
    ])
    if robot.observe()['episode_status'] == 'running':
        robot.stop()
```

This first interface exposes a restricted subset of SONIC's protocol. Arbitrary modes, VR targets, full motion clips, finger control and unconstrained joint trajectories are not implemented. Bounds are uncalibrated development assumptions. In particular, the low-speed walking range can yield negligible physical progress even when the generated root reference advances. Constant world-direction conditioning is not behaviorally interchangeable with BODY-velocity commands: the latter can trigger extra upstream replanning as measured yaw changes. Do not assume this sequence tracks a walking trajectory from its duration result. Validation must separate successful command transport from physical tracking and task success. No general contact-free motion or hardware qualification is claimed. See [shared assumptions](README.md).
