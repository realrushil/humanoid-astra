# Execution and observation boundaries

## Current Arena `sensor_state_v2` path

The persistent session described below also carries the current v2 path. With `control_track: sensor_state_v2`, each 20 ms `ArenaWorld.step()` reads encoder/IMU state; every fifth step it consumes an RGB-D frame. `SensorState` estimates stance-foot/gyro odometry, `BoxTracker` estimates the box, and the table map retains observed patches in the estimated odometry frame. These are simulator-derived measurements, distinct from truth used by `arena_task.py` for scoring.

`ControlV2.command()` sends the active bounded skill's action to the sole native controller path; `ControlV2.update()` checks completion and hard stops against the next estimated observation. Agent-visible data passes through `arena_public.py` and the [v2 tool contract](../g1cap/tool_docs/arena_v2.md). A walking completion, grasp encoder lag or `at_surface` report is not proof of a supported carried object or a fully contained placement. The current table patch can be clipped; its reported motion uncertainty omits box and map-extent error. The v2 path remains a development candidate for simulation.

The older sensor path described later in this document uses different callbacks and assumptions. Its historical sequence video is not evidence that v2 completes a full transfer.

## One program revision

`interactive.py` selects Arena from `recipe["backend"] == "arena"`; otherwise it uses SONIC. `RemoteSession` creates a unique remote run. Arena uploads the exact local package into an isolated source overlay, then starts `arena_launch.py`. SONIC expects matching code already installed in its remote root.

`workspace_agent.py` gives Codex documented tool contracts and a task, and obtains `policy.py`. The program defines `run(robot, task)`. With `--policy`, the same submission path loads a developer-written file without model generation.

The Arena session starts a restricted Python worker through `execution.py`. Tool requests cross a queue; only the world thread touches Isaac and native controllers. `arena_control.py` dispatches bounded operations. `arena_world.py` owns dependencies and the final actuator path. GR00T proposes acquisition actions; HOMIE handles body control; paired wrist corrections preserve retained targets through supported tool transitions.

A returned tool result is feedback about that operation. `arena_task.py` independently evaluates physical task progress and faults. Previous failed requests, moved objects and elapsed time remain part of the episode when the agent edits. Terminal failures do not trigger an automatic fresh episode.

## Sensor path

The sensor recipe explicitly enables `record_sensors`, `sensor_balance`, `visual_grasp_checks` and optional `arm_gravity_compensation`. Recording alone does not make a controller sensor-based. `rgbd_period_steps: 2` requests 25 Hz delivery from a 50 Hz outer loop; body encoder/IMU updates are at 50 Hz. Native contact recording is evaluator-side.

Permitted measurements include 29 body joint positions/velocities/efforts, pelvis IMU, calibrated onboard RGB-D and fourteen Dex3 position channels. Effort is an ideal simulated actuator-output proxy. No finger torque, tactile array or wrist force/torque sensor is supplied. Static calibration is bundled under `g1cap/assets/`.

Kinematics and RGB-D yield local estimates. Color-based association is a prototype, not general recognition. Floor/heading control and local stance segments are not external world localization. Source geometry uses a floor-parallel table assumption and conditional uncertainty; missing/stale or ambiguous geometry remains unavailable. Typical freshness checks reject geometry older than 150 ms.

Sensor publication hides true mass, fixture identifiers and true table geometry. Physics truth is retained for recording/scoring. Synchronous simulation assumes delivery timing that the current frontend cannot meet in real time: the included successful recording used about 953 wall seconds for 32.60 simulated seconds. A nominal video is not hardware validation.

## Scoring and artifacts

`selected_table_transfer_v2` requires stable lift, at least 0.5 m box travel and stable contained support on the selected table with both hands released. Collision, floor-contact, tilt and missing-evidence failures are preserved. The included sensor witness stops after hold, so it does not satisfy this full-transfer objective.

`arena_video.py` labels and combines recorded camera frames and tool/program events. It validates provenance and frame counts without new physics. `session/` retains programs and feedback, while `physics/` retains evaluator state, actions, sensors and camera recordings. Do not pass private evaluator scores or human-facing successful programs into benchmark agent context.
