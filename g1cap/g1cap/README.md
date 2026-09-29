# Implementation reading guide

## Current `sensor_state_v2` flow

Read this section first for the refreshed source. `interactive.py` preserves one physical episode across programs, while `workspace_agent.py` provides the bounded coding workspace and the [v2 tool contract](tool_docs/arena_v2.md). `arena_session_runtime.py` routes requests to one physics owner. In `arena_world.py`, `sensor_state.py` combines encoder/IMU odometry with camera-time box and table estimates from `box_tracker.py`; `control_v2.py` dispatches bounded grasp, walk, approach and place skills. `arena_public.py` filters the agent view, and `arena_task.py` scores separately with simulator truth.

V2 positions are estimated in an odometry frame; table patches are incomplete camera observations. Its reported map-motion uncertainty does not cover the full box or table extent. The present code is a simulation development candidate, with pickup reliability and full sensor-based transfer still open.

For SONIC, follow `session_runtime.py`, `session_tools.py`, `sonic_runtime.py`, then `toolkit/`. Workstation, ordered-reach and mobility programs preserve an episode across revisions. SONIC observations include explicitly labeled simulator values; they are not a deployed sensor interface.

## Earlier tracks

Start at the [project README](../README.md) for capabilities and commands. The original sharing cleanup reorganized delivery and metadata; this refresh also includes newer CAP implementation and tests.

## Read in this order

1. [interactive.py](interactive.py): launch a remote episode, request or load a program, submit it, collect feedback and revise without resetting physics.
2. [workspace_agent.py](workspace_agent.py): bounded Codex workspace, model settings, contract files and optional image input. Generated programs define `run(robot, task)`.
3. [arena_session.py](arena_session.py) and [arena_session_runtime.py](arena_session_runtime.py): restricted worker requests are queued to the sole physics-owning thread. Deadlines survive individual tool waits; a revision does not restart the episode.
4. [arena_control.py](arena_control.py): admission and dispatch of one active request, retaining actual command targets across tool boundaries.
5. [arena_world.py](arena_world.py): explicit wiring for native simulation, GR00T/HOMIE, sensor callbacks, wrist control and recording. This is the native-dependency boundary.
6. [arena_public.py](arena_public.py) and [arena_task.py](arena_task.py): keep public observations separate from independent ground-truth evaluation.

## Sensor components

| Responsibility | Files |
| --- | --- |
| Timestamped body, hand and camera packets | [arena_sensors.py](arena_sensors.py), [hand_sensors.py](hand_sensors.py), [arena_sensor_recording.py](arena_sensor_recording.py) |
| Encoder kinematics and attitude | [sensor_kinematics.py](sensor_kinematics.py), [sensor_attitude.py](sensor_attitude.py), [assets](assets/README.md) |
| RGB-D geometry and object estimates | [arena_perception.py](arena_perception.py), [rgbd_box_tracking.py](rgbd_box_tracking.py), [visual_box_state.py](visual_box_state.py) |
| Observed clearance and validity | [observed_approach.py](observed_approach.py), [observed_hand.py](observed_hand.py), [sensor_guard.py](sensor_guard.py), [source_plane.py](source_plane.py) |
| Shared paired wrist owner | [scene_wrist_hold.py](scene_wrist_hold.py), [carry_frame.py](carry_frame.py), [arm_gravity.py](arm_gravity.py) |
| Bounded raise, retreat, turn and stop | [sensor_lift.py](sensor_lift.py), [sensor_retreat.py](sensor_retreat.py), [sensor_turn.py](sensor_turn.py), [loaded_stop.py](loaded_stop.py) |
| Destination observation primitives | [destination_surface.py](destination_surface.py), [sensor_destination_approach.py](sensor_destination_approach.py) |

Destination helpers do not imply completed sensor-based transport or placement. Consult the [sensor contract](tool_docs/arena_sensor.md) for the public tool boundary. Privileged placement is separately implemented in `arena_placement.py`, `arena_alignment.py` and `arena_retraction.py`.

## Separate SONIC path

Read [session_runtime.py](session_runtime.py), [session_tools.py](session_tools.py), [sonic_runtime.py](sonic_runtime.py), then [toolkit](toolkit/README.md). `__main__.py`, `runner.py` and `pipeline.py` also retain historical single-turn mock/SONIC workflows. Their fresh-episode iteration semantics must not be confused with `interactive.py`.

The [walkthrough](../docs/code-walkthrough.md) explains ownership, observation assumptions and scoring. Historical videos and benchmark solutions belong in [human-facing evidence](../evidence/README.md), not tool documentation.
