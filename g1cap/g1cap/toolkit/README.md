# Robot toolkit

Read the [main package guide](../README.md) first. `api.py` and `robot.py` define the legacy bounded Robot facade; `sonic_backend.py` handles measured state, frozen episode-frame conversion and leased planner messages. `stationary.py` and `arm_planner.py` implement trusted posture/reach helpers. Compatibility shims remain at `g1cap.robot`, `g1cap.sonic_backend` and `g1cap.sonic`.

Persistent SONIC sessions use the [session tool contracts](../tool_docs/README.md), including MuJoCo world-frame goals. The older worker uses a frozen episode frame and a smaller API; its real SONIC `reach_right` is unsupported. Do not infer method availability from the shared name alone.

Arena uses `loaded_wrists.py`, `hand_clearance.py`, `acquisition_wrists.py` and geometry helpers for bounded manipulation. Read the [sensor contract](../tool_docs/arena_sensor.md) or [privileged contract](../tool_docs/arena_box.md) for the selected mode. These adapters depend on upstream control and simulation; they do not train a new controller or teleport the robot.
