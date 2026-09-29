# Robot toolkit

This folder contains our robot-facing APIs and SONIC adapters. Start with `api.py` for the contract the coding agent sees, then read `Robot.walk_to`, `Robot.turn_to` and `Robot._motion` in `robot.py`.

| File | Responsibility |
| --- | --- |
| `api.py` | Maintained API documentation supplied to coding agents |
| `robot.py` | Public `Robot` facade, bounded motion loops and explicit synthetic mock |
| `sonic_backend.py` | Measured state, episode-frame conversion and leased SONIC planner messages |
| `sonic.py` | Offline planner-field conversion helper; not the live controller |
| `stationary.py` | Additive trusted posture/reach executor for component qualification |
| `arm_planner.py` | Position-only local right-arm IK, named SONIC mapping and bounded local collision correction |

`from g1cap.toolkit import Robot` is the canonical import for harness code. Generated policies receive a `robot` proxy from the worker and do not import this package.

Worker tools are `observe`, `move_base`, `walk_to`, `turn_to`, `stop`, `hold` and `reach_right`. The existing SONIC worker does not support `reach_right`; it returns an explicit rejection. Positions are metres in the frozen episode frame; headings are radians. Velocity commands use body coordinates, positive forward/left. Turning and holding can displace the base; `completed` is a tool outcome, not a guarantee that the entire task is complete.

The separate `StationaryTools` class now supplies real SONIC posture/reach operations to the trusted qualification harness. It uses **MuJoCo world coordinates**, not the worker's episode frame. Read the [stationary contract](../../docs/stationary-toolkit.md) before using it. The local planner tries bounded signed-distance corrections before rejecting a colliding path; it can still fail on feasible targets. See the [follow-up validation](../../runs/collision-planner-001/report.md). No synthetic reach fallback is used.

SONIC's neural controller and MuJoCo dynamics remain upstream in `references/GR00T-WholeBodyControl/`. Our wrappers issue commands and measure outcomes; they do not teleport the robot or train a new controller. Current nominal observation access uses selected simulator state and is privileged. It does not establish hardware localization or tool qualification.

Task definitions, scoring and episode execution remain outside this folder in `models.py`, `runner.py` and `execution.py`. Historical imports at `g1cap.robot`, `g1cap.sonic_backend` and `g1cap.sonic` are compatibility shims. New code should use this folder.

See the [package reading guide](../README.md) for the full flow and [tool contracts](../../docs/tool-contracts.md) for operational limits and evidence.


The separate Arena box workflow uses `loaded_wrists.py` for measured loaded manipulation. During pickup, `hand_clearance.py` estimates hand/table approach and `acquisition_wrists.py` applies a bounded common upward correction to native neural wrist targets. These use current geometry, not stored pickup trajectories. Read the [Arena tool contract](../tool_docs/arena_box.md) and [package guide](../README.md) for ownership, limits and evidence; the SONIC worker API above is a separate backend.
