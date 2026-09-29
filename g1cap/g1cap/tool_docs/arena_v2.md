# G1 box transfer: robot API (sensor_state_v2)

Write `def run(robot, task):` in plain Python. The physical world persists
between your programs: whatever you did last time has already happened, and
time keeps running while you think.

## What the robot knows

Everything comes from the robot's own sensors: head RGB-D camera, joint
encoders, IMU and finger positions. There is no simulator ground truth.

`robot.observe()` returns:

- `robot`: `position_m` [x, y, z] and `yaw_rad` in the **odom frame**. The origin
  is the start pose, x points toward the initial facing direction, z is up.
  This comes from leg odometry and drifts by about 10–20% of the distance walked.
- `box`: `position_m`, `size_m`, `held` (the box is between the hands and off
  any table), `height_above_source_m`, and `age_s` since it was last seen.
  `box` is `None` if the box has never been seen.
- `tables`: for `source` (grey) and `destination` (green). Each is `None` until
  it has been seen, otherwise `min_xy_m`, `max_xy_m`, `top_z_m` and
  `position_uncertainty_m`. The extent covers only the part seen so far.
  This uncertainty is drift accumulated since the last table view; it does not
  include all box-tracking, depth or table-boundary errors. Zero does not prove
  that the whole box fits on the tabletop.
- `surfaces`: per seen table, `clearance_m` (box bottom above that tabletop) and
  `contained` (box footprint inside the seen tabletop).

Tables are only added to `tables` once the head camera has seen them. To find
a table, turn toward where it might be and observe again.

## Calls

Every call blocks until done and returns
`{"status": "completed" | "failed" | "rejected" | "cancelled", "reason": str, "advice": dict, "observation": dict}`.
Check `status` after every call. `advice` holds numbers useful for recovery,
such as `remaining_m`, `heading_error_rad`, `edge_distance_m` and
`carried_table_clearance_m`.

| Call | What it does |
|---|---|
| `robot.observe()` | Current estimate; no motion |
| `robot.wait(duration=1.0)` | Stand still, 0 < duration ≤ 5 s |
| `robot.grasp_box(lift_m=0.06)` | **Recommended pickup.** Scripted two-hand grasp from the measured box pose. It turns square to a rotated box, lines itself up with the box (stepping up to 0.40 m), raises and opens the hands wider than the box, steps in, lowers them beside it, squeezes until the arm encoders sense contact and lifts (up to 6 cm per call; it re-lifts if the box is held but low). While you then hold, walk, turn or approach, the grip is kept and tightened if it loosens. Farther than 0.40 m it refuses with `box_out_of_reach` and a suggested `walk`/`approach_surface("source")`. `hands_hit_box`: a hand touched the box before squeezing and the arms backed off; `observe()` and call again. `box_tipped_in_hands` / `grasp_not_stable`: the box did not come up level; check `advice` and `observe()` before retrying. Known limit: heavier boxes (about 0.5 kg) can turn in the hands while lifting; if that happens twice, try `pickup_box` |
| `robot.pickup_box("brown_box")` | Fallback: learned (GR00T) two-hand grasp. It walks itself into position and needs the source table seen. Observed pickup results are mixed across changed box poses and sizes; heavier-box starts can succeed but are not guaranteed. Adds a small lift if the box ends up low. Completes when the box is held ≥ 5 cm above the table for 1 s |
| `robot.raise_held_box(clearance_m=0.08)` | Lift the held box to 3–15 cm above the source top (up to 6 cm per call; the grip is kept) |
| `robot.hold_box(duration=1.0)` | Keep holding; completes after ≥ 1 s of a stable held box |
| `robot.walk(forward_m, left_m=0.0, yaw_rad=0.0)` | Walk to an odometry-frame goal set from the robot's current frame, 5 cm–1.5 m. Optional relative `yaw_rad` (up to ±π/2) changes heading during the walk; zero holds the starting heading. Negative `forward_m` walks backward. Works with or without the box |
| `robot.turn(yaw_rad)` | Turn in place by a relative angle, up to ±π. Positive is counter-clockwise (left) |
| `robot.approach_surface("destination", standoff_m=0.24)` | Square up to the nearest seen edge of that table, centre on it sideways, and walk until the camera sees the requested standoff (0.2–0.6 m). Carrying keeps the 0.12 m/s distant approach target and has a bounded 50 s plus mapped-distance/0.05 m/s allowance; unloaded approach has 30 s plus distance/0.05 m/s. `at_surface` does not verify that the held box fits on the table or can be released; recheck support geometry before placement |
| `robot.place_box("destination")` | Box must remain between the hands at least 4 cm inside the seen tabletop edges, and between about 1 cm below and 20 cm above that tabletop (it moves the box with the arms first when needed). A box already touching the tabletop may have `held=false` and still be eligible. Crouches until the box rests on the table, opens the fingers, withdraws the wrists, stands up and checks that the box stayed on the table with the hands clear |

## Safety behaviour you will see

These are the only automatic stops:
- **Body tilt above 15°:** the episode ends.
- **Carried box leaves the hands while walking or turning:** that call fails with `box_retention_lost`, and the robot stops.
- **Predicted contact with a seen table while walking or turning:** that call fails with `predicted_table_contact` if the motion brings the box or arms closer to the table. Re-observe and choose a path that does not continue into the predicted contact.

Table legs stand at the corners. `walk`, `turn` and `approach_surface` do not stop near them, but report
`nearest_table_corner_m` / `nearest_corner_m` (estimated; it can read 10-30 cm too close near the destination) in `advice`. Below about 0.22 m,
move away from the corner and approach the middle of an edge instead. Bumping a leg fails
the task.

Everything else is information, not a stop. A failed call leaves the robot
where it is, still holding the box if it was holding it. Decide what to do next
from `reason`, `advice` and a fresh `observe()`.

## Failure recovery

- `hands_hit_box`: the hands backed off before squeezing. Observe the box again and retry the grasp once. [Evidence](../../lessons/README.md#scripted-grasp-v2c-2026-09-22).
- `predicted_table_contact`: the commanded path stopped. Observe and change the path before retrying. Raising the box is limited to about 6 cm per call and may leave this stop unchanged; a different motion must be checked from the new pose. [Evidence](../../lessons/README.md#scripted-grasp-v2c-2026-09-22).
- `height_goal_not_reached`: check the achieved height in the returned observation. `raise_held_box` has a 3–15 cm goal range, so do not keep requesting an unreachable height. [Evidence](../../lessons/README.md#scripted-grasp-v2c-2026-09-22).
- `box_not_well_inside_surface`: use `advice.view_margins_m` and `advice.suggest`; move the box toward the side with more visible tabletop, or forward if the near margin is short, then observe before placing again. [Evidence](../../lessons/README.md#scripted-grasp-v2c-2026-09-22).
- `release_not_verified`: observe whether `box.held` is false and `surfaces.destination.contained` is true, with clearance near zero, before deciding whether more motion is needed. The release check can report failure after the box is already supported. [Evidence](../../lessons/README.md#scripted-grasp-v2c-2026-09-22).
- If the destination is unseen, turn toward the rough direction in the task instruction before searching elsewhere; the table enters the map only when seen by the camera. [Evidence](../../lessons/README.md#scripted-grasp-v2c-2026-09-22).
- If a heavier box turns or falls during two scripted grasps, use the documented `pickup_box` fallback. [Evidence](../../lessons/README.md#scripted-grasp-v2c-2026-09-22).

## Two ways to pick the box up

Use `grasp_box` first. It is scripted from the measured box pose, handles box
size, rotation and weight explicitly, reports why it failed, and lines itself up
over short distances, but it will not cross the room: get within about 0.4 m
(for example `approach_surface("source")`) before calling it.
`pickup_box` is a learned policy that walks to the box itself; it works well on
the nominal box but can miss, tilt, or time out on other boxes. If `grasp_box`
fails twice, try `pickup_box`.

## Verifying stages

After each stage, check it from observations yourself before moving on:
- After pickup, `box.held` should be true and `height_above_source_m` > 0.04.
- After placing, `box.held` should be false, `surfaces.destination.clearance_m`
  should be about 0, and `contained` should be true.

Tool completion plus your own check is stronger than either alone.
