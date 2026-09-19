# walk_to

`robot.walk_to(position_xy, timeout=15.)`

**Purpose and inputs:** Approach a world XY base position in metres; timeout (0,20] seconds. Legacy/workstation tasks retain a 1.2 m radius from initial pelvis XY. Mobility tasks explicitly declare a larger workspace and fixed route corridor; the requested goal must lie inside both. The route is not automatically followed: choose waypoints and reobserve progress.

**Preconditions:** Running episode with healthy observations. Space for the entire body must be available.

**Feedback and execution:** Outer feedback loop converts position error to bounded body velocities (up to 0.25 m/s, lateral <=0.20). Within 6 cm it requests idle and checks base planar speed <=0.05 m/s and yaw rate <=0.10 rad/s for 0.5 s.

**Result:** completed/measured_goal for that arrival predicate; timed_out if it cannot arrive/settle; cancelled on terminal fault.

**Side effects and continuation:** Releases old posture/arm targets. Robot can overshoot and settle elsewhere; feet and arms may hit obstacles. Always requests idle on exit. Does not preserve a grasp.

A long route can exceed one call's time budget. A timeout retains the distance already walked. `observe()['task_progress']['waypoint_index']` identifies the first unfinished mobility region across program revisions. The benchmark's arrival radius is 12 cm; this tool's tighter 6 cm arrival predicate is distinct from task success.

**Evidence and limits:** Useful motion is demonstrated in nominal scenes, but not general arrival reliability. Near-goal velocity requests can stall outside tolerance; shorter targets can overshoot and lateral targets can make little progress. Inspect the returned status and current position instead of assuming smaller corrections are easier. This is not scene-aware route or footstep planning. Reobserve before posture/reaching.

See [shared assumptions](README.md).
