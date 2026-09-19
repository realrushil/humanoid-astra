# check_hands(goals)

Compute local paired wrist IK from the current measured state without changing commands. Inputs, frames, local envelope and collision assumptions match [reach_hands](reach_hands.md). Both measured hand poses must be valid; faults reject planning.

Returns `planned` or `rejected`, `reason`, snapshot `sequence`/`sim_time`, and a `joint_path` when planned. Each point contains fourteen radians: left shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw, then the seven right-arm joints in the same order. `errors` reports predicted position/orientation residuals. The root, waist, fingers and measured free-object poses are frozen in this planning state.

This is geometry computation, not physical motion or a guarantee of tracking, balance or grasping. Physics continues while planning. `reach_hands()` always makes a fresh plan; it does not execute a previously returned path blindly.
