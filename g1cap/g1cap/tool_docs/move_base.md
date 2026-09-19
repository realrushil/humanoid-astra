# move_base

`robot.move_base(vx, vy, yaw_rate, duration)`

**Purpose and inputs:** Request BODY forward/left velocity (m/s) and yaw rate (rad/s). Norm <=0.30, |vy|<=0.20, |yaw_rate|<=0.40; duration (0,2] seconds.

**Preconditions:** Running episode; no path clearance calculation is provided.

**Feedback and execution:** Maintains a bounded velocity request. Independent leases stop refreshing if the executor stalls. No destination or displacement feedback.

**Result:** completed/duration_elapsed means only the time interval elapsed; actual displacement and speed can differ.

**Side effects and continuation:** Releases stationary posture/arm references. Requests idle afterward; the robot may continue settling. Previous physical motion is retained.

**Evidence and limits:** Short lateral commands may produce little net displacement. Small forward speed requests can also produce planted feet or negligible progress; a lower requested speed is not a calibrated slower version of a working gait. The backend converts BODY movement to world direction using current yaw, which can trigger upstream replanning as yaw changes. Duration completion does not validate the resulting gait. After short working bursts, stopping can add substantial displacement, so burst duration is not a calibrated distance command. No exact velocity tracking or contact-free route is guaranteed.

See [shared assumptions](README.md).
