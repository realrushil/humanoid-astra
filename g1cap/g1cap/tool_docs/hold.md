# hold

`robot.hold(duration=1.)`

**Purpose and inputs:** Refresh current stationary references for [0.5,5] simulation seconds and assess settling.

**Preconditions:** Healthy episode. A previous reference invalidated by supervisor fallback cannot be assumed active.

**Feedback and execution:** Requires two feet >=5 N, tilt <=15 degrees, base speed <=0.05 m/s, yaw rate <=0.10 rad/s and wrist speed <=0.05 m/s while completing its duration/dwell rule. Fixed-arm holds monitor configuration drift.

**Result:** completed/measured_goal for hold conditions; rejected/reference_expired or failed/timed_out otherwise.

**Side effects and continuation:** Successful stationary references persist under the supervisor while code is edited. Failures release the reference toward idle. A fresh hold without an existing stationary target requests nominal idle.

**Evidence and limits:** Does not retain a sonic_motion trajectory after that tool returns. Does not create a grasp or guarantee stability under external loads.

See [shared assumptions](README.md).
