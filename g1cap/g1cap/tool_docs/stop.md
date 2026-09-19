# stop

`robot.stop(timeout=3.)`

**Purpose and inputs:** Request idle during navigation, or maintain the active stationary posture/arm reference while checking settling.

**Preconditions:** Running episode. Current control context determines which behavior applies.

**Feedback and execution:** Without a stationary executor: idle until base speed/yaw-rate settling for 0.5 s, using timeout (0,20]. With a stationary executor: delegates to hold(0.5), including wrist settling; this branch currently uses its own bounded duration rather than the supplied timeout.

**Result:** completed/measured_goal, timed_out/failed, or cancellation according to the active path.

**Side effects and continuation:** Idle does not freeze joints. Successful stationary stop retains the supervised reference; unsuccessful stationary stop releases it. Not a hardware emergency stop.

**Evidence and limits:** Stopping delay and drift remain. Measured nominal stops after short forward commands can travel tens of centimetres before settling; net displacement can also reverse sign across command histories. Do not treat idle as a position hold or use a fixed stopping-distance assumption. After sonic_motion, stop uses navigation idle because motion references have been released.

See [shared assumptions](README.md).
