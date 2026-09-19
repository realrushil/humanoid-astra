# set_posture

`robot.set_posture(height=None, timeout=8.)`

**Purpose and inputs:** Request neutral idle (None) or SONIC squat-mode height conditioning in [0.60,0.85] metres; timeout (0,20] seconds.

**Preconditions:** At least 0.5 s measured two-foot base settling; otherwise rejected/base_not_settled without a new posture command.

**Feedback and execution:** Sends the height condition and monitors actual pelvis height. Does not calibrate or compensate the request-to-height mapping. Completion requires height within 3 cm plus two-foot/base settling for 0.5 s.

**Result:** completed/measured_goal; timed_out if the measured condition is not attained; failed on local physical/configuration faults.

**Side effects and continuation:** Releases old arm targets. Lowering can translate the base and invalidate prior IK. Success retains the posture during editing; failure/timeout releases it and the robot may rise.

**Evidence and limits:** The input interval is not a demonstrated height envelope. Nominal 0.70 m completion and lower-height timeouts have been observed; these are not a general calibration. Reobserve and replan afterward.

See [shared assumptions](README.md).
