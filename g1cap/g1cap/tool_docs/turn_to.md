# turn_to

`robot.turn_to(yaw, timeout=8.)`

**Purpose and inputs:** Request world pelvis yaw in radians, within pi/2 of current heading; timeout (0,20] seconds.

**Preconditions:** Running episode and room for the whole body to turn.

**Feedback and execution:** Outer yaw-error feedback requests angular rate up to 0.35 rad/s. Within 0.08 rad it requests idle and checks base planar speed <=0.05 m/s and yaw rate <=0.10 for 0.5 s.

**Result:** completed/measured_goal, timed_out, or cancelled on terminal fault.

**Side effects and continuation:** Releases old stationary targets and requests idle on exit. Turning can translate the base and undo a previous position goal.

**Evidence and limits:** Several nominal small/medium turns have reached their measured heading tolerance, but large-turn reliability and body clearance are not qualified. Net translation at return does not bound the body excursion during a turn. Heading can also drift outside tolerance after a completed return: idle retains the backend's integrated facing reference, which can differ from this tool's goal. A nominal experiment aligning that reference still showed temporary loss of measured heading tolerance after completion; reference equality alone does not ensure physical holding. The experimental change was not adopted. Reobserve heading before relying on it. A correct heading does not imply the whole task is complete.

See [shared assumptions](README.md).
