# reach_right

`robot.reach_right(target_world, timeout=10.)`

**Purpose and inputs:** Attempt a stationary position-only right-wrist reach in world XYZ metres; timeout (0,20] seconds.

**Preconditions:** 0.5 s base settling; healthy state; model planner; target within 25 cm local wrist envelope.

**Feedback and execution:** Replans from fresh measured state, time-parameterizes the seven-joint path, then sends upper-body references through SONIC. Monitors configuration drift, support/contact faults, wrist error <=5 cm and wrist speed <=0.05 m/s together with base settling for 0.5 s after the path.

**Result:** completed/measured_goal only from measured tracking; rejected for planning/entry failures, failed on invalidated configuration or physical faults, timed_out on unmet tracking.

**Side effects and continuation:** A successful arm/posture reference persists during editing; navigation or sonic_motion releases it. Failure can leave displacement and releases references.

**Evidence and limits:** Local collision checks sample configurations, not all continuous dynamics. Current workstation trials have not established a complete reach witness with the new profiles. No wrist orientation, finger grasp, load or simultaneous carrying support.

See [shared assumptions](README.md).
