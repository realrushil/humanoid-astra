# observe

`robot.observe()`

**Purpose and inputs:** Read the latest supervised robot snapshot; world Z up, metres/radians/seconds.

**Preconditions:** Available even after the episode terminates.

**Feedback and execution:** No motor command. Returns pelvis/wrist poses and velocities, named joints, foot forces, timing, profile labels, settled_for and episode_status. These are simulator-derived observations.

**Result:** A state dictionary, not completed/failed. Check age and episode_status.

Raw contact rows identify both geometries and include `position_world` (metres), `normal_world` (unit direction from geometry 1 to geometry 2), and `normal_force` (newtons). These are privileged simulator measurements, not deployed tactile sensing. `first_environment_contact` retains the first forbidden external contact, with its simulation time; it is historical evidence rather than a current-contact flag. The runtime rejects this event during startup. These fields do not authorize intentional contact through existing tools.

The legacy `approached` field applies only to the older approach/height task.
It has no meaning for workstation reaching; it must not be used as completion.

**Side effects and continuation:** No command changes, but physics continues during the call.

**Evidence and limits:** An observation does not establish future stability. Nominal timing/contact/localization assumptions are uncalibrated.

See [shared assumptions](README.md).
