# check_reach

`robot.check_reach(target_world)`

**Purpose and inputs:** Read-only right-wrist position IK, world XYZ metres, within 25 cm of current wrist.

**Preconditions:** Healthy state and available model planner; no motor command is installed.

**Feedback and execution:** Uses measured root/waist/left arm/fingers fixed in a separate model; solves seven right-arm joints with limits and discrete collision checks. Returns snapshot sequence/time and joint_path.

**Result:** planned/local_ik means kinematic candidate found; rejected with reasons such as target_outside_local_envelope, ik_not_converged, initial_model_collision or path_model_collision.

**Side effects and continuation:** Planning consumes physical time and does not freeze the body. No new motor command, no reset. Result can become invalid as posture changes.

**Evidence and limits:** No balance/tracking guarantee, full-body route planning or proof of global unreachability on rejection. Endpoint is wrist body origin, not a fingertip; no orientation/grasp constraint.

See [shared assumptions](README.md).
