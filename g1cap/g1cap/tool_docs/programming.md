# Writing a robot program

Write ordinary Python `run(robot, task)` using the documented tools. Helpers and bounded loops can live in the same file. The submitted worker receives that file; development files are not automatically bundled. Read the task, current observation and the contracts for tools you use. This guide is general program-writing knowledge, not a benchmark solution.

Use [tool selection and composition](choosing_tools.md) to choose a sequence by its physical objective, check which references survive between calls and identify missing capabilities before writing code.

## Work from measured state

Observe at entry and after an action changes the world. Continue from current task progress; never assume a new program starts a fresh episode. Stop issuing actions when `episode_status` is terminal. A Python exception, timeout or rejected request does not undo earlier movement. Simulation time also passes while code is generated or revised.

Use object-local geometry to calculate candidates, then transform it to the required world frame. For an object pose `(p,R)`, a local offset `d` gives world point `p + R*d`; a local orientation becomes `R*R_local`. Quaternions are WXYZ here. Wrist, palm and fingertip frames differ. Always use a fresh measured object pose instead of its scene initialization coordinates.

## Make feedback useful

Check the action's status and measured result. A planning rejection calls for a different target or stance. A tracking timeout means motion may have occurred: reobserve before choosing whether to retry. A terminal contact/fall/time fault ends the episode. Bound retries and total action time; repeated identical requests without measured progress are not a recovery strategy.

For navigation, reason about facing and route geometry, then use measured position/heading and settling. Turning limits apply per call. Long travel may need multiple bounded calls. `move_base` uses body-frame velocities; persistent position goals use world coordinates. A completed timed command means its interval elapsed, not that a spatial goal was reached.

## Preserve intended holds

Know which tool owns the current reference. Successful stationary reaches leave a supervised reference; navigation/posture changes can release it. A paired pose reference is not an object grasp. Until a load-preserving navigation tool is qualified, composing reaching and walking cannot establish carrying.

Print concise action labels and results so feedback and video can show what the program intended. Prefer readable conditionals and a few helpers over broad search, long sleeps, or unbounded loops. Benchmark success is determined independently from measured physical state.
