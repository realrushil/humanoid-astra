# Camera observations for code generation

The outer runner attaches raw head and overview images before code generation.
These are inputs to you, the coding agent. `robot.observe()` returns measured
state; it does not call a vision model or attach another image during Python
execution. Write the same self-contained `run(robot, task)` program as usual.

Each pair is ordered head then overview. Head is robot-mounted; overview is a
fixed external camera, an explicit development observation. Images can hide
hands, contact patches and obstacles. Do not infer exact world coordinates,
force or stable grasp from their appearance.

Each packet records simulation time in seconds, action/physics steps, camera
identity and hashes. Structured state in that packet was captured at the same
step. Positions are world metres; root/box quaternions are WXYZ; contacts are
simulator-derived normal forces in newtons. These remain privileged simulation
observations, not estimates recovered from RGB or calibrated tactile sensors.

On revisions, the earlier pair is the **previous generation's observation**.
The intervening period includes motion while code was being written. Execution
feedback supplies actual program start/end observations when available; do not
attribute every visible change to the last tool call. Source `decoded_recording`
means an offline input decoded from a saved video, not a new physical episode.

Start the program with fresh `robot.observe()` and check the actual tool
preconditions and results. Earlier movements, changed grasps, elapsed time and
terminal faults persist. A partial retreat is not permission to request the
entire distance again. Use only supported parameters and recovery operations;
if none applies, report the limitation and stop issuing motion requests.

When the prompt requests `observation.md`, write a short decision record:
visible facts; changes since the earlier snapshot; uncertainty/occlusion;
relevant measured feedback; next code action. Distinguish facts visible in the
images from facts supplied by measurements. Do not narrate unobserved motion or
claim stable load-bearing contact just because the box appears between hands.

If images and measurements disagree, compare timestamps and reobserve before
motion. Do not override a measured fault because a picture looks successful.
The independent evaluator decides benchmark success; its scores and annotated
presentation videos are not part of your input.
