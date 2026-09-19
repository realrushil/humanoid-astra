# Arena box tools

Write one self-contained Python file defining `run(robot, task)`. Use ordinary
Python, `math`, functions, dictionaries, bounded loops and `print`. Other imports,
files, network and simulator access are unavailable in the submitted worker.
The parent executes this file once; helper files are not bundled. Inspect the
provided source when useful. Do not import that source into your program.

This session exposes **only** `observe`, `wait`, `pickup_box`, `lift_supported_box`, `raise_held_box`,
`hold_box`, `return_box_to_source`, `retreat_with_box`, `turn_with_box`,
`move_with_box`, and `place_box`.
The original development fixture is a20cm,0.1kg cube. Explicit benchmark recipes
also vary box dimensions/mass and desk geometry. Read `task.box_size_m` and
`task.box_mass_kg`; never assume the original cube. These new scenes are tests
of capability, not qualified operating ranges. Loaded tools have nominal
evidence with the light cube; larger/heavier objects, other layouts, reliability
and hardware transfer remain unqualified. Tools may fail after changing the world.

## Observations and physical time

Start every program/revision with `robot.observe()`. It returns `session_id`,
`episode_status` (`running` or a terminal reason), `time` in simulation seconds,
`step`, `loaded_contacts`, `approach_clearance_m`, `state_age_s` in wall seconds, `root_pos`, `root_quat`, `box_pos`,
`box_quat`, `box_size_m`, `box_mass_kg`, `clearance`, `bilateral`, `supported`, `hand_forces_N`,
`foot_upward_N`, `stance_clear`, `hold_observation_v2`, `controller_phase`,
`box_bounds`, `surfaces`, `turn_clearance_m`, and `box_floor_contact_peak_N`.
Positions use metres in a fixed simulation world; quaternions are WXYZ.
`clearance` is the lowest box point above the source top, not centre height.
`bilateral` means each hand's summed box contact exceeds 0.5 N. `supported`
means table upward force is at least half the box weight and clearance is
within 1 cm of the tabletop. These are simulator-derived development signals,
not a deployed perception system. `task.source_bounds` gives the source's
world min/max bounds. `surfaces` is a dictionary keyed by `source` and
`destination`. Each entry contains world `bounds.min/max`, `clearance_m`,
`box_upward_N`, `robot_contact_peak_N`, `supported`, `contained`, and
`lower_body_clearance_m`. The latter is conservative measured 3D collision-box
separation from the lower body to that table's top/legs. `box_bounds.min/max`
is the rotated cuboid's world AABB using its actual full dimensions. Containment concerns its whole XY
footprint, not just its center. `turn_clearance_m` is circle-to-table-footprint
clearance using the current whole robot/box radius about the pelvis. These are
sampled geometric bounds, not a continuous collision or reachability proof.
Both supports and cameras belong to the same recorded observation step.
The independent task scorer is outside the worker.

Compute routes from current support geometry and measured robot heading. No
absolute destination heading is guaranteed. Respect `retreat_with_box`'s
world−X admission restriction and the relative-heading contracts of turning
and forward motion. Do not increase grip limits or assume an object is feasible
merely because its mass is known. Prior-scene route code is not a task solution.

For a proposed world-XY waypoint `(x, y)`, compute the displacement from the
current robot position, then `desired_yaw = math.atan2(dy, dx)`. Extract current
yaw from the normalized WXYZ root quaternion as
`math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))`. A relative turn request is
`math.atan2(math.sin(desired_yaw-current_yaw), math.cos(desired_yaw-current_yaw))`.
Skip a zero displacement/angle; split angles exceeding the admitted per-call
range and reobserve between calls. This is coordinate conversion, not route
planning: check the turn-clearance contract before asking the robot to turn.

The task concerns the **box** footprint on the selected support. A waypoint for
the robot's root is not automatically a placement pose: the held box is offset
from the root and moves around it during a turn. Use current box bounds and
surface bounds to check approach/containment, and recompute them after turning
or translating. Neither a waypoint bearing nor a positive current clearance
certifies collision-free motion along the whole route.

`hold_observation_v2.ready` means a **raised** stable hold, including clearance
>=5 cm throughout the observed second. It is not a general grasp-validity flag.
The legacy reason `moving_or_unsupported` also covers a stationary grasp below
5 cm. Inspect height, contact and the reported pose-motion maxima separately.
In particular, do not require raised-hold readiness before asking a low-grasp
recovery tool to evaluate its own admission conditions.

The same physics world advances during generation and between programs. A failed
tool or Python exception does not undo actions or reset the box. Only the owner
can end the episode. Stop requesting motion if `episode_status != 'running'`.
Observe again after motion/failure; generation-time observations may be old.
The owner rejects stale control observations over 0.25 wall seconds. Native
neural inference is step-synchronous and can slow wall-clock progress; this
runtime is not a qualified real-time controller. Initial standing and all waiting
are recorded. No recorded-command setup is used.

## Results and bounds

Motion calls block while physics continues. They return a dictionary with
`status` (`completed`, `failed`, `rejected`, `cancelled`), `reason`, and measured
`observation`. This result observation is a compact measurement subset; it does
not contain `episode_status` or the full `observe()` schema. Call
`robot.observe()` for fresh session status after an action or failure. A failed
operation may have moved the robot or box. A rejected
request does not start motion. Normal return/failure holds the last loaded
joint/torso references with zero requested navigation; it does not guarantee
that the body or object stops moving. Inspect every result and print concise
labels/results for feedback. Bound retries; repeating an identical request
without a physical reason is not useful recovery. Tool waits are bounded to
180 wall seconds; a program to `task.worker_timeout_s` (default600) and100 calls.
A recipe may declare a longer finite worker/session wall budget. Collision above
5 N with either table, measured box-floor contact above0.5 N, tilt above15 degrees, missing control observations,
native termination or the episode deadline ends this episode. These limits and
contact/actuator assumptions have not been calibrated to hardware.

### `robot.wait(duration=1.0)`

Advance this same physical episode for0 < duration <=5 simulated seconds.
Navigation is zero; the last arm, finger, pelvis-height and torso targets are
retained while HOMIE continues balancing. The tool observes and records physics;
it does not freeze the robot or box. It does not require an already stable grasp.

`completed / dwell_elapsed` means the requested time elapsed, **not** that the
box is held or the robot task succeeded. Call `observe()` afterward and inspect
`bilateral`, `clearance`, `stance_clear` and `hold_observation_v2` before another
motion. Waiting may let a stopped motion settle, or reveal that a grasp was lost.
Physical faults, native termination and episode time continue to count. Earlier
tool failures and their motion are retained. This is useful after a conservative
motion stop; repeating a failed move without measuring its consequences is not.

### `robot.pickup_box(object_id)`

Only `brown_box` is accepted. The native GR00T neural policy proposes 43 joint
targets, base velocity, pelvis height and torso orientation. HOMIE supplies
balancing. A conservative lower-body/table clearance limiter changes navigation;
measured hand/table clearance and approach speed also drive a bounded common
upward wrist adjustment. Only arm targets change in that adjustment; fingers,
torso and height remain neural proposals. These checks use simulator collision
geometry, do not certify collision avoidance, and may alter or fail acquisition.
They do not change the completion criteria or automatically recover a low grasp.
The native model retains the tool's fixed acquisition
instruction mentioning transfer to a destination bin, but this operation stops
navigation after acquisition and verifies a held box. Its learned behavior is
not deterministic or guaranteed by the Python function signature.

The coding agent's task text is not forwarded as a replacement policy instruction.
Changing that learned-policy input or the visible fixture requires separate
qualification; a natural-language paraphrase is not guaranteed to preserve the
same grasp. The legacy destination in its conditioning does not authorize this
tool to execute the rest of a transfer. Subsequent motion remains separate
explicit tool calls.

Acquisition is bounded to 15 simulated seconds, then at most 5 seconds of hold
verification. Completion requires bilateral contact, at least 5 cm clearance
and a full stable second. Known failures include `acquisition_timeout`,
`pose_hold_timeout`, and `clearance_below_goal`. The last means a settled
bilateral grasp is retained but too low; no automatic height correction occurs.
Contact loss or a terminal body collision is not a retained-grasp recovery case.

### `robot.lift_supported_box()`

Lift an already settled two-handed grasp while the source table still supports
the box. This is a separate recovery operation; it does not call the neural
policy again or make the earlier pickup successful retroactively. Admission
requires a complete stable second with bilateral contact, clear stance,
source support and absolute source clearance at most1 cm. Hand contact must
remain above0.5 N and at most25 N across all four physics substeps; total foot
support must exceed5 N. The normal pose-motion limits apply.

Both wrist targets translate6 cm along world+Z at1 cm/s, retaining wrist
orientations, fingers, pelvis height and torso targets, with zero navigation.
After7 seconds including tracking allowance, it verifies up to5 seconds for
an actual stable raised hold at least5 cm above the source. `completed /
supported_grasp_lifted` requires measured success. Known outcomes include
`stable_supported_grasp_required` (rejected), `preparation_force_limit`,
`lift_contact_or_stance_lost` and `supported_lift_hold_timeout` (failed).
The force reason shares the existing contact-bound vocabulary; no inward
squeeze is performed by this operation.

A pickup `pose_hold_timeout` may leave a supported grasp that qualifies, but
check the current observation and result. An unsupported low grasp uses
`raise_held_box` instead. A terminal collision cannot be recovered by this call.
The6 cm request and25 N ceiling have nominal controller witnesses with the
20 cm/0.1 kg development cube; bigger boxes, grasp disturbances and hardware
performance are unqualified. Failure retains its physical consequences.

### `robot.raise_held_box(clearance_m=0.08)`

Requires a full second of settled bilateral grasp, clear stance and current
clearance above 1 cm. Target clearance is 5–10 cm. The necessary common upward
wrist translation must be at most 6 cm. The arm controller follows the measured
world-up translation at 1 cm/s, retains loaded references and finger targets,
then verifies a stable raised hold. It is not a fresh grasp tool. Preconditions
and the admission result determine whether a retained low grasp is recoverable.

This tool deliberately does not require `hold_observation_v2.ready`. Its owner
checks the previous second for bilateral contact, clear stance, clearance>1 cm,
box speed<=0.05 m/s, angular speed<=0.2 rad/s and root planar speed<=0.05 m/s.
A low held box after a stopped carry or turn can qualify under the same rules
as a low pickup. A bounded `wait` followed by fresh observation can establish
settling; it cannot itself increase commanded height. The raise may still be
rejected or fail, so inspect its result and reobserve. A source-supported grasp
at clearance<=1 cm uses the separate supported-lift contract.

### `robot.hold_box(duration=1.0)`

Requires an already verified raised grasp and accepts 0 < duration <= 3 simulated
seconds. Checks contact, clearance, stance and pose motion, with at least one
full observed second even if a shorter dwell is requested. It can fail if the
box slips or motion does not settle. It does not actively reacquire a lost box.

### `robot.return_box_to_source()`

Compatibility spelling for `place_box(surface_id="source")`, using the same
selected-support controller and checks below. It does not navigate back to the
source or undo earlier travel. An untouched supported box cannot satisfy the
pickup/hold/return task.

### `robot.retreat_with_box(distance_m)`

Move backward along **fixed world−X**, then stop and verify holding. Distance
must be0.20–0.50 metres. The witnessed scene starts facing world+X; admission
requires heading within5 degrees of that direction, a full stable raised grasp,
clear two-foot stance and at least3 cm lower-body clearance from the source.
This is not navigation around obstacles or movement in an arbitrary direction.
It has only been qualified with the20 cm/0.1 kg development cube in this scene.

The first loaded motion in an acquired grasp prepares it once. It moves each
wrist target inward by at most4 mm along
the measured horizontal wrist axis, at2 mm/s. It preserves wrist orientation
and finger targets, keeps navigation zero, and requires at least3 seconds plus
a full stable raised hold; preparation times out after5 seconds. This is a
bounded position adjustment, not calibrated force control. Hand-box contact
must stay<=25 N per hand across every physics substep, with bilateral contact,
clear stance and box clearance>=3 cm. These are development assumptions for
this light cube. Further arm-reference changes stop when either hand reaches
20 N, or after the2 s displacement interval; the controller retains those
references while verifying the hold. It never resumes tightening when force
falls. Preparation can fail and leave the grasp changed.

Subsequent retreat, forward and turn calls reuse the prepared grasp, including
a retry after a recoverable motion stop and measured settling. They do not add
another4 mm of closing travel. Failed/interrupted preparation cannot be repeated
without a new explicit acquisition; it returns `grasp_preparation_not_verified`.
A new pickup or verified release clears preparation ownership.

HOMIE then balances under the prepared arm/hand/height/torso targets. The controller
caps planar velocity at0.12 m/s and uses simulator position feedback to correct
sideways drift. Single-foot support is allowed while walking. Four200 Hz physics
samples per action are summarized in `loaded_contacts`: `samples`,
`minimum_hand_N` (minimum across both hands and substeps),
`maximum_hand_N` (maximum individual hand contact across substeps), and
`minimum_total_foot_N`. Missing summaries are rejected/failed; any hand<=0.5 N
or total ground support<=5 N causes failure even if the final sample recovers.

During movement and settling, clearance must stay>=3 cm, sideways error<=5 cm,
and body orientation change<=0.20 rad. Box-to-pelvis displacement is reported
as a diagnostic: compliant arm/body movement can change it while the box is retained. Retreat
is bounded to6 simulated seconds after preparation, then settling to5 seconds.
The stop trigger targets the requested distance center. Completion needs
actual root travel in [distance−2 cm,distance+3 cm], box travel>=distance−5 cm,
and the original full stable raised-hold second with clear two-foot stance.
The result adds `measurements` containing `root_retreat_m`, `box_retreat_m` and
`lateral_error_m` and `box_pelvis_change_m` (current change from carry start);
command duration alone cannot establish success.

Known failures include `preparation_force_limit`, `invalid_preparation_force`,
`preparation_contact_stance_or_height`, `preparation_did_not_settle`, `lateral_drift`, contact loss,
`retreat_timeout` and `loaded_hold_timeout`. Failure requests zero navigation
and retains loaded targets; observe the changed world before deciding anything
else. Retreat does not return toward a table. `return_box_to_source()` rejects
a box no longer projected above the source; it cannot undo retreat.

For a `retreat_hold` task, use `task.retreat_distance_m`, inspect every tool
result, and finish while holding the box. Do not add placement to that task.

### `robot.turn_with_box(yaw_rad)`

Turn by a **relative** yaw about world+Z, in radians; positive is counterclockwise.
Require0 < abs(yaw_rad) <= pi/2. The requested angle is relative to measured
heading at tool entry. Nominal physical witnesses are clockwise turns near90°;
other admitted angles/directions remain unqualified.

Admission requires a stable raised grasp and `turn_clearance_m >=0.18 m`:
15 cm for allowed pelvis drift plus3 cm clearance around the measured robot/box
circle. It also checks lower-body separation from both tables. The tool caps
yaw command at0.25 rad/s and angular slew at0.5 rad/s², with planar position
correction capped at0.04 m/s. It is allowed to move its pelvis within15 cm;
this is not rotation around a perfectly fixed point. Original strict hand/ground
substep guards and3 cm box clearance remain active. Leaving the region,
wrong-way motion or overshoot beyond5° fails.

Turning is bounded to12 s and initial settling to5 s. The tool then owns3 s of
**zero navigation**, still checking region, heading and held-state stability.
Completion needs heading error<=5°, measured yaw rate<=0.1 rad/s and a stable
held box. The result reports relative yaw, heading error and pelvis travel.
Failure retains the changed heading; compute a remaining angle from fresh
observations instead of repeating the original turn blindly.

Known timeout outcomes are `turn_timeout`, `turn_hold_timeout` and
`zero_navigation_hold_not_verified`. The last means the final zero-navigation
verification failed; inspect actual heading, height, contact and motion before
choosing recovery. It does not imply that the requested rotation never occurred.

### `robot.move_with_box(distance_m)`

Carry forward0.20–0.50 m along the **measured entry heading**, then stop.
The heading is held as a world direction; this call does not turn or choose a
path. It requires a stable raised grasp and measured lower-body separation
>=3 cm from both tables. Preparation is shared with other loaded tools as above.

The desired forward walking command is0.12 m/s, with total planar cap0.12 m/s,
0.3 m/s² command slew and bounded sideways correction. It checks all four hand/
ground contact samples, box clearance>=3 cm, lateral error<=5 cm and body
orientation change<=0.20 rad. Motion is bounded to6 s, settling to5 s.
Completion requires robot travel in[distance−2 cm,distance+3 cm], box travel
>=distance−5 cm, and the full stable held-pose window. Measurements are
`root_forward_m`, `box_forward_m`, `lateral_error_m`.

Known bounded failures include `translation_timeout`,
`translation_hold_timeout`, `final_displacement_outside_goal`,
`grasp_or_height_lost` and `lateral_drift`. The accepted distance range is not
a promise of completing it within six seconds. Current forward witnesses are
short20–30 cm segments; a50 cm request has timed out after partial travel.
Do not treat a timeout as zero displacement or invent a different reason name.

Reobserve after **every** segment or failed request. Coast and stopping errors
count. Choose subsequent distance from the actual box footprint and target
bounds; do not calculate all chunks from the initial location. A final segment
must fit the far boundary as well as the near boundary. If no admitted20 cm
segment fits, do not force a request outside the tool's range.

A general approach calculation is the intersection of XY intervals for forward
translation `t >=0`: for each axis i, require
`surface.min[i]+margin <= box.min[i]+t*heading[i]` and
`box.max[i]+t*heading[i] <= surface.max[i]-margin`.
Handle zero and negative heading components explicitly. A planning margin is
your choice; it is separate from actual final containment. Check robot clearance
too. This formula does not choose a route or certify a feasible grasp.

### `robot.place_box(surface_id)`

Select `"source"` or `"destination"` from current `surfaces`. The robot must
already be stationary with a verified raised grasp and the entire box footprint
above the selected tabletop. Its bottom-to-table clearance must be positive
and reachable by a pelvis decrease of at most14 cm without commanding height
below0.60 m. No approach or route is hidden in this tool.

Placement now prepares the held box before lowering. It rotates both wrist targets together about the measured box center to bring the nearest box face toward horizontal. It preserves their relative grasp transforms and finger references. The request is bounded to45° at5°/s, with at most12 s active alignment; measured tilt<=3° stops active IK, followed by3 s retained-reference verification with final tilt<=5°. Both hands must remain in contact, each hand's peak summed normal-contact magnitude must stay<=25 N, wrist error<=2 cm, box-center drift<=3 cm, and clearance above every table>=3 cm. Orientation tracking error must stay<=10°. These are simulator-derived checks and development bounds.

The controller then derives a horizontal direction from the box toward the robot root and searches for a common wrist retraction of0–8 cm. A geometry proxy transforms static collision geometry by measured link poses, then shifts it down by the current box/table clearance plus2 mm. Native placement uses source convex hulls for mesh colliders and conservative oriented bounds for primitive/non-hull colliders, with cached Coal distance queries. These source hulls are not asserted identical to PhysX cooked meshes. Bounds-only offline callers retain the conservative AABB screen. Planning requires at least5 mm predicted arm/table separation and1 cm whole-box edge margin. It prefers8 mm arm clearance: choose the smallest retraction attaining that reserve, or the greatest feasible clearance if the preference is unreachable. After motion, verification still requires the full5 mm separation and the same edge margin. The desired3 mm reserve is uncalibrated. The proxy is not a swept-volume, self-collision or physical-contact guarantee; actual physical faults remain enforced. If no candidate exists, placement fails at that stage instead of moving the box off the support.

Only placement preparation opts into the8 cm wrist allowance; ordinary common translation and upward recovery retain6 cm. Retraction requests1 cm/s, allows10 s active motion and2 s held verification, and enters verification only after measured displacement is within2 mm of its request and current geometry meets the full required clearance. It retains loaded references during verification. Transverse/height drift must stay<=1 cm, box orientation change<=5°, wrist error<=2 cm, overshoot<=5 mm and hand peak force<=25 N. All thresholds and the result use the current physical state; the operation does not replay a stored placement trajectory.

After preparation, the controller lowers the pelvis at2 cm/s while retaining loaded upper-body targets. It needs15 consecutive actual-support frames, then2 s of continuous support before withdrawing wrists horizontally outward by6 cm per hand at3 cm/s. Withdrawal is bounded to4 s, followed by2 s settling. Fingers remain unchanged. During withdrawal/settling the box must remain fully contained and within1 cm of the selected surface. Completion requires a full stable second of actual selected-table support, each hand<=0.5 N, clear standing, box speed<=0.05 m/s and angular speed<=0.2 rad/s. Zero contact at one frame or successful admission is insufficient.

The stages share one episode and retain every action consequence. Preparation can fail after changing the box pose; reobserve and use the returned reason before continuing. The complete operation can consume up to about42 simulated seconds. It does not navigate, reacquire a lost grasp, regrasp with different fingers or guarantee arbitrary orientations/objects. The8 cm source-hull placement preparation has bounded controller witnesses; a20 cm/0.1 kg cube completes release. A clear lowering pose does not establish a clear outward-withdrawal path: later wrist/table contact can still terminate the operation after box support, even when hand/object forces have dropped to zero. Fresh workflow qualification across seeds and changed scenes is separate; no arbitrary grasp, mass or hardware qualification is implied.

Known failures include `box_left_selected_footprint` (XY containment lost),
`box_left_selected_surface` (absolute height gap exceeded1 cm, even if XY still
fits), `selected_support_lost` and `release_not_verified`. Retain the failure
and inspect current physical state; do not declare placement from a contact
signal or assume future natural settling has already occurred.

A missing selected-support observation is rejected; source force cannot stand
in for destination force. Forbidden robot-table contact and body faults remain
active. Supported placement does not imply arbitrary larger/heavier boxes or
lower tables are feasible. The development transfer fixture uses two empty,
equal-height tables and a20 cm/0.1 kg cube; the original lower-table/bin fixture
is a separate environment.

### Task scoring versus tool guards

The `retreat_hold` benchmark reports `scoring_version="retreat_outcome_v2"`.
It keeps the first measured stable-hold anchor for root/box distance throughout
intentional lifting and preparation. Success requires actual transport,
continuous bilateral contact/clearance, the declared route/body limits and
a final stable hold. `maximum_box_pelvis_change_m` is a diagnostic, not a task
failure: intended lifting can change it. The carry tool likewise reports box/pelvis change diagnostically, following
a matched physics continuation that completed carrying beyond the former4 cm alarm.
It still enforces contact, clearance, route/body limits and stable holding. A stopped
tool is not proof of a dropped box or a completed task. Previous recorded
scores are versioned separately and must not be silently recomputed as current.

The installed Isaac contact sensor supplies **normal contact forces** in
`force_matrix_w`. Our `hand_forces_N` sums their per-link magnitudes after
filtering for this box. It excludes separately tracked friction forces. The
current recorder does not enable contact-point or friction tracking, so these
scalars do not locate a contact patch or demonstrate resistance to sliding.
They are simulator observations, not calibrated tactile measurements.


For `table_transfer`, the task names `surface_id` and requires at least0.5 m
box displacement after the first stable raised grasp, then stable full footprint
support and both hands released on that selected table. Its scorer reads actual
poses/forces independently of tool results. Recoverable tool failures remain in
the log and consume physical time; they do not automatically mean the box was
dropped. Box-floor contact, forbidden table collision, body tilt, missing evidence
or native termination remain failures. End by observing the released box and
standing robot. The old `retreat_hold` scoring version keeps its own stricter
continuous-contact contract; do not reinterpret historical results.
