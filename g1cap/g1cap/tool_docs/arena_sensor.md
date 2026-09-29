# Sensor estimates for the G1 coding worker

Write ordinary Python `def run(robot, task):`. State and physical consequences
persist between programs. The task names the object and intended destination;
it does not provide physical object properties or exact table geometry.

Current track: **sensor-based pickup, short retreat, relative turn and hold**. Body balance, acquisition,
local box/scene estimates, paired holding and controller stops consume permitted
measurements. Independent scoring uses simulator truth outside the controller.
This track has not qualified full sensor-only transfer.

Pickup admits paired stabilization after two consecutive fresh raised
camera/encoder observations (at most 150 ms apart). Cached flags do not count
as new images. This does not mean pickup succeeded: the full subsequent
visual/scene stability check must still pass. A visible local grasp may be
stabilized while scene motion is unavailable, but cannot report success then.

The shared wrist controller uses current 50 Hz encoders and IMU orientation.
Floor height comes from the last camera estimate and expires after 150 ms;
propagation does not create a new image or global robot position. Both wrist
targets, their loaded reference offset and accumulated lift budget persist
across holding, raising, backward carry, turning, waiting and program revisions. Missing
floor/IMU continuity or loss of observed retention stops correction explicitly.
These timing and motion limits are simulation assumptions, not calibrated
hardware capabilities.

RGB-D delivery is configured by the runtime (currently 10 Hz by default, or
25 Hz explicitly). Grasp, scene-stability and raise readiness use elapsed image
time, not repeated control ticks. Changing camera rate does not change the
one-second requirement. Processing adds no simulated latency in this setup;
real-time throughput has not been qualified.

A failed task and loaded stabilization have separate outcomes. When navigation
stops, `observe()['loaded_stop']` can report `active`, `failed`, `inactive` or
`unavailable`, with its own timestamp and, when available, a conditional arm-path
clearance lower bound in metres. `active` is not task completion or proof of
contact. The same wrist owner can correct only with fresh visual retention,
floor/IMU continuity and an accepted tabletop path screen. Missing evidence or
a failed paired controller retains prior references and reports failure; there
is no implicit regrasp, reset or new anchor. Sensor faults prohibit this path.
Do not infer free space, height precision or permission to resume navigation
from a successful stabilization screen. This coupled-stop integration remains
under simulation validation.

Available calls:

| Call | Meaning |
| --- | --- |
| `robot.observe()` | Fresh public estimate snapshot; no physical step requested |
| `robot.wait(duration=1.0)` | Advance the existing episode, 0 < duration <= 5 seconds |
| `robot.pickup_box('brown_box')` | Live learned acquisition, visual/scene verification and bounded recovery of a settled low grasp |
| `robot.raise_held_box(clearance_m=0.08)` | Raise a visually retained, settled box to 0.05–0.10 m above the observed source plane; no downward motion |
| `robot.hold_box(duration=1.0)` | Retain paired scene targets; 0 < duration <= 3 seconds, with at least one second of verification |
| `robot.retreat_with_box(distance_m)` | Move away from the observed source front, 0.20–0.50 m, then verify distance and settling; development qualification |
| `robot.turn_with_box(yaw_rad)` | Relative loaded yaw, nonzero magnitude at most π/2 radians, then fresh visual settling; development qualification |

Forward carry and placement are not yet exposed in this track. A transfer instruction remains
the final task; pickup/hold alone does not solve it. Do not invent unsupported
calls or parameters. Check each returned status before further action.

`observe()` includes named body joints in radians, velocities in rad/s,
estimated actuator torque in N m, and pelvis-axis gyro/specific force.
Torque is not measured external hand force. Object mass remains explicitly
unknown. Accepted box dimensions and pose are RGB-D estimates in optical
camera coordinates (+X right, +Y down, +Z forward), at `observed_at_s`.
Each estimate retains its timestamp/age; it is not a global world pose.

When the declared green destination is visible in the current onboard head
camera, `observe()` also includes a `destination` candidate with a timestamped
finite RGB-D plane in optical coordinates. `unavailable` means the strict
color, depth, connected-component or plane test failed. The candidate has no
world pose, route, support/contact or release meaning and cannot be reused
across a view epoch or observation gap. Its centroid and range are measured
optical-frame summaries for a future visual servo, not a destination distance
or navigation instruction.

`grasp` describes observed gap/opposing wrist proximity and stability, not
tactile proof. Its `settled_retention` checks the one-second grasp/scene motion
window without requiring the raised threshold, but still needs an available
height observation. `ready` also requires a scene-motion window. `scene_motion`
has status `initialized_local_segment`, `tracked_local_segment`, or `unavailable`
(not `available`). Initialization starts a new local reference; tracked status
means continuation within that segment, not global localization. It
uses local stance segments under a no-slip hypothesis. Missing/stale motion
has no body transform; never bridge segment IDs or assume global localization.
Unknown/rejected observations must not be filled with remembered box sizes.

The separate `retention` observation does not require source height. `retained`
means the current accepted box lies near opposing wrists with acceptable body
attitude. `settled` additionally requires one second of quiet wrist-relative
and scene-relative motion. Both are visual hypotheses, not tactile evidence.
The observation has its own `time_s`, `age_s`, `track_epoch` and stance `segment`,
expires after150ms, and resets after admitted acquisition or identity/time loss.
`pickup_proven` is always false: this field supplies no pickup proof. A box
resting on a table can satisfy retention geometry. Never use retention alone to
advance from pickup or authorize travel; check the tool result and stage verifier.
Existing tools still use their documented grasp/clearance gates. Missing
`grasp.gap_m` remains unknown even when retention is available. Camera-rate
settling can disagree with independent physical scoring.

Raise requires a fresh available estimate, opposing nearby wrists, acceptable
attitude, at least 2 cm observed clearance and `settled_retention`. It moves
both wrist targets together in the observed floor/heading frame, with balance
active. Maximum target speed is 1 cm/s and cumulative added lift is 4 cm per
observed grasp, including pickup recovery and repeated raise calls. Completion
requires a full second above the requested clearance plus a 5 mm simulation
reserve and visual readiness. The operation has an eight-second limit.
The reserve and all limits are development assumptions, not calibrated accuracy.

A retained low grasp during pickup may enter the reported `sensor_lift` phase
using this same controller with an 8 cm target. Repeating `pickup_box` while a
retained grasp is observed is rejected; it cannot recharge the lift budget.
A cancelled/failed lift is invalidated for that grasp. Keep its physical state
and inspect the result rather than blindly retrying. Loaded targets persist
after success through hold, idle and carry. No true mass, force or pose is used.

Retreat requires a ready raised grasp and fresh source-front/floor observations.
It holds both wrists relative to observed floor height/tilt and gyro heading,
while HOMIE controls balance and walking. This frame travels horizontally with
the body; it does not measure global XY or lateral drift. Distance is change in
the visible source-front plane offset, with −2/+3 cm tolerance, measured braking,
at most two refinements and a 12-second limit. It requires a fresh local scene
stability window after stopping. Visual readiness may precede physical settling;
it is an estimate, not an exact contact/stability certificate.

Once pickup commits a fresh source association, `grasp.gap_reference` is
`source_height_plane`. For this reference, `gap_m` is a conditional lower bound:
`gap_estimate_m - gap_error_m`, with `gap_semantics="conditional_lower_bound"`.
The estimate is height above the observed original plane, even outside its
footprint; it does not mean a table supports the box. Raised/clearance predicates
and lift targets use this lower bound. Temporal gap-rate checks use the nominal
estimate, so they do not bound time-correlated pose error. Before source
association, under-box support `gap_m` retains its nominal-estimate meaning.

Tracking requires fresh floor/source depth and the original source identity.
Initial association retains the 1 cm fit-uncertainty limit; later observations
carry their explicit uncertainty rather than become unavailable solely at that
cutoff. Missing or invalid uncertainty is unavailable, never zero. Missing,
ambiguous, tilted or ill-conditioned geometry still rejects the estimate. The
level-source model and ±3 mm normal-point perturbation are conditional fit
assumptions, not calibrated camera/box-pose uncertainty. Coplanar tables cannot
be distinguished, and a lost floor/control-frame segment cannot silently
establish a new source identity.
Neither the front plane nor this tool establishes rear free space; its current
qualification is limited to an unobstructed local retreat. Loaded arm references
continue across retreat, hold, idle and program boundaries, without replay/reset.
Unavailable geometry stops navigation and retains references; a failed paired
arm controller requires reacquisition rather than silently reanchoring the load.

Turning requires an already active paired floor-frame hold, a ready visual grasp,
and fresh source/front/floor/hand geometry. Positive yaw is counterclockwise
about observed up; it is relative to the heading at admission. Maximum yaw
request is 0.18 rad/s, with an active minimum of 0.08 rad/s to stay above this
HOMIE version's standing-policy switch. Enter zero-command settling within
0.02 rad; completion requires heading within 0.05 rad and at least eleven distinct ready
camera samples spanning at least one second. Two refinements and ten seconds are the
maximum. These are simulation development limits, not hardware calibration.
The post-step return check also requires current readiness and heading within
tolerance; `turn_completion_not_retained` reports failure if either is lost.

Both paired targets follow measured body heading, preserving their original
loaded offsets and any added lift. This continues after the turn returns,
through hold/wait and subsequent relative turns. No separate arm/leg motor
writers are introduced. A geometry loss stops navigation; existing paired-hold
invalidation rules still apply. Relative yaw does not promise fixed XY position:
turning can translate the robot. There is no rear/full obstacle map. The source
must remain observable; destination acquisition and complete transfers remain
unqualified. Do not infer unseen free space or correct translation using a
simulator pose.

Tool replies include status, a permitted reason and the same estimate view.
`controller_stopped` deliberately provides no inaccessible contact/force detail.
An ended episode cannot be repaired by another program. The runtime retains
prior action consequences and times, including time spent generating code.

When vision is enabled, onboard head RGB and enabled wrist RGB images are attached; structured
geometry is computed from synchronized RGB-D. External views and evaluator
videos are human evidence, not coding-agent inputs. Pixels do not establish
unseen geometry, exact force, or success. No hardware qualification is claimed.

Pickup navigation uses a visible table-front estimate and body encoders. It requires two consecutive camera measurements and expires after 150 ms. `sensor_approach_unavailable` rejects or stops pickup with zero navigation and retained joint references. A later wait/new request may recover after visibility returns; no automatic reset is implied. The observed front is not a complete obstacle map. The hand/table guard uses synchronized body and Dex3-1 finger encoders with observed tabletop/front planes. Missing or stale geometry reports `sensor_hand_clearance_unavailable`, stops acquisition before another policy action, and retains joint references with zero navigation. It covers the visible front half-plane only. Sensor stops use invalid/gapped measurements, estimated tilt above 15 degrees, negative observed local clearance, or motor output at least 95% of a configured effort limit for 0.5 seconds. These checks do not measure contact force and can miss collisions. A latched sensor fault ends the episode; unavailable geometry ends the operation and can recover after visibility returns. Geometry is required during pickup and hold. This path is under simulation qualification.
# Onboard RGB for generated verification code

`robot.observe_camera(camera='head')` is read-only. Permitted names are `head`,
`left_wrist`, `right_wrist`; cameras not enabled/captured return `unavailable`.
The opt-in wrist track provides wrist RGB only. `observe()` lists `camera_names`
when this track is enabled. No wrist depth or external overview is supplied.

An available reply includes original `time_s`/`step`, `age_s`, camera identity,
calibration ID and `calibration`, full-image RGB hash, thumbnail `width`/`height`,
`sample_stride_xy` and `rgb_base64`. Decode with standard-library
`base64.b64decode`: consecutive R,G,B bytes, rows then columns. A thumbnail
pixel `(u,v)` samples original pixel `(stride_x*u,stride_y*v)`. Thumbnails are
at most 48 by 32 pixels; they are coarse visual evidence, not precise geometry
or proof of grasp/support. Full-resolution onboard images are supplied in the
generation context when vision is enabled. Each image has its own capture time.

Repeated requests can return the same frame. Count distinct capture times,
not calls, when checking persistence. Frames older than 150 ms of simulation
time or from the future return `unavailable`. This is a simulation freshness
assumption; real processing/network delays are not calibrated. Check
`episode_status` and observation validity before any transition. Missing or
occluded evidence means unknown, not success. Wrist mounts are assumed, and
the new wrist path is under qualification. A generated verifier cannot override
motion guards or independent task scoring.

`calibration` contains only static camera intrinsics, original dimensions, robot
parent frame, mount translation in metres, XYZW quaternion and axis convention
when available. Intrinsics use original pixels; account for thumbnail strides.
These mounts are simulation assumptions. Camera-to-body transforms still require
timestamp-matched encoders/robot kinematics; no scene or object pose is supplied.

Each accepted `pickup_box` request starts with fresh policy memory and no pending
action chunk. This resets only the acquisition policy, never the physical episode.
A policy reset failure rejects the request as `acquisition_policy_reset_failed`;
it does not grant movement or erase earlier failures.

## Agent-written stage verification

For multi-stage programs define `verify_stage(stage, observations)` returning a
dictionary with `status` (`pass`, `fail`, `unknown`), `reason`, and
`evidence_times_s`. Advance only after both a completed tool outcome and a pass
from current permitted evidence. Pickup needs an uninterrupted fresh one-second
window of observed lift, opposing wrist proximity and stationary readiness.
Use distinct camera capture times, not body times or repeated calls; check
causality, age, session/track identity and the latest observation. During movement,
retention is distinct from stationary readiness. Unsupported release/support
verification stays unknown. Unknown allows only bounded sensing or supported
recovery; it is not success. Shared guards remain authoritative.

Print concise decisions when diagnostics are useful; the worker does not expose
an arbitrary `run()` return value as task completion. Neither returning nor
printing success changes independent scoring.

Sensor pickup now ends with `acquisition_grasp_attempt_lost` after two distinct images show an opposed-wrist lift of at least2cm, followed by0.3s of fresh evidence with no opposed wrists and gap at most5mm above the fresh observed under-box or tracked source-height plane. Changing reference type clears lift and loss evidence; each admitted pickup starts a new guard. A tracked source-height plane is refitted from current depth, never a remembered height. This marks a failed partial grasp, not success or measured contact. Pregrasp approach remains allowed; a new admitted pickup clears this attempt history and policy chunks while preserving physics. Thresholds are ideal-sensor simulation assumptions. This stop rule does not cover every failed pickup or certify unobserved travel space.

Destination geometry uses head RGB-D and synchronized measured up to select a visible near-level patch within one connected green color candidate. Table sides may share that color without becoming tabletop extent. Multiple level candidates, missing depth or enough unresolved geometry for another candidate return unavailable. The published patch remains partial; its bounds do not establish complete support, free space or safe placement. The5degree/3mm selection assumptions are uncalibrated simulation settings.


## Conditional remembered-front stopping (fresh qualification pending)

`SourceStopClearance` in `source_stop.py` extends the existing zero-navigation
stop screen. Each complete convex arm part must lie above a fresh observed
source top or in front of the original observed source front, with the declared
error allowance. It never combines different safe sides per vertex. A front
observation can survive visual occlusion for at most10s in the same uninterrupted
stance segment; motion age is at most150ms. First front loss freezes the anchor,
and later front-tracker reinitializations cannot replace it. A new source-tracker
object clears memory; source-identity or stance loss invalidates it.

This assumes a static source obstacle contained behind its observed front and
below its top, no protruding unseen source geometry, and10cm/2degree pose-error
allowances plus3mm/.5degree plane error. These are simulation sensitivity
assumptions, not hardware calibration or a complete collision map. There is no
second source-height reconstruction. Existing fresh grasp/vertical-motion
checks, arm-path3cm margin, paired wrist owner and zero navigation remain.
`loaded_stop` feedback labels `source_reference` and publishes remembered
observation age and assumed pose errors when that branch is available.

Recorded-state feasibility and13 adversarial/end-to-end tests pass; fresh
physical stopping under this extension is not yet qualified. It supplies no
forward-travel, box/body clearance, destination support or release permission.
A joint-reference screen is not a dynamic tracking bound.


## Unwired destination approach contract

`sensor_destination_approach.py` remains a draft with no public tool or live
factory. It now requires a callable path screen; there is no permissive default.
The callback must certify observed travel space, arm/body/box reference paths
and stopping for every proposed command. Its certificate must match the
camera capture. A destination plane or remembered source half-space cannot
supply this certificate alone.

The observer provides synchronized camera-time destination/grasp/stance data,
continuous view epoch and segment identity, plus a separate current-IMU
`navigation` record (`time_s`, `yaw_segment_rad`, `segment`). Segment directions
are rotated into current native navigation axes; held images never substitute
for current heading. Admission requires observed stationary readiness; motion
uses retained grasp. Wrist handoff stays at zero navigation and has a150ms
bound. Evidence/path/clock failure latches. Completion requires distinct fresh
observations over1s after zero navigation, retained readiness and observed
progress speed at most2cm/s; current command evidence is also required.

These are tested software conditions, not a qualified travel-space observer,
physical braking, placement or hardware capability. The controller is still
unavailable to agents pending those components and native qualification.

### Fresh source association at pickup admission

Sensor pickup validates a fresh strict under-box support and floor-parallel source candidate before resetting GR00T. Successful policy reset installs that source once and clears grasp dwell; acquisition commands retain it. Tracking continues to fit current depth, with no remembered-height fallback. Preparation or reset failure preserves the previous tracker and the physical episode.

An explicit pickup retry uses a separate fresh, single-image under-box/wrist check. While idle or waiting, fresh strict under-box support may rebuild front and hand-clearance observations after source loss; entering this recovery branch clears closing-rate history and requires two new camera images. Active acquisition/carry still uses its existing source. The failed source and loaded-stop memory remain invalid until explicit admission installs a new source. The real-estimator lifecycle regression passes; physical retry remains unqualified. This preview never reports settled readiness and cannot bypass retained opposed-wrist rejection. It does not repair an active operation or enable hold/carry. Only admitted pickup replaces source identity and resets the partial-grasp guard. Cached feedback retains the reference type that produced its measurement.

This tightens acquisition admission to the existing level-source model. Fresh floor/support loss rejects admission; no hidden geometry is used. [Integration evidence](../../runs/sensor-transfer-2026-09-19/early-source-integration-01/report.md) separates tests and recorded-input replay from future physical qualification.


`depart_source(yaw_rad)` backs away from the source until the requested turn has sensor-screened remembered-front clearance that does not depend on continued tabletop visibility, then settles the held box. Nonzero yaw is in radians, magnitude <=pi/2; sensors and existing turn margins are unchanged. `completed` means clearance and settled retention were continuously verified for1s, not that a prescribed distance was travelled. Inspect current `departure` evidence (`status=ready`, matching `yaw_rad`, fresh `time_s/age_s`, matching box `track_epoch` and scene `segment`, `retained/settled`, `clearance_lower_m>=required_margin_m`). Stop on unavailable evidence or a failed result. This tool does not execute the turn, cover unrelated obstacles, or prove destination placement. No automatic retry or episode reset is implied.

Departure translation preserves the existing live-front/hand checks while both are available. If either becomes unavailable, it requires a fresh current-command body/box/finger/paired-arm clearance check against admitted measured source geometry. Retreat direction uses that plane in the current floor-relative navigation frame. A fresh tabletop may support the current translation check; it cannot replace remembered-front clearance for the requested eventual turn. Missing or rejected travel evidence stops the stage. Source-only controlled-workspace scope does not certify unseen rear obstacles.

A briefly missing tabletop detection may use its original measured plane only within the existing150ms age bound and continuous source identity, floor and stance tracking. Stopping accounts for the full measurement age. This does not grant a turn or renew stale evidence.


## Isolated destination qualification snapshot

This snapshot alone exposes `move_with_box(distance_m=0.2)` to the existing worker queue so the already-integrated sensor destination controller can receive a bounded physical test. It is not a qualified public capability or a full-transfer success. Use only after verified pickup, retained settled hold, source departure, and a currently observed destination. The existing factory checks carry proof and owner continuity, destination view/segment, both-table geometry and combined stopping. Failure/unknown must stop the program; do not retry or call privileged placement. `place_box` remains unavailable. Head RGB-D/wrist RGB and truth isolation are unchanged. No tests here establish physical destination reachability or release.


## Batched departure computation candidate
Full-yaw source-departure clearance evaluates the same sampled headings in one NumPy batch per convex part. Each heading still chooses a single whole top/front bound; source-reference order, uncertainty, sample density, margins and per-command checks remain unchanged. PreparedSourceSides.bounds remains the scalar reference. No observation is retained across commands by this optimization. Local recorded-input numerical/timing tests do not qualify physical approach or release.
