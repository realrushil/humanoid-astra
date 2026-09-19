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
tactile proof. `settled_retention` checks the one-second grasp/scene motion window independently of height. `ready` also requires a scene-motion window. `scene_motion`
uses local stance segments under a no-slip hypothesis. Missing/stale motion
has no body transform; never bridge segment IDs or assume global localization.
Unknown/rejected observations must not be filled with remembered box sizes.

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

After the first retreat, `grasp.gap_reference` becomes `source_height_plane`:
the gap is height above the observed original table plane, even outside its
footprint. It does not mean a table supports the box. Source-plane tracking
requires fresh floor and source depth, including at initial carry admission.
The model assumes the source surface is parallel to the observed floor;
incompatible tilt, ambiguous association or poorly constrained height rejects
the estimate. Its ±3 mm normal-point perturbation assumption is a conditional
fit-sensitivity check, not calibrated camera/pose uncertainty. It cannot
distinguish coplanar tables. A missing floor/control-frame segment cannot
silently establish a new source identity.
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

When vision is enabled, only onboard head RGB images are attached; structured
geometry is computed from synchronized RGB-D. External views and evaluator
videos are human evidence, not coding-agent inputs. Pixels do not establish
unseen geometry, exact force, or success. No hardware qualification is claimed.

Pickup navigation uses a visible table-front estimate and body encoders. It requires two consecutive camera measurements and expires after 150 ms. `sensor_approach_unavailable` rejects or stops pickup with zero navigation and retained joint references. A later wait/new request may recover after visibility returns; no automatic reset is implied. The observed front is not a complete obstacle map. The hand/table guard uses synchronized body and Dex3-1 finger encoders with observed tabletop/front planes. Missing or stale geometry reports `sensor_hand_clearance_unavailable`, stops acquisition before another policy action, and retains joint references with zero navigation. It covers the visible front half-plane only. Sensor stops use invalid/gapped measurements, estimated tilt above 15 degrees, negative observed local clearance, or motor output at least 95% of a configured effort limit for 0.5 seconds. These checks do not measure contact force and can miss collisions. A latched sensor fault ends the episode; unavailable geometry ends the operation and can recover after visibility returns. Geometry is required during pickup and hold. This path is under simulation qualification.
