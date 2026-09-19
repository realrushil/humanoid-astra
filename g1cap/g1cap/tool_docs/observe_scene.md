# observe_scene

`robot.observe_scene()`

**Purpose and inputs:** Read named scene geometry and current free-object simulator state.

**Preconditions:** Requires a scene-enabled task.

**Feedback and execution:** No motor command. Returns box initial world centres, half sizes, mass, friction, markers and sample/session timestamps. Mass zero denotes fixed geometry; positive kg denotes a free body. `objects[name]` carries the current world position, WXYZ quaternion and world linear/angular velocity from the same sample. `contacts` records body pairs, distance (m) and normal force (N). Static geometry is exact by assumption.

**Result:** Scene dictionary; rejected/scene_unavailable on legacy open-floor recipes.

**Side effects and continuation:** No route or grasp planning. Initial geometry stays fixed; measured free-object state changes with physics. Open-floor mobility returns an empty scene.

**Evidence and limits:** Values are simulator ground truth, not vision or a deployed object-state estimator. The box fixture is a physics foundation; no grasp/contact tool or pickup benchmark is admitted. Robot–box contacts still trigger the existing collision rule. Box–table support does not count as robot support or a robot collision.

See [shared assumptions](README.md).

Each contact also provides `geoms`, two dictionaries containing model-local `id`, geometry `name`, owning `body`, and mesh asset `mesh` (or null for primitive geometry). A shared body name does not imply the same permitted contact surface. These fields identify measured contacts; they do not permit grasping or override collision rules.
