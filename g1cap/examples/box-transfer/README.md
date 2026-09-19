# Box transfer v1: ten scenarios, one activity

Pick up the brown box with both hands, carry it from the gray table to the green table, place the whole box on the destination and release both hands, finishing in stable standing.

These are **ten instances of one task family**, not ten distinct manipulation skills. The suite stays within rigid two-hand cuboid transport. Arbitrary household objects, search, semantic object selection, furniture interaction and hardware deployment are outside this version.

| Recipe | Change from nominal | Role |
|---|---|---|
| [01-nominal.json](01-nominal.json) |20cm/0.1kg cube, equal-height desks | Development |
| [02-box-offset.json](02-box-offset.json) | Box center6cm farther along world+Y | Validation |
| [03-box-rotated.json](03-box-rotated.json) | Initial box yaw offset+25° | Validation |
| [04-opposite-side.json](04-opposite-side.json) | Destination reflected across robot's initial Y=0.18m line | Validation |
| [05-longer-carry.json](05-longer-carry.json) | Destination1m farther along world−Y | Validation |
| [06-higher-table.json](06-higher-table.json) | Destination surface5cm higher | Validation |
| [07-smaller-table.json](07-smaller-table.json) | Destination70×60cm instead of110×80cm | Validation |
| [08-rectangular-box.json](08-rectangular-box.json) |20×24×16cm cuboid, same0.1kg mass | Validation |
| [09-heavier-box.json](09-heavier-box.json) | Same20cm cube, mass0.5kg | Validation |
| [10-combined.json](10-combined.json) |24×24×20cm/0.5kg parcel, shifted/rotated start, shifted/higher/smaller destination | Validation |

Each recipe is complete and uses the existing Arena execution path. Configuration validity is not proof of controller capability. Per-case status, all attempts and the schematic video tour are in the original CAP archive `runs/box-transfer-benchmark-001/report.md` (not bundled; see the shared project README for outcome limits).

## Geometry and information

Positions/full dimensions are metres, angles radians, mass kilograms. World Z is up; floor Z=-0.795m and initial root=(0,0.18,0). Nominal tabletop Z=-0.03m is76.5cm above the floor. Box center sampling is±1cm per XY axis. Seeds42/43/44 and GR00T seeds0/1/2 define three repeat settings, changing startup sampling and policy randomness; they are not extra task families.

Nominal box center is(0.60,0.18), not the historical trial68 reset sample. Its±1cm range also differs from the old±2.5cm reset. Historical success is evidence for the previous fixture, not an automatic witness for case01. The full rotated box and its reset range must fit its source. Validation checks static geometry/standing clearance, not grasp reachability or dynamic feasibility.

The native textured box is physically scaled before construction. A short spawn wrapper authors MassAPI on the single nested rigid body, which has no original mass schema. Writing mass to the asset container or using a modify-only API leaves density-derived mass unchanged. Runtime verifies actual collision dimensions and PhysX mass before any action, and records inertia. Desks remain axis-aligned and fixed. Revisions never reset physics or alter physical parameters.

The observation track is **simulator state plus head/overview images**. The overview is external; numeric pose/contact measurements are privileged. Dimensions/mass are in `task` and `robot.observe()`. No case-specific route, heading, grasp pose or recovery recipe is supplied. GR00T pickup conditioning and tool limits remain unchanged. New scene ranges are testing assumptions, not qualified operating ranges.

## Outcomes

All ten instances share an independent evaluator: first a stable raised bilateral hold, then at least0.5m box displacement in XY, followed by the entire oriented footprint inside the destination with actual support≥half the configured weight. Support, hand force≤0.5N each, stable standing, box speed≤0.05m/s and angular speed≤0.2rad/s must coincide for one second. Table contact>5N, box-floor contact>0.5N, excessive tilt and missing/reordered observations remain failures. These thresholds are simulation assumptions.

Report lift, actual displacement, supported release, physical faults, tool outcomes, simulation/wall time, revisions, source hashes and generation usage. Tool/program completion is not task success; partial lift/carry is diagnostic progress only. Distinguish fixture-invalid, startup-failed, tool-failed, program-failed, timeout, pass and not-run. Keep valid unsolved cases. For this suite, solver witnesses are separate from fixture admission; the older witness-only admission rule does not apply. Historical mobility labels remain unchanged.

## Development loop

1. Develop primarily on01. Extra controller probes need a stated failure mechanism and stay outside benchmark scores.
2. Initial sweep gate: three consecutive full transfers on the three declared seed pairs, with no physical faults. This is a regression gate, not statistical reliability.
3. Freeze recipes, toolkit, scoring, prompt/model/reasoning/vision and budgets. Run one episode per case, at most three generated programs per episode. Revisions retain consequences. No developer edits or case-specific tool changes during a sweep.
4. Diagnose acquisition, retention/balance, routing, placement/release and coding failures. Improve shared capabilities, then run a new frozen sweep. Validation feedback is allowed and disclosed; these cases are not permanently unseen tests.
5. If useful, repeat the frozen suite across all three seed pairs. Report every attempt, not the best result. Scientific held-out testing will require separate scenes later.

`suite.json` indexes recipes and repeat settings. Pass an individual case JSON to `python3 -m g1cap.interactive --recipe`, not the index. Existing defaults are Luna/low. Stronger settings follow the project's evidence-based escalation rule and remain fixed within a sweep. Fixture checks need zero coding-agent calls. Existing SONIC mobility tasks remain separate diagnostics, not additional tasks in this ten-case transfer suite.

## Research grounding

[LIBERO, section4.2](https://proceedings.neurips.cc/paper_files/paper/2023/file/8c3c666820ea055a77726d66fc7d447f-Paper-Datasets_and_Benchmarks.pdf) motivates separating spatial/object/goal changes. [RoboCasa365, sections3.2–3.3](https://arxiv.org/html/2603.04356v1) distinguishes scene diversity from atomic/composite activities. [BEHAVIOR-1K, section6.1](https://arxiv.org/html/2403.09227v1) motivates completion/cost metrics and explicit primitive assumptions; its assisted-grasp/direct-state-setting baselines are not imported. These are design adaptations, not equivalent scores or inherited capabilities.
