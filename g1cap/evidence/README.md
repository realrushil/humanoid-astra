# Selected historical physical evidence

These videos were copied from existing recordings. Packaging created no simulation episodes and made no model-generation calls. Videos retain task/action/outcome labels. They are for human review, not agent benchmark context.

| Recording | Result | What it does not establish |
| --- | --- | --- |
| [Sensor sequence](sensor-sequence.mp4) | Attempt 48: pickup, raise to 8 cm observed clearance, 20 cm retreat, relative −30° turn and hold completed; stable hold persisted 7.06 s after turn return | No destination placement, agent-generated solution, full benchmark or hardware qualification |
| [Pickup failure](pickup-failure.mp4) | Later view-change probe 02 stopped during pickup with `hold_visual_retention_lost`, before either planned turn | No conclusion about destination visibility after turning |

The successful sequence ends at 32.60 simulated seconds, with 8.372 cm true source-plane clearance. Final yaw was −27.818° against −30° requested; turning also translated the root about 9.87 cm. These are evaluator measurements, not all controller-visible observations. Runtime was about 953 wall seconds; this was not real-time execution.

[sensor-sequence-policy.py](sensor-sequence-policy.py) is the exact developer-written witness program. [The shared recipe](../examples/arena-sensor-sequence.json) preserves physical/settings fields but changes the host-specific model GPU from 1 to 0. Running them again produces new physics, not a replay guarantee.

Original CAP archive paths:

- `runs/coupled-transfer-001/integrated-coupled-stop-02/report.md`, `fresh-sequence.mp4`, `policy.py`, `recipe.json`, `audits/results.json`.
- `runs/coupled-transfer-001/view-change-probe-02/report.md`, `remote/artifacts/video/full.mp4`.
- Privileged baseline results: `runs/transfer-workflow-001/evaluation.md` (3/3 nominal gate, 6/10 task definitions in the sweep/retry).

Full state/action/sensor archives and native audits remain in CAP. Their absence here means this small handoff is not the full reproducibility archive. `SOURCE_MANIFEST.json` preserves hashes and original paths for the included videos and program.
