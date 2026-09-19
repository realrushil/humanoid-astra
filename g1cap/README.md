# G1 code-as-policy: colleague handoff

A research implementation that lets an ordinary coding agent write and revise Python programs for a **simulated Unitree G1**. Programs call bounded robot tools; a trusted runtime owns physics, controller execution, recording and independent scoring. Revisions share one live episode, including elapsed time and previous action consequences.

This folder is a portable source snapshot of the CAP implementation, prepared on **2026-09-18**. It contains our Python code, calibration assets, tests, examples and selected video evidence. It sits alongside the existing `humanoid-astra` server/client implementation and has no integration dependency on those sibling files. Python module names and controller code are preserved.

## What currently works

| Path | Implementation | Evidence and limits |
| --- | --- | --- |
| Synthetic mock | Local Python backend for harness and worker checks | No physics; useful for software plumbing only |
| SONIC | MuJoCo + released SONIC controller/planner | Walking, posture and contact-free reaching; simulator-derived observations |
| Arena privileged baseline | Isaac Lab/Arena + GR00T acquisition + HOMIE + bounded manipulation controllers | Earlier nominal transfer gate passed 3/3; 6/10 task definitions passed the frozen sweep/retry. Not a sensor-only result |
| Arena sensor development | RGB-D, encoder/IMU estimates, visual grasp checks and coupled wrist holding | One developer-written pickup → raise → retreat → turn → hold sequence completed. Full sensor-based destination transfer remains unfinished |

Watch the [successful sensor component sequence](evidence/sensor-sequence.mp4) and the [later pickup failure](evidence/pickup-failure.mp4). Both are **previously recorded physics**, not new runs or reconstructed motion. [Evidence notes](evidence/README.md) explain their scope. A tool returning `completed` does not establish whole-task success.

The sensor path still needs robust source-loss handling, active destination viewing, forward transport/placement and fresh coding-agent/benchmark qualification. It has no demonstrated final-contract full transfer. All results are simulation results; timing, sensing, contact and calibration assumptions are not hardware qualification.

## Start locally

Run commands from this folder, the directory containing this README and `pyproject.toml`:

```sh
cd humanoid-astra/g1cap
python3 --version  # Python 3.10 or newer
python3 -m g1cap --help
python3 -m g1cap.interactive --help
python3 -m g1cap demo --out runs/mock-demo-001
```

Use a new output directory for each run. The mock demo executes saved programs without a model call. The standard-library harness can run directly from the checkout. Install the analysis dependencies below before running the complete test suite: a few sensor tests import NumPy/SciPy directly. Other scientific/native tests skip when their optional dependencies are unavailable. Worker tests require macOS Seatbelt or Linux Bubblewrap with working user namespaces. Nested application sandboxes can prevent worker launch; the worker fails closed.

Install an editable copy and the analysis dependencies, then run the tests:

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e '.[analysis]'
python3 -m unittest discover -s tests -v
```

The `analysis` extra provides NumPy, SciPy and image utilities. Pinocchio, MuJoCo, Torch, Isaac Lab/Arena, controller models and their native dependencies require the matching backend environment; this extra does not install a robot simulator. Continue to launch interactive sessions from this source root: source upload and some asset paths are checkout-relative.

## Run a live simulation session

First read [runtime setup](docs/setup.md). A configured Linux/GPU runtime and SSH access are required. Replace these example environment values with your own prepared machine and absolute runtime root:

```sh
export G1_HOST=user@linux-host
export G1_REMOTE_ROOT=/absolute/path/to/prepared-runtime
export G1_GPU=0
ssh "$G1_HOST" nvidia-smi

python3 -m g1cap.interactive \
  --ssh-host "$G1_HOST" --remote-root "$G1_REMOTE_ROOT" --gpu "$G1_GPU" \
  --recipe examples/arena-sensor-sequence.json \
  --policy evidence/sensor-sequence-policy.py --max-rounds 1 \
  --out runs/sensor-sequence-001
```

This submits the recorded developer-written program to **new physics**; success is not guaranteed. The recipe uses `model_gpu: 0`; select a suitable model GPU before launch. With rendering and inference on the same GPU, the Arena launcher's current admission check requires at least 13,000 MiB free. Inspect utilization and memory, keep runs bounded and leave other users' jobs untouched.

For an ordinary coding-agent session, omit `--policy` and use `--max-rounds 2 --generation-timeout 180`. Default settings are `gpt-5.6-luna`, low reasoning, standard speed and saved ChatGPT authentication. The installed Codex CLI must support those requested settings. There is no silent model upgrade. `--vision-mode direct` optionally supplies Arena images; sensor mode supplies onboard head RGB, while privileged mode can include an external overview. Do not provide the human-facing evidence folder as benchmark solution context.

For the separate SONIC workstation path, use `--recipe examples/workstation-upper.json` and a SONIC runtime root. A handwritten example is `--policy examples/workstation_upper_reference.py`. SONIC is simulation-only, using the dedicated loopback build and DDS domain 1.

## Understand and edit the code

Read [the package guide](g1cap/README.md), then [the execution walkthrough](docs/code-walkthrough.md). Exact agent-facing contracts are [Arena sensor tools](g1cap/tool_docs/arena_sensor.md), [Arena privileged tools](g1cap/tool_docs/arena_box.md), and [SONIC tools](g1cap/tool_docs/README.md).

```text
g1cap/                 Python implementation; import name stays g1cap
  toolkit/             Bounded robot tools, geometry and controller adapters
  tool_docs/           Contracts copied into the coding-agent workspace
  assets/              Static robot kinematics, geometry and mass calibration
tests/                 Unit/contract tests and small fixtures
examples/              Recipes, handwritten SONIC policies and ten-scene suite
scripts/               Existing SONIC setup/download/build helpers
docs/                  Setup, architecture and collaboration notes
evidence/              Two labeled historical videos and a witness program
SOURCE_MANIFEST.json   Original paths and hashes for copied files
```

The two levels named `g1cap` are intentional: the outer folder is the standalone project, and the inner folder is the importable package. Start with `interactive.py`, `arena_session_runtime.py`, `arena_control.py` and `arena_world.py`; individual estimation/control files can be read as needed.

Units are metres, seconds, radians, kilograms and newtons unless a field explicitly says otherwise. Persistent SONIC uses MuJoCo world coordinates; older single-turn tools use a frozen episode frame. Sensor Arena uses camera and locally estimated floor/heading or stance frames, not global localization. Read the relevant contract before reusing a vector between paths.

## Results and collaboration

Arena runs collect exact submitted programs, tool traces, state/action/contact records, GPU accounting and videos under the chosen run directory. The labeled output is normally `remote/artifacts/video/full.mp4`; raw cameras remain under `remote/artifacts/physics/`. Exporting video replays recorded frames without stepping physics. Use `python3 -m g1cap.arena_video /absolute/path/to/artifacts` to re-export. Keep failure recordings as well as successes.

Before sharing a change, run the test command above, record skips and environment details, and update the affected tool contract and reading guide. Keep probes in `temp/` and new results in `runs/`; both are ignored here. [Collaboration notes](docs/collaboration.md) describe snapshot ownership and dependency boundaries. [Verification results](docs/verification.md) record what was actually checked for this delivery.
