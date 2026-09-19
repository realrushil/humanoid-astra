# Runtime setup and portability

The checkout contains our Python implementation. It is not a complete, redistributable native simulator environment. No checkpoints, credentials, virtual environments, third-party repositories or private host configuration are included. No clean-machine GPU installation was performed for this handoff.

## Local harness

Python 3.10+ runs the standard-library harness and synthetic mock from the source root. `pip install -e '.[analysis]'` adds common numerical/image dependencies. The full suite requires NumPy/SciPy because some tests import them directly. Other scientific tests additionally need the robotics Pinocchio binding; other tests need MuJoCo or installed native assets. Missing dependencies produce skips, so a green lightweight suite is not backend qualification.

macOS workers use `sandbox-exec`; Linux workers require `bwrap` and user namespaces. There is no unsandboxed fallback. Interactive execution uses `ssh`/`scp`. Model generation uses saved ChatGPT authentication through the Codex CLI. Native environments should retain their known-compatible package versions rather than be replaced by the optional local extra.

## Arena runtime expected by the launcher

Under the absolute `--remote-root`, prepare this layout with the project owner's compatible native environment:

```text
remote-root/
  runs/                         writable session outputs (must exist)
  temp/
    Arena/                      matching Arena checkout and submodules
      submodules/Isaac-GR00T/
      isaaclab_arena_gr00t/embodiments/g1/g1_sim_wbc_data_config.py
    arena-venv/bin/python       Isaac Lab/Arena simulation environment
    gr00t-venv/bin/python       compatible GR00T inference environment
    arena-model/               matching acquisition model checkpoint
    hf-cache/                  cache location
    arena-runtime-cache/       native runtime cache
```

Native requirements include NVIDIA GPU/driver, Isaac Lab/Isaac Sim, Arena's G1 embodiment and assets, Torch, GR00T, HOMIE and the compatible wrist/kinematics dependencies. The authoritative launch paths and commands are in `g1cap/arena_launch.py` and imports in `g1cap/arena_world.py`. Obtain the matching upstream revisions, local modifications, model files and licenses from the environment owner before reproducing on a new host; this snapshot does not pin that external stack.

The launcher binds inference to loopback, rejects an occupied model port, checks free GPU memory, records allocation and cleans up only its own process groups. It reserves admission headroom of 9,000 MiB for inference and 4,000 MiB for rendering (13,000 MiB if shared). These are checks, not measured memory caps. `model_gpu` comes from the recipe, rendering GPU from `--gpu`.

Arena uploads our local `g1cap/` source per session. It does not upload native dependencies or repair an incompatible native environment. Run from this project's root and create the remote `runs/` parent first. CPU video export requires FFmpeg, Pillow and available fonts; see `g1cap/arena_video.py` for runtime lookup.

## SONIC runtime

SONIC uses a separate prepared root containing matching `g1cap/`, `.venv-sonic/`, the `deps/GR00T-WholeBodyControl` checkout, dedicated loopback build, model files and robot assets. The retained `scripts/` helpers document bootstrap, download and build steps. They are environment-specific preparation tools, not an all-in-one fresh-machine installer; inspect prerequisites and upstream licenses before running.

MuJoCo, the released SONIC controller/planner, Bubblewrap and controller transport dependencies must be available. Rendering additionally requires FFmpeg, Pillow, PyOpenGL and Mesa EGL. Keep DDS domain 1 and loopback-only networking. Physical robot deployment is outside this implementation's scope.

## Reproduction boundary

Preserving Python source hashes establishes which wrapper/controller code was shared. It does not establish identical upstream models, simulator versions, startup timing or native patches. Save those versions, recipe, exact program, source overlay manifest, GPU allocation and videos with each new trial. Do not describe a new asynchronous run as deterministic replay of a past success.
