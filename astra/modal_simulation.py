"""Shared Modal simulator image and episode lifecycle for Jev and OpenAI Decisions."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

WORKSPACE = "/workspace/humanoid-astra/astra"
PYTHON = "/workspace/SIMPLE/.venv/bin/python"


def run_simulation(client, seeds, max_actions, duration_s, min_confidence, carry_back, record_video, streaming,
                   *, provider, runs):
    sys.path.insert(0, WORKSPACE)
    from run_jev import run_episode
    from sim_client import SimClient

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out = Path("/runs") / provider / run_id
    out.mkdir(parents=True, exist_ok=True)
    server_log = (out / "sim_server.log").open("w")
    server = subprocess.Popen([PYTHON, f"{WORKSPACE}/sim_server_simple.py", "--port", "8765", "--sim-mode", "mujoco"],
                              cwd=WORKSPACE, env={**os.environ, "MUJOCO_GL": "egl"}, stdout=server_log,
                              stderr=subprocess.STDOUT)
    try:
        client.warmup()
        deadline = time.monotonic() + 15 * 60
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"SIMPLE server exited early with code {server.returncode}; see {out / 'sim_server.log'}")
            try:
                with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=2):
                    break
            except OSError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(f"SIMPLE server did not become healthy within 15 minutes; see {out / 'sim_server.log'}")
                time.sleep(2)
        sim = SimClient("http://127.0.0.1:8765", timeout=60)
        (out / "health.json").write_text(json.dumps(sim.health(), indent=2))
        results = [run_episode(client, sim, seed, max_actions, duration_s, min_confidence, out,
                               carry_back=carry_back,
                               record_dir=(out / f"seed_{seed}_frames") if record_video else None,
                               streaming=streaming) for seed in seeds]
        (out / "summary.json").write_text(json.dumps(results, indent=2))
        runs.commit()
        return {"run_id": run_id, "summary": results, "volume_path": str(out)}
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        server_log.close()
        runs.commit()
