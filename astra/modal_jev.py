"""Launch the SIMPLE + intent-first Jev experiment on Modal.

The simulator container never receives the TypeSafe API key. A separate small gateway function owns the Modal secret
and returns structured choices only; reports are committed to the ``humanoid-astra-runs`` Modal Volume.
"""
import json
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent.parent
app = modal.App("humanoid-astra-jev")
image = modal.Image.from_dockerfile(Path(__file__).with_name("Dockerfile.modal"), context_dir=ROOT, add_python="3.10").add_local_dir(
    ROOT / "astra", remote_path="/workspace/humanoid-astra/astra", copy=False)
runs = modal.Volume.from_name("humanoid-astra-runs", create_if_missing=True)
jev_secret = modal.Secret.from_name("jev-secrets")
PYTHON = "/workspace/SIMPLE/.venv/bin/python"
WORKSPACE = "/workspace/humanoid-astra/astra"
API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"


@app.function(image=modal.Image.debian_slim(python_version="3.12"), secrets=[jev_secret], timeout=60,
              max_containers=1, scaledown_window=120)
def decide_with_jev(state: dict, questions: dict):
    """The sole function permitted to read TYPESAFE_API_KEY."""
    import json
    import os
    import time
    import urllib.error
    import urllib.request

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is absent from the jev-secrets Modal Secret")
    payload = {"model": MODEL, "state": json.dumps(state, separators=(",", ":")), "questions": questions}
    request = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Jev API returned HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Jev API request failed: {exc.reason}") from None
    return {"answers": body["answers"], "usage": body.get("usage", {}), "latency_s": time.monotonic() - started}


@app.function(image=image, gpu="L4", timeout=60 * 60, volumes={"/runs": runs})
def run_experiment(seeds: list[int], max_actions: int = 50, duration_s: float = 0.2, min_confidence: float = 0.0,
                   carry_back: bool = False, record_video: bool = False, streaming: bool = False):
    """Run headless MuJoCo with the bounded intent-first controller and persist every transcript."""
    import os
    import subprocess
    import sys
    import time
    import urllib.request

    sys.path.insert(0, WORKSPACE)
    from run_jev import (AXIS_CRITERIA, BODY_CRITERIA, GRIPPER_CRITERIA, INTENT_ACTION_CONTEXT, INTENT_CRITERIA,
                         INTENT_INSTRUCTIONS, axis_field_instructions, run_episode)
    from sim_client import SimClient

    class GatewayJevClient:
        model = MODEL

        @staticmethod
        def warmup():
            """Pay the gateway/container cold start before the episode clock begins."""
            decide_with_jev.remote({"task": "Connection warm-up only."}, {
                "choice": {"type": "choice", "instructions": "Acknowledge the connection.",
                           "criteria": {"ready": "The connection is ready."}},
            })

        @staticmethod
        def decide_intent_axis(state):
            intent_request = {"choice": {"type": "choice", "instructions": INTENT_INSTRUCTIONS,
                                          "criteria": INTENT_CRITERIA}}
            intent_reply = decide_with_jev.remote(state, intent_request)
            intent_answer = intent_reply["answers"]["choice"]
            intent = intent_answer["choice"]
            if intent not in INTENT_CRITERIA:
                raise RuntimeError(f"Jev returned unknown intent {intent!r}")
            action_state = {**state, "selected_intent": intent, "selected_intent_description": INTENT_CRITERIA[intent],
                            "selected_intent_action_context": INTENT_ACTION_CONTEXT[intent]}
            questions = {
                "body_action": {"type": "choice", "instructions": axis_field_instructions("body_action", intent), "criteria": BODY_CRITERIA},
                "right_x": {"type": "choice", "instructions": axis_field_instructions("right_x", intent), "criteria": AXIS_CRITERIA},
                "right_y": {"type": "choice", "instructions": axis_field_instructions("right_y", intent), "criteria": AXIS_CRITERIA},
                "right_z": {"type": "choice", "instructions": axis_field_instructions("right_z", intent), "criteria": AXIS_CRITERIA},
                "gripper": {"type": "choice", "instructions": axis_field_instructions("gripper", intent), "criteria": GRIPPER_CRITERIA},
            }
            action_reply = decide_with_jev.remote(action_state, questions)
            return (intent_answer, action_reply["answers"],
                    {"intent": intent_reply["usage"], "action": action_reply["usage"]},
                    intent_reply["latency_s"] + action_reply["latency_s"])

    run_id = time.strftime("%Y%m%d_%H%M%S")
    out = Path("/runs/jev") / run_id
    out.mkdir(parents=True, exist_ok=True)
    server_log = (out / "sim_server.log").open("w")
    server = subprocess.Popen([PYTHON, f"{WORKSPACE}/sim_server_simple.py", "--port", "8765", "--sim-mode", "mujoco"],
                              cwd=WORKSPACE, env={**os.environ, "MUJOCO_GL": "egl"}, stdout=server_log,
                              stderr=subprocess.STDOUT)
    try:
        GatewayJevClient.warmup()
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
        results = [run_episode(GatewayJevClient(), sim, seed, max_actions, duration_s, min_confidence, out,
                               carry_back=carry_back,
                               record_dir=(out / f"seed_{seed}_frames") if record_video else None,
                               streaming=streaming)
                   for seed in seeds]
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
        server_log.close()
        runs.commit()


@app.local_entrypoint()
def main(seeds: str = "0", max_actions: int = 50, duration_s: float = 0.2, min_confidence: float = 0.0,
         carry_back: bool = False, record_video: bool = False, streaming: bool = False):
    parsed = [int(seed) for seed in seeds.split(",") if seed.strip()]
    if not parsed:
        raise ValueError("--seeds needs one or more comma-separated integer seeds")
    print(json.dumps(run_experiment.remote(parsed, max_actions, duration_s, min_confidence, carry_back, record_video,
                                            streaming), indent=2))
