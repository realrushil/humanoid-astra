"""Modal entry point for the image-based OpenAI Decisions controller. See DECISIONS.md."""
import json
from pathlib import Path

import modal

app = modal.App("humanoid-astra-decisions")
ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = "/workspace/humanoid-astra/astra"
image = modal.Image.from_dockerfile(Path(__file__).with_name("Dockerfile.modal"), context_dir=ROOT,
                                  add_python="3.10").add_local_dir(ROOT / "astra", remote_path=WORKSPACE, copy=False)
runs = modal.Volume.from_name("humanoid-astra-runs", create_if_missing=True)
# Only the small gateway receives the secret. Mount just the pure-Python client dependencies there.
gateway_image = modal.Image.debian_slim(python_version="3.12")
for filename in ("run_decisions.py", "run_jev.py", "sim_client.py"):
    gateway_image = gateway_image.add_local_file(Path(__file__).with_name(filename),
                                                remote_path=f"/controller/{filename}", copy=False)
openai_secret = modal.Secret.from_name("openai-decisions-secrets", required_keys=["OPENAI_API_KEY"])


@app.function(image=gateway_image, secrets=[openai_secret], timeout=60, max_containers=1, scaledown_window=120)
def decide_with_openai(payload: dict):
    import os
    import sys
    sys.path.insert(0, "/controller")
    from run_decisions import post_decision

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is absent from the openai-decisions-secrets Modal Secret")
    body, latency = post_decision(payload, key)
    return {"body": body, "latency_s": latency}


@app.function(image=image, gpu="L4", timeout=60 * 60, volumes={"/runs": runs})
def run_experiment(seeds: list[int], max_actions: int = 25, duration_s: float = 0.2, min_confidence: float = 0.0,
                   carry_back: bool = False, record_video: bool = True, streaming: bool = True,
                   model: str = "gpt-6-luna", image_detail: str = "auto", cameras: str = "head,third_person",
                   single_request: bool = False):
    import sys
    sys.path.insert(0, WORKSPACE)
    from run_decisions import DecisionsClient
    from modal_simulation import run_simulation

    def transport(payload):
        reply = decide_with_openai.remote(payload)
        return reply["body"], reply["latency_s"]

    client = DecisionsClient(model, transport=transport, image_detail=image_detail, cameras=cameras,
                             single_request=single_request)
    return run_simulation(client, seeds, max_actions, duration_s, min_confidence,
                          carry_back, record_video, streaming, provider="decisions", runs=runs)


@app.local_entrypoint()
def main(seeds: str = "0", max_actions: int = 25, duration_s: float = 0.2, min_confidence: float = 0.0,
         carry_back: bool = False, record_video: bool = True, streaming: bool = True,
         model: str = "gpt-6-luna", image_detail: str = "auto", cameras: str = "head,third_person",
         single_request: bool = False):
    parsed = [int(seed) for seed in seeds.split(",") if seed.strip()]
    if not parsed:
        raise ValueError("--seeds needs one or more comma-separated integer seeds")
    if max_actions < 1 or not 0.04 <= duration_s <= 0.5:
        raise ValueError("Use a positive action budget and duration between 0.04 and 0.50 seconds")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min-confidence must be between zero and one")
    if image_detail not in {"auto", "low", "high", "original"} or cameras not in {"head", "third_person", "head,third_person"}:
        raise ValueError("Invalid image detail or camera selection")
    print(json.dumps(run_experiment.remote(parsed, max_actions, duration_s, min_confidence,
                                            carry_back, record_video, streaming, model, image_detail, cameras,
                                            single_request), indent=2))
