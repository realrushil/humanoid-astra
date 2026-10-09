"""OpenAI Decisions controller with head and third-person images plus structured state.

Uses the existing bounded intent/axis controller and streaming simulator loop. No SDK dependency is required.
API contract: https://developers.openai.com/api/docs/guides/decisions
"""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from run_jev import (AXIS_CRITERIA, BODY_CRITERIA, GRIPPER_CRITERIA,
                     INTENT_CRITERIA, JevClient, ROOT,
                     compact_state, logged_decision_state, run_episode)
from sim_client import SimClient

API_URL = "https://api.openai.com/v1/decisions"
MODEL = "gpt-6-luna"
VISUAL_CONTEXT = (
    "You control a Unitree G1 humanoid with a right Dex3 hand, IK arm control and AMO locomotion. "
    "Use the supplied camera views together with measured geometry/contact feedback. Images and state describe "
    "the same simulation instant. Head is egocentric; third_person is an external camera. "
    "Directions are pelvis-relative, not image-pixel directions. The pinch-ready flag is contact evidence, "
    "not a guarantee of a stable grasp."
)
MODEL_STATE_KEYS = (
    "task", "task_goal", "selected_intent",
    "box_pelvis_m", "box_lift_m", "right_grasp_error_m",
    "right_grasp_distance_m", "right_openness", "right_any_target_contact", "right_contact_forces",
    "right_pinch", "body_height_m",
)
AXIS_DIRECTION_CONTEXT = (
    "Use signed pelvis-frame grasp error: positive needs plus, negative needs minus; negative z means the box is below.")
DECISIONS_INTENT_INSTRUCTIONS = (
    "Choose exactly one current task intent for the next whole-body control tick. Use the current task goal, "
    "geometry, pinch/contact evidence, and body posture. Do not invent an intent.")


def model_input_state(state):
    """Keep only state that can change a visual decision or disambiguate its action effect.

    The complete state remains in the local transcript. Camera manifests, absolute odometry, wrist RPY,
    duplicate history fields, and streaming implementation details are bookkeeping rather than evidence.
    """
    logged = logged_decision_state(state)
    compact = {key: logged[key] for key in MODEL_STATE_KEYS if key in logged}
    return compact
SINGLE_REQUEST_INTENT_INSTRUCTIONS = (
    "Choose the task intent and all five control fields together for one simultaneous 0.2-second whole-body tick. "
    "Base every field on the same current geometry, contacts, and current contact evidence. Do not invent an intent.")


def single_request_axis_instructions(field):
    """Prompt one independently-returned field to be coherent with the jointly selected intent."""
    specific = {
        "body_action": "You are answering ONLY body_action: crouch/stand/base/hold. It does not move the hand axes.",
        "right_x": "You are answering ONLY right_x: minus moves the hand backward, plus moves it forward, hold leaves x unchanged.",
        "right_y": "You are answering ONLY right_y: minus moves the hand right, plus moves it left, hold leaves y unchanged.",
        "right_z": "You are answering ONLY right_z: minus moves the hand down, plus moves it up, hold leaves z unchanged.",
        "gripper": "You are answering ONLY gripper: it opens, holds, or closes the fingers and does not move the hand.",
    }
    return ("Choose this field jointly and coherently with the intent answer for the same simultaneous control tick. "
            + AXIS_DIRECTION_CONTEXT +
            " Infer the appropriate intent directly from the shared current state: approach keeps the gripper open "
            "and reduces signed error; grasp closes only near the object and corrects a failed thumb-versus-fingers "
            "pinch; lift/carry keep a verified pinch closed; recover reverses worsening motion. " + specific[field])


def two_stage_axis_instructions(field, intent):
    """Short action prompt for the second request after intent has already been selected."""
    specific = {
        "body_action": "Answer ONLY body_action: crouch/stand/base/hold.",
        "right_x": "Answer ONLY right_x: minus/backward, plus/forward, hold/unchanged.",
        "right_y": "Answer ONLY right_y: minus/right, plus/left, hold/unchanged.",
        "right_z": "Answer ONLY right_z: minus/down, plus/up, hold/unchanged.",
        "gripper": "Answer ONLY gripper: open, hold, or close.",
    }
    return (f"Choose {field} for this simultaneous 0.2-second tick; the selected intent is {intent!r}. "
            + AXIS_DIRECTION_CONTEXT + " " + specific[field])


def single_request_questions():
    """All semantic and motor choices to be answered from one image/state snapshot."""
    return {
        "intent": {"type": "choice", "instructions": SINGLE_REQUEST_INTENT_INSTRUCTIONS,
                   "criteria": INTENT_CRITERIA},
        "body_action": {"type": "choice", "instructions": single_request_axis_instructions("body_action"),
                        "criteria": BODY_CRITERIA},
        "right_x": {"type": "choice", "instructions": single_request_axis_instructions("right_x"),
                    "criteria": AXIS_CRITERIA},
        "right_y": {"type": "choice", "instructions": single_request_axis_instructions("right_y"),
                    "criteria": AXIS_CRITERIA},
        "right_z": {"type": "choice", "instructions": single_request_axis_instructions("right_z"),
                    "criteria": AXIS_CRITERIA},
        "gripper": {"type": "choice", "instructions": single_request_axis_instructions("gripper"),
                    "criteria": GRIPPER_CRITERIA},
    }


def load_openai_key():
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            name, sep, value = line.strip().removeprefix("export ").partition("=")
            if sep and name.strip() == "OPENAI_API_KEY":
                key = value.strip().strip("\"'")
                if key:
                    return key
    raise RuntimeError("Set OPENAI_API_KEY in the environment or the ignored root .env file")


def build_payload(state, questions, model=MODEL, image_detail="auto"):
    """Translate the controller's named criteria into the Decisions wire schema."""
    content = [{"type": "input_text", "text": VISUAL_CONTEXT + "\n" +
                json.dumps(model_input_state(state), separators=(",", ":"))}]
    for name, url in state.get("_image_data_urls", {}).items():
        if not url.startswith("data:image/jpeg;base64,"):
            raise ValueError("Camera inputs must be inline JPEG data URLs")
        content.extend([{"type": "input_text", "text": f"Camera: {name}"},
                        {"type": "input_image", "image_url": url, "detail": image_detail}])
    return {"model": model, "input": [{"role": "user", "content": content}],
            "questions": [{"type": "choice", "name": name,
                           "instructions": question["instructions"].replace("Jev", "model"),
                           "choices": [{"value": value, "description": description}
                                       for value, description in question["criteria"].items()]}
                          for name, question in questions.items()]}


def normalize_answers(body, questions):
    """Validate the entire answer set before any robot action can be applied."""
    answers = {}
    raw = body.get("answers")
    if not isinstance(raw, list):
        raise RuntimeError("Decisions response is missing its answers array")
    for answer in raw:
        if not isinstance(answer, dict):
            raise RuntimeError("Decisions returned a malformed answer")
        name = answer.get("name")
        if answer.get("type") == "refusal":
            raise RuntimeError(f"Decisions refused question {name!r}")
        if not isinstance(name, str) or name not in questions or name in answers or answer.get("type") != "choice":
            raise RuntimeError("Decisions returned an unexpected or duplicate question")
        if not isinstance(answer.get("choice"), str) or answer["choice"] not in questions[name]["criteria"]:
            raise RuntimeError(f"Decisions returned an invalid choice for {name}")
        confidence = answer.get("confidence")
        if not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise RuntimeError(f"Decisions returned invalid confidence for {name}")
        answers[name] = answer
    if set(answers) != set(questions):
        raise RuntimeError("Decisions returned an incomplete answer set")
    return answers


def post_decision(payload, key):
    request = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # Do not copy response bodies, request images, or credentials into exception logs.
        raise RuntimeError(f"OpenAI Decisions returned HTTP {exc.code}") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise RuntimeError("OpenAI Decisions request failed or timed out") from None
    except (ValueError, UnicodeError):
        raise RuntimeError("OpenAI Decisions returned invalid JSON") from None
    if not isinstance(body, dict):
        raise RuntimeError("OpenAI Decisions returned a non-object response")
    return body, time.monotonic() - started


class DecisionsClient(JevClient):
    """Reuse intent/axis question construction; override transport and observation preparation."""
    provider = "openai_decisions"
    expected_delay = "Unknown for this provider; physics continues throughout both requests."

    def __init__(self, model=MODEL, *, transport=None, image_detail="auto", cameras="head,third_person",
                 single_request=False):
        self.model = model
        self.image_detail = image_detail
        self.cameras = tuple(cameras.split(","))
        if not self.cameras or any(cam not in {"head", "third_person"} for cam in self.cameras):
            raise ValueError("cameras must contain head and/or third_person")
        if image_detail not in {"auto", "low", "high", "original"}:
            raise ValueError("Invalid image detail")
        # A remote transport owns credentials in Modal. The simulator never needs a key.
        self.key = None if transport else load_openai_key()
        self.transport = transport
        self.single_request = single_request

    def request_questions(self, state, questions):
        payload = build_payload(state, questions, self.model, self.image_detail)
        body, latency = self.transport(payload) if self.transport else post_decision(payload, self.key)
        return normalize_answers(body, questions), body.get("usage", {}), latency

    def warmup(self):
        # Decisions choice questions require at least two distinct options.
        self.request_questions({"task": "Connection warm-up only."}, {
            "warmup": {"instructions": "Select ready to acknowledge this request.",
                       "criteria": {"ready": "Connection acknowledged.", "unavailable": "Cannot acknowledge."}}})

    def decide_intent_axis(self, state, refresh_action_state=None):
        """Select intent, optionally refresh, then select motor actions in two requests by default."""
        if not self.single_request:
            intent_answer, intent_usage, intent_latency = self.decide_choice(state, INTENT_CRITERIA,
                                                                              DECISIONS_INTENT_INSTRUCTIONS)
            intent = intent_answer["choice"]
            if intent not in INTENT_CRITERIA:
                raise RuntimeError(f"Decisions returned unknown intent {intent!r}")
            action_state = state
            action_state.update({"selected_intent": intent,
                                 "selected_intent_description": INTENT_CRITERIA[intent]})
            if refresh_action_state:
                action_state = refresh_action_state(action_state)
            answers, action_usage, action_latency = self.request_questions(action_state, {
                "body_action": {"type": "choice", "instructions": two_stage_axis_instructions("body_action", intent),
                                "criteria": BODY_CRITERIA},
                "right_x": {"type": "choice", "instructions": two_stage_axis_instructions("right_x", intent),
                            "criteria": AXIS_CRITERIA},
                "right_y": {"type": "choice", "instructions": two_stage_axis_instructions("right_y", intent),
                            "criteria": AXIS_CRITERIA},
                "right_z": {"type": "choice", "instructions": two_stage_axis_instructions("right_z", intent),
                            "criteria": AXIS_CRITERIA},
                "gripper": {"type": "choice", "instructions": two_stage_axis_instructions("gripper", intent),
                            "criteria": GRIPPER_CRITERIA},
            })
            return intent_answer, answers, {"intent": intent_usage, "action": action_usage}, intent_latency + action_latency
        questions = single_request_questions()
        answers, usage, latency = self.request_questions(state, questions)
        return answers["intent"], {name: answers[name] for name in questions if name != "intent"}, \
            {"combined": usage}, latency

    def prepare_state(self, sim, state, out_dir, seed, decision, sim_time):
        if hasattr(sim, "get_state_and_observation"):
            observation = sim.get_state_and_observation(
                include_third_person="third_person" in self.cameras, marks=False, body_map=False)["observation"]
        else:
            observation = sim.get_observation(include_third_person="third_person" in self.cameras,
                                              marks=False, body_map=False)
        if abs(observation["sim_time"] - sim_time) > 1e-6:
            raise RuntimeError("Camera and state timestamps differ")
        return self._attach_observation(state, observation, out_dir, seed, decision)

    def refresh_action_state(self, sim, state, out_dir, seed, decision):
        """Capture a new state and image pair between the intent and action requests."""
        if hasattr(sim, "get_state_and_observation"):
            snapshot = sim.get_state_and_observation(
                include_third_person="third_person" in self.cameras, marks=False, body_map=False)
            current, observation = snapshot["state"], snapshot["observation"]
        else:
            current = sim.get_state()
            observation = sim.get_observation(include_third_person="third_person" in self.cameras,
                                              marks=False, body_map=False)
        old_goal = state.get("task_goal", {})
        goal = dict(old_goal)
        if "box_lift_target_m" in goal:
            goal["box_lifted"] = bool(current["task_success"])
            old_base = state.get("base_odom", [0.0])[0]
            goal["backward_progress_m"] = round(max(0.0, float(goal.get("backward_progress_m", 0.0))
                                                  + old_base - current["pelvis"]["odom"][0]), 3)
        first_visual_observations = list(state.get("visual_observations", []))
        refreshed = compact_state(current, state.get("previous", {}), state.get("recent_history", []),
                                  task=state["task"], task_goal=goal)
        for key in ("control_timing", "selected_intent", "selected_intent_description"):
            if key in state:
                refreshed[key] = state[key]
        refreshed = self._attach_observation(refreshed, observation, out_dir, seed, decision)
        # Keep both manifests in the request-state object used by the transcript while sending only
        # the freshly captured JPEG bytes in the second request.
        refreshed["visual_observations"] = first_visual_observations + refreshed["visual_observations"]
        state.clear()
        state.update(refreshed)
        return state

    def _attach_observation(self, state, observation, out_dir, seed, decision):
        folder = Path(out_dir) / f"seed_{seed}_inputs" / f"decision_{decision:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        images, manifest = {}, []
        for camera in self.cameras:
            encoded = observation["images"].get(camera)
            if not encoded:
                raise RuntimeError(f"Missing requested camera {camera}; refusing a text-only fallback")
            data = base64.b64decode(encoded, validate=True)
            path = folder / f"{camera}.jpg"
            path.write_bytes(data)
            images[camera] = "data:image/jpeg;base64," + encoded
            manifest.append({"camera": camera, "path": str(path.relative_to(out_dir)),
                             "sha256": hashlib.sha256(data).hexdigest(), "sim_time": observation["sim_time"],
                             "image_size": observation["image_size"], "detail": self.image_detail})
        return {**state, "visual_observations": manifest, "_image_data_urls": images}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--max-actions", type=int, default=25)
    parser.add_argument("--duration-s", type=float, default=0.2)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--carry-back", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sim-url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", default="runs/decisions")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--image-detail", choices=["auto", "low", "high", "original"], default="auto")
    parser.add_argument("--cameras", choices=["head", "third_person", "head,third_person"], default="head,third_person")
    parser.add_argument("--single-request", action="store_true",
                        help="Use one request for intent and actions instead of the default two-request topology.")
    args = parser.parse_args()
    if args.max_actions < 1 or not 0.04 <= args.duration_s <= 0.5:
        parser.error("Use a positive action budget and duration between 0.04 and 0.50 seconds")
    if not 0 <= args.min_confidence <= 1:
        parser.error("min-confidence must be between zero and one")
    client = DecisionsClient(args.model, image_detail=args.image_detail, cameras=args.cameras,
                             single_request=args.single_request)
    client.warmup()
    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    sim = SimClient(args.sim_url, timeout=60)
    results = [run_episode(client, sim, seed, args.max_actions, args.duration_s, args.min_confidence, out,
                           carry_back=args.carry_back, streaming=args.streaming,
                           record_dir=(out / f"seed_{seed}_frames") if args.record else None) for seed in args.seed]
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
