"""Intent-first Jev controller for bounded whole-body SIMPLE actions.

Each tick uses two Jev requests: select a semantic intent, then select concurrent body, x/y/z, and gripper actions.
The server owns all action magnitudes and workspace limits; Jev never emits an absolute pose or calls ``move_to``.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

from sim_client import SimClient, make_video


ROOT = Path(__file__).resolve().parent.parent
API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"
TASK = "Pick up the box with the right hand and lift it at least 12 cm without dropping it or falling."
CARRY_BACK_TASK = (
    "Pick up the box with the right hand, lift it at least 12 cm, then walk backward at least 15 cm while keeping "
    "the box held and without falling or dropping it.")

BODY_CRITERIA = {
    "crouch": "Lower the body to reach a low target.", "stand": "Return to standing height; this can raise a held object.",
    "base_forward": "Walk forward briefly.", "base_back": "Walk backward briefly.",
    "base_left": "Walk left briefly.", "base_right": "Walk right briefly.",
    "turn_left": "Turn left briefly.", "turn_right": "Turn right briefly.", "hold": "Do not move the body this tick.",
}
AXIS_CRITERIA = {
    "minus": "Move this hand axis one bounded 4 cm increment in its negative direction.",
    "hold": "Leave this hand axis unchanged for this tick.",
    "plus": "Move this hand axis one bounded 4 cm increment in its positive direction.",
}
GRIPPER_CRITERIA = {
    "open": "Open the right gripper for this tick.", "hold": "Keep the current right gripper command for this tick.",
    "close": "Close the right gripper to attempt or maintain a pinch grasp.",
}
INTENT_CRITERIA = {
    "approach": "Bring the OPEN right grasp point into a stable pre-grasp pose. Choose while grasp distance is above 6 cm, while the box is substantially below the grasp point, or before the body posture can reach the box.",
    "grasp": "Secure a pinch grasp. Choose when grasp distance is about 6 cm or less and the body/hand are in a reachable pre-grasp pose. Stay in grasp while fingers are still closing (openness above 0.1) so they can settle on the box.",
    "lift": "Raise the box while maintaining the grasp. Choose only after fingers are nearly closed (openness at most 0.1) and the grasp point/contact remains near the box; then begin the lift.",
    "carry_back": "Carry the already lifted box backward. Choose only after the box is lifted at least 12 cm; keep it held while backward progress is below the requested distance.",
    "recover": "Correct a clearly worsening or blocked situation while preserving a safe robot posture.",
}
INTENT_INSTRUCTIONS = (
    "Choose exactly one current task intent for the next whole-body control tick. This is a semantic purpose, not an "
    "action gate: a separate call will retain every body, hand-axis, and gripper action. Use measured geometry, "
    "contacts/stage, and recent action effects. Do not invent an intent.")
INTENT_ACTION_CONTEXT = {
    "approach": "The hand is not pinch-ready yet: keep the gripper open, reduce signed position error, and consider crouching when the box is below reach.",
    "grasp": "The hand should already be close and reachable: preserve position, keep commanding close until measured openness is at most 0.1, and avoid moving the box away before contact.",
    "lift": "A grasp should already be established: keep fingers closed. If crouched, standing is a valid way to raise the held box; otherwise move it upward without dropping it.",
    "carry_back": "The box is already lifted: keep fingers closed and the hand stable, stand if necessary, and use base_back to make measured backward progress without dropping it.",
    "recover": "Use measured action effects to undo worsening motion while keeping the hand and body safe.",
}
DIRECTION_CONTEXT = (
    "Signed grasp error is box position minus right grasp point, in the pelvis frame. Positive error needs plus motion "
    "on that axis; negative error needs minus motion. Negative z means the box is below the grasp point. "
    "error_change_since_previous_action_m is descriptive feedback, not a constraint.")


def axis_field_instructions(field, intent):
    specific = {
        "body_action": "You are answering ONLY body_action: crouch/stand/base/hold. It does not move the hand axes.",
        "right_x": "You are answering ONLY right_x: minus moves the hand backward, plus moves it forward, hold leaves x unchanged.",
        "right_y": "You are answering ONLY right_y: minus moves the hand right, plus moves it left, hold leaves y unchanged.",
        "right_z": "You are answering ONLY right_z: minus moves the hand down, plus moves it up, hold leaves z unchanged.",
        "gripper": "You are answering ONLY gripper: it opens, holds, or closes the fingers and does not move the hand.",
    }
    return ("Choose this field for one simultaneous 0.2-second whole-body tick. Every option remains available; there are "
            "no action masks or server-enforced phases. " + DIRECTION_CONTEXT +
            f" The current Jev-selected task intent is {intent!r}: {INTENT_ACTION_CONTEXT[intent]} " + specific[field])


def load_local_key():
    key = os.environ.get("TYPESAFE_API_KEY")
    if key:
        return key
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TYPESAFE_API_KEY="):
                key = line.partition("=")[2].strip().strip("\"'")
                if key:
                    return key
    raise RuntimeError("TYPESAFE_API_KEY is not set; export it or add it to the ignored .env file")


class JevClient:
    def __init__(self, model=MODEL):
        self.key, self.model = load_local_key(), model

    def request_questions(self, state, questions):
        payload = {"model": self.model, "state": json.dumps(state, separators=(",", ":")), "questions": questions}
        request = urllib.request.Request(API_URL, data=json.dumps(payload).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Jev API returned HTTP {exc.code}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Jev API request failed: {exc.reason}") from None
        return body["answers"], body.get("usage", {}), time.monotonic() - started

    def decide_choice(self, state, criteria, instructions):
        answers, usage, latency = self.request_questions(state, {
            "choice": {"type": "choice", "instructions": instructions, "criteria": criteria},
        })
        return answers["choice"], usage, latency

    def decide_intent_axis(self, state):
        intent_answer, intent_usage, intent_latency = self.decide_choice(state, INTENT_CRITERIA, INTENT_INSTRUCTIONS)
        intent = intent_answer["choice"]
        if intent not in INTENT_CRITERIA:
            raise RuntimeError(f"Jev returned unknown intent {intent!r}")
        action_state = {**state, "selected_intent": intent, "selected_intent_description": INTENT_CRITERIA[intent],
                        "selected_intent_action_context": INTENT_ACTION_CONTEXT[intent]}
        answers, action_usage, action_latency = self.request_questions(action_state, {
            "body_action": {"type": "choice", "instructions": axis_field_instructions("body_action", intent), "criteria": BODY_CRITERIA},
            "right_x": {"type": "choice", "instructions": axis_field_instructions("right_x", intent), "criteria": AXIS_CRITERIA},
            "right_y": {"type": "choice", "instructions": axis_field_instructions("right_y", intent), "criteria": AXIS_CRITERIA},
            "right_z": {"type": "choice", "instructions": axis_field_instructions("right_z", intent), "criteria": AXIS_CRITERIA},
            "gripper": {"type": "choice", "instructions": axis_field_instructions("gripper", intent), "criteria": GRIPPER_CRITERIA},
        })
        return intent_answer, answers, {"intent": intent_usage, "action": action_usage}, intent_latency + action_latency


def compact_state(st, previous, history, *, task, task_goal):
    right, box = st["hands"]["right"], st["box"]
    grasp, box_p = right["grasp_point_pelvis"], box["pelvis"]
    error = [round(box_p[i] - grasp[i], 3) for i in range(3)]
    old_error = previous.get("grasp_error_m")
    error_delta = [round(error[i] - old_error[i], 3) for i in range(3)] if isinstance(old_error, list) else None
    recent = history[-8:]
    effects = [{"command": item["action"], "distance_m": [item["distance_before_m"], item["distance_after_m"]],
                "grasp_error_m": [item["error_before_m"], item["error_after_m"]], "stage_after": item["stage_after"],
                "lift_after_m": item["lift_after_m"], "backward_progress_after_m": item["backward_progress_after_m"]}
               for item in recent]
    return {
        "task": task, "task_goal": task_goal, "box_pelvis_m": [round(v, 3) for v in box_p],
        "box_lift_m": round(box["lift"], 3), "right_wrist_pelvis_m": [round(v, 3) for v in right["wrist_pelvis_pos"]],
        "right_wrist_rpy_rad": [round(v, 3) for v in right["wrist_pelvis_rpy"]], "right_grasp_error_m": error,
        "error_change_since_previous_action_m": error_delta, "recent_history": recent, "recent_action_effects": effects,
        "right_grasp_distance_m": round(right["box_surface_dist"], 3), "right_openness": round(right["openness_measured"], 3),
        "right_target_contact": bool(st["right_target_contact"]),
        "right_contact_forces": {name: round(force, 2) for name, force in st["right_contact_forces"].items()},
        "body_height_m": round(st["pelvis"]["height"], 3), "base_odom": [round(v, 3) for v in st["pelvis"]["odom"]],
        "task_stage": st["max_stage"], "previous": previous,
    }


def run_episode(client, sim, seed, max_actions, duration_s, min_confidence, out_dir, *, carry_back=False, record_dir=None):
    sim.reset(seed, record_dir=str(record_dir) if record_dir else None)
    initial_base_x = float(sim.get_state()["pelvis"]["odom"][0])
    task = CARRY_BACK_TASK if carry_back else TASK

    def backward_progress(st):
        return max(0.0, initial_base_x - float(st["pelvis"]["odom"][0]))

    def success(st):
        picked = bool(st["task_success"])
        return picked and (not carry_back or (backward_progress(st) >= 0.15 and bool(st["right_target_contact"])))

    transcript, history = [], []
    previous, end_reason = {"action": "reset", "outcome": "fresh episode"}, "action budget exhausted"
    for turn in range(1, max_actions + 1):
        st = sim.get_state()
        if st["fallen"] or st["box_on_floor"]:
            end_reason = "robot fell" if st["fallen"] else "box reached floor"
            break
        if success(st):
            end_reason = "task success detected"
            break
        goal = ({"box_lift_target_m": 0.12, "backward_target_m": 0.15,
                 "backward_progress_m": round(backward_progress(st), 3), "box_lifted": bool(st["task_success"]),
                 "requires_right_target_contact": True} if carry_back else {})
        state = compact_state(st, previous, history, task=task, task_goal=goal)
        intent_answer, axes_answers, usage, latency = client.decide_intent_axis(state)
        intent = intent_answer["choice"]
        body, gripper = axes_answers["body_action"]["choice"], axes_answers["gripper"]["choice"]
        labels = [axes_answers[name]["choice"] for name in ("right_x", "right_y", "right_z")]
        axes = [{"minus": -1, "hold": 0, "plus": 1}[label] for label in labels]
        confidence = min(float(intent_answer.get("confidence", 0.0)),
                         *(float(answer.get("confidence", 0.0)) for answer in axes_answers.values()))
        choice = {"intent": intent, "body": body, "right_axes": labels, "gripper": gripper}
        record = {"turn": turn, "state": state, "choice": choice, "confidence": confidence, "usage": usage,
                  "latency_s": round(latency, 3), "intent_answer": intent_answer, "axis_answers": axes_answers}
        if confidence < min_confidence:
            record["outcome"] = "abstained: below confidence threshold"
            transcript.append(record)
            end_reason = f"low confidence {confidence:.3f} below {min_confidence:.3f}"
            break
        outcome = sim.step_axes(body, axes, gripper, duration_s=duration_s)
        post = sim.get_state()
        post_grasp, post_box = post["hands"]["right"]["grasp_point_pelvis"], post["box"]["pelvis"]
        history.append({"action": f"intent={intent}; body={body}; x={labels[0]}; y={labels[1]}; z={labels[2]}; gripper={gripper}",
                        "error_before_m": state["right_grasp_error_m"],
                        "error_after_m": [round(post_box[i] - post_grasp[i], 3) for i in range(3)],
                        "distance_before_m": state["right_grasp_distance_m"],
                        "distance_after_m": round(post["hands"]["right"]["box_surface_dist"], 3),
                        "lift_after_m": round(post["box"]["lift"], 3), "stage_after": post["max_stage"],
                        "backward_progress_after_m": round(backward_progress(post), 3)})
        record["outcome"] = outcome
        transcript.append(record)
        previous = {"action": choice, "grasp_error_m": state["right_grasp_error_m"],
                    "grasp_distance_m": state["right_grasp_distance_m"],
                    "outcome": {key: outcome.get(key) for key in ("blocked", "grasp_events", "stage", "fallen", "box_on_floor")}}
    final = sim.get_state()
    result = {"seed": seed, "model": client.model, "controller_mode": "intent_axis",
              "task_variant": "carry_back" if carry_back else "pickup", "success": success(final),
              "actions": len(transcript), "sim_time": final["sim_time"], "stage": final["max_stage"],
              "fallen": final["fallen"], "box_on_floor": final["box_on_floor"], "box_lift_m": final["box"]["lift"],
              "end_reason": end_reason}
    if carry_back:
        result["backward_progress_m"] = backward_progress(final)
    if record_dir:
        video = out_dir / f"seed_{seed}_third_person.mp4"
        try:
            make_video(Path(record_dir) / "third_person", video, fps=10)
            result["video"] = str(video)
        except Exception as exc:
            result["video_error"] = f"{type(exc).__name__}: {exc}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"seed_{seed}.json").write_text(json.dumps({"result": result, "transcript": transcript}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-smoke", action="store_true", help="Call the intent selector once without a simulator.")
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--duration-s", type=float, default=0.2)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--carry-back", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--sim-url", default="http://127.0.0.1:8765")
    parser.add_argument("--out", default="runs/jev")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()
    client = JevClient(args.model)
    if args.api_smoke:
        answer, _, latency = client.decide_choice({"task": TASK, "right_grasp_distance_m": 0.25,
                                                    "right_openness": 1.0, "task_stage": 0},
                                                   INTENT_CRITERIA, INTENT_INSTRUCTIONS)
        print(json.dumps({"model": client.model, "intent": answer["choice"], "confidence": answer.get("confidence"),
                          "latency_s": round(latency, 3)}))
        return
    sim, run_id = SimClient(args.sim_url, timeout=60), time.strftime("%Y%m%d_%H%M%S")
    print("sim:", json.dumps(sim.health()))
    out_dir = Path(args.out) / run_id
    results = [run_episode(client, sim, seed, args.max_actions, args.duration_s, args.min_confidence, out_dir,
                           carry_back=args.carry_back,
                           record_dir=(out_dir / f"seed_{seed}_frames") if args.record else None)
               for seed in args.seed]
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"report: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
