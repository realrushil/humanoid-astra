"""Offline contract and streaming integration checks; no API key, network, or MuJoCo needed."""
import base64
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from run_decisions import DecisionsClient, normalize_answers, post_decision
from run_jev import run_episode


def answer_payload(payload):
    answers = []
    for question in payload["questions"]:
        choices = [c["value"] for c in question["choices"]]
        choice = "hold" if "hold" in choices else choices[0]
        answers.append({"type": "choice", "name": question["name"], "choice": choice, "confidence": 0.9,
                        "probabilities": [{"value": c, "probability": 1.0 if c == choice else 0.0} for c in choices]})
    return {"answers": list(reversed(answers)), "usage": {"input_tokens": 100}}


class FakeSim:
    def __init__(self, unblock):
        self.unblock = unblock
        self.steps = 0
        self.sim_time = 1.0
        self.captures = []
        self.state = {"box": {"pelvis": [0.3, 0, -0.2], "lift": 0.0},
                      "hands": {"right": {"grasp_point_pelvis": [0.2, 0, 0],
                                           "wrist_pelvis_pos": [0.05, -0.08, 0], "wrist_pelvis_rpy": [0, 0, 0],
                                           "box_surface_dist": 0.15, "openness_measured": 1.0}},
                      "pelvis": {"odom": [0, 0, 0], "height": 0.0},
                      "right_target_contact": False, "right_contact_forces": {},
                      "right_pinch": {"thumb_force_n": 0.0, "finger_force_n": 0.0,
                                      "other_force_n": 0.0, "ready": False},
                      "task_success": False, "max_stage": 0, "fallen": False, "box_on_floor": False}

    def reset(self, seed, record_dir=None):
        self.steps = 0
        self.sim_time = 1.0

    def get_state(self):
        return {**copy.deepcopy(self.state), "sim_time": self.sim_time}

    def get_observation(self, **kwargs):
        self.captures.append((threading.get_ident(), kwargs))
        # These opaque fixture bytes test transport fidelity, not a remote image decoder.
        return {"images": {c: base64.b64encode(b"jpeg-fixture-" + c.encode()).decode()
                           for c in ("head", "third_person")},
                "image_size": [640, 480], "sim_time": self.sim_time}

    def step_axes(self, body, axes, gripper, duration_s):
        self.steps += 1
        self.sim_time += duration_s
        if self.steps >= 3:
            self.unblock.set()
        return {"stage": 0, "fallen": False, "box_on_floor": False, "sim_time": self.sim_time,
                "blocked": {"right": None}, "grasp_events": []}


class DecisionsTests(unittest.TestCase):
    def test_multimodal_two_request_contract_uses_fresh_capture(self):
        requests = []

        def transport(payload):
            requests.append(payload)
            return answer_payload(payload), 0.02

        client = DecisionsClient(transport=transport)
        sim = FakeSim(threading.Event())
        with tempfile.TemporaryDirectory() as folder:
            state = client.prepare_state(sim, {"task": "pickup"}, Path(folder), 0, 1, 1.0)
            intent, answers, usage, latency = client.decide_intent_axis(
                state, refresh_action_state=lambda action_state: client.refresh_action_state(
                    sim, action_state, Path(folder), 0, 2))
            self.assertEqual(len(requests), 2)
            self.assertEqual([q["name"] for q in requests[0]["questions"]], ["choice"])
            self.assertEqual([q["name"] for q in requests[1]["questions"]],
                             ["body_action", "right_x", "right_y", "right_z", "gripper"])
            self.assertEqual(set(answers), {"body_action", "right_x", "right_y", "right_z", "gripper"})
            for payload in requests:
                self.assertEqual(payload["model"], "gpt-6-luna")
                self.assertNotIn("state", payload)
                parts = payload["input"][0]["content"]
                self.assertEqual(sum(p["type"] == "input_image" for p in parts), 2)
                self.assertNotIn("base64", parts[0]["text"])
                self.assertNotIn("right_wrist_rpy_rad", parts[0]["text"])
                self.assertNotIn("visual_observations", parts[0]["text"])
                self.assertNotIn("recent_action_effects", parts[0]["text"])
                self.assertNotIn("error_change_since_previous_action_m", parts[0]["text"])
                self.assertNotIn("criteria", payload["questions"][0])
                self.assertTrue(all("phase" not in q["instructions"].lower() and
                                    "action mask" not in q["instructions"].lower()
                                    for q in payload["questions"]))
            self.assertAlmostEqual(latency, 0.04)
            self.assertEqual(len(sim.captures), 2)
            self.assertNotEqual(requests[0]["input"][0]["content"][0]["text"],
                                requests[1]["input"][0]["content"][0]["text"])
            for entry in state["visual_observations"]:
                self.assertEqual((Path(folder) / entry["path"]).read_bytes(),
                                 b"jpeg-fixture-" + entry["camera"].encode())

    def test_refusal_missing_duplicate_and_invalid_choices(self):
        questions = {"body": {"criteria": {"hold": "", "stand": ""}}}
        answer = {"type": "choice", "name": "body", "choice": "hold", "confidence": 0.9}
        bad_sets = [[{"type": "refusal", "name": "body"}], [], [answer, answer],
                    [{**answer, "choice": "move_to"}], [{**answer, "confidence": float("nan")}],
                    [None], [{**answer, "name": []}], [{**answer, "choice": {}}]]
        for answers in bad_sets:
            with self.subTest(answers=answers), self.assertRaises(RuntimeError):
                normalize_answers({"answers": answers}, questions)

    def test_warmup_schema(self):
        requests = []
        client = DecisionsClient(transport=lambda p: (requests.append(p) or answer_payload(p), 0.0))
        client.warmup()
        self.assertGreaterEqual(len(requests[0]["questions"][0]["choices"]), 2)

    def test_missing_or_stale_camera_fails(self):
        client = DecisionsClient(transport=lambda p: (answer_payload(p), 0.0))
        sim = FakeSim(threading.Event())
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError, "timestamps"):
                client.prepare_state(sim, {}, Path(folder), 0, 1, 0.0)
            with patch.object(sim, "get_observation", return_value={"images": {}, "sim_time": 1.0}):
                with self.assertRaisesRegex(RuntimeError, "Missing requested camera"):
                    client.prepare_state(sim, {}, Path(folder), 0, 1, 1.0)

    def test_physics_advances_during_inference_and_waits_do_not_consume_budget(self):
        event = threading.Event()
        sim = FakeSim(event)
        requests = []

        def transport(payload):
            requests.append(payload)
            if not event.wait(timeout=2):
                raise RuntimeError("Physics failed to advance while the API was pending")
            return answer_payload(payload), 0.1

        client = DecisionsClient(transport=transport)
        with tempfile.TemporaryDirectory() as folder:
            result = run_episode(client, sim, 0, 1, 0.04, 0.0, Path(folder), streaming=True)
            trace = json.loads((Path(folder) / "seed_0.json").read_text())
            self.assertEqual(result["actions"], 1)
            self.assertGreaterEqual(result["control_ticks"], 4)
            self.assertEqual(len(requests), 2)
            self.assertEqual(len(sim.captures), 2)
            self.assertEqual(len(sim.captures), 2)
            self.assertEqual(sim.captures[0][0], threading.get_ident())
            self.assertNotIn("_image_data_urls", trace["model_decisions"][0]["state"])
            self.assertEqual(len(trace["model_decisions"][0]["state"]["visual_observations"]), 4)
            self.assertGreater(trace["model_decisions"][0]["staleness_sim_s"], 0)
            self.assertEqual(result["provider"], "openai_decisions")

    def test_refusal_preserves_transcript(self):
        sim = FakeSim(threading.Event())
        client = DecisionsClient(transport=lambda p: ({"answers": [{"name": "choice", "type": "refusal"}]}, 0.0))
        with tempfile.TemporaryDirectory() as folder:
            result = run_episode(client, sim, 0, 1, 0.04, 0.0, Path(folder), streaming=True)
            self.assertEqual(result["actions"], 0)
            self.assertIn("refused", result["end_reason"])
            self.assertTrue((Path(folder) / "seed_0.json").exists())

    def test_transport_uses_decisions_endpoint(self):
        with patch("run_decisions.urllib.request.urlopen") as send:
            send.return_value.__enter__.return_value.read.return_value = b'{"answers":[]}'
            post_decision({"model": "gpt-6-luna"}, "test-only-not-a-secret")
            request = send.call_args.args[0]
            self.assertEqual(request.full_url, "https://api.openai.com/v1/decisions")
            self.assertEqual(request.get_header("Authorization"), "Bearer test-only-not-a-secret")


if __name__ == "__main__":
    unittest.main()
