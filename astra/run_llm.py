"""run_llm.py: the agent loop. A client of sim_server.py that lets a vision LLM (AWS Bedrock Converse API, any
provider) drive the humanoid through move_to / done / give_up tool calls.

  python run_llm.py --dry-run --seed 0        # fake LLM that replays scripted_policy.scripted_plan; no tokens spent
  python run_llm.py --seed 0 --model ...      # one real episode (milestone 6); gpt-* models go to the OpenAI API directly

Per episode, runs/<name>/ gets transcript.jsonl, frames/ (per-turn JPEGs), record/ (10 fps recording of every
camera), episode.mp4 (third-person camera), first_request.json, result.json and index.html."""

import argparse
import copy
import html
import json
import math
import os
import sys
import time

from sim_client import SimClient, make_episode_videos, make_video

DEFAULT_MODEL = "us.anthropic.claude-haiku-4-5-20251001-v1:0"  # cross-region inference profile; bare IDs are INFERENCE_PROFILE-only
# USD per 1M tokens (input, output), matched by substring of the model id. Edit freely.
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "gpt-6-astra": (10.0, 50.0),  # OpenAI API list price, Sept 2026
    "claude-opus-5-5": (4.0, 20.0),  # Bedrock offer rate card; claude-cli runs use the CLI's own cost estimate instead
}
DEFAULT_PRICE = (3.0, 15.0)  # used (with a warning) when the model id matches nothing above
IMAGE_TOKENS = 320 * 240 // 750  # rough vision cost of one frame, only used by the fake client
CAMERAS = [("head", "head camera, forward-looking, mounted on the torso"),
           ("left_wrist", "left wrist camera"), ("right_wrist", "right wrist camera"),
           ("body_map", "top-down body map (not a camera: head image re-projected onto the plane at hand height, pelvis frame)")]
KEEP_FIRST_TURNS, KEEP_LAST_TURNS, KEEP_IMAGE_OBS = 4, 10, 2
TASK = "Pick up the box from the table in front of you and place it on the other table."
PLAYGROUND_TASK = ("Practice session: there is nothing to complete. Work in short experiments: state a small goal you are unsure your body "
                   "can achieve (any movement or interaction your body allows), try it, then say what actually "
                   "happened and whether it matched your expectation. Prefer goals whose outcome you cannot predict; spend at most about 8 "
                   "calls on one experiment, then move on. Do not repeat an experiment whose result you have already seen once: after a "
                   "surprise, change something you have not varied yet (orientation, height, a different object) or move on. Include at "
                   "least one experiment that touches or moves a movable object. Coverage: your notes list every kind of action you have (each hand's "
                   "open/close, hand orientation, height, walking, turning, every tool); over the session exercise each kind at least once and "
                   "prefer kinds you have not tried yet. Call done(summary) when you have finished practising.")


def load_memory(path):
    """Lessons written by reflect.py: [{id, lesson, confidence, count, evidence}], injected verbatim as guidance."""
    if not path:
        return []
    if not os.path.exists(path):
        print(f"WARNING: --memory {path} does not exist; running without lessons", flush=True)
        return []
    with open(path) as f:
        return json.load(f)


def memory_text(lessons):
    if not lessons:
        return ""
    lines = [f"{i + 1}. [{m.get('confidence', 'may')}] {m['lesson']}" for i, m in enumerate(lessons)]
    return ("\n\nWhat I learned about my own body in earlier sessions. These are guidance, not rules, and some may be wrong or too broad: "
            "when a lesson would make you give up or skip an approach, a cheap probe beats trusting it.\n"
            + "\n".join(lines))
SYSTEM = """You control a simulated humanoid robot through tool calls. Each turn you get camera images, the current hand
poses and base odometry, and the result of your previous call. Make small, deliberate motions and re-check the
observation after each one. In a single move_to call either walk (base_*) or move the hands, never both. Call
exactly one tool per turn; always fill in `note` with what you see and why you chose the motion, in at most two
short sentences. You have a budget of {n} tool calls; call done(summary) when {done_when}, or give_up(reason) if stuck.
Name base_x/base_y/base_yaw only when you intend to walk or turn; otherwise leave them out so the base stands still.
measure(camera, x, y) tells you exactly where a pixel is in 3D and how far it is from each hand; it costs a call but no motion.

Embodiment notes:
{notes}"""


def price_for(model):
    for k, v in PRICES.items():
        if k in model:
            return v
    print(f"WARNING: no price entry for model {model}; using default {DEFAULT_PRICE}", flush=True)
    return DEFAULT_PRICE


def build_tools(bounds, cameras=("head", "left_wrist", "right_wrist"), image_size=(320, 240)):
    w, h = image_size
    props = {n: {"type": "number", "minimum": round(lo, 3), "maximum": round(hi, 3)} for n, (lo, hi) in bounds.items()}
    return [
        {"toolSpec": {"name": "move_to", "description": (
            "Move the robot. `targets` names one or more dimensions and their absolute target values; unnamed "
            "dimensions keep their current commanded value. base_x/base_y/base_yaw (odometry frame) walk the base; "
            "left_*/right_* set hand poses in the pelvis frame (metres, radians); left_hand/right_hand set the hand "
            "opening (0 closed, 1 open). The simulation advances until the motion completes or times out, then "
            "returns the final tracking error per named dimension."),
            "inputSchema": {"json": {"type": "object", "properties": {
                "targets": {"type": "object", "properties": props, "additionalProperties": False},
                "note": {"type": "string", "maxLength": 300,
                         "description": "At most two short sentences (under 40 words): what you see and why this motion."}},
                "required": ["targets", "note"]}}}},
        {"toolSpec": {"name": "measure", "description": (
            f"Measure where something is: give a pixel (x right, y down, origin top-left) in one of the {w}x{h} camera images "
            "and get back that point in the pelvis frame plus its offset from each hand's grasp point. The robot does not move. "
            "Use it before reaching or walking to know how far things are."),
            "inputSchema": {"json": {"type": "object", "properties": {
                "camera": {"type": "string", "enum": list(cameras)},
                "x": {"type": "integer", "minimum": 0, "maximum": w - 1}, "y": {"type": "integer", "minimum": 0, "maximum": h - 1},
                "label": {"type": "string", "description": "What you are pointing at, a few words."}},
                "required": ["camera", "x", "y", "label"]}}}},
        {"toolSpec": {"name": "done", "description": "Declare the task complete.",
                      "inputSchema": {"json": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}}},
        {"toolSpec": {"name": "give_up", "description": "Stop trying; explain why.",
                      "inputSchema": {"json": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}}}},
    ]


# ------------------------------------------------------------------------------------------ LLM clients
# Both return the Bedrock Converse response shape: {"output": {"message": ...}, "usage": {...}, "stopReason": ...}.
class BedrockClient:
    def __init__(self, model, region):
        import boto3
        self.model, self.rt = model, boto3.client("bedrock-runtime", region_name=region)

    def converse(self, request):
        from botocore.exceptions import ClientError
        for attempt in range(5):
            try:
                return self.rt.converse(**request)
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code not in ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException") or attempt == 4:
                    raise
                wait = 5 * 2 ** attempt
                print(f"  bedrock {code}; retrying in {wait}s", flush=True)
                time.sleep(wait)


class OpenAIClient:
    """OpenAI Responses API (direct, not Bedrock) behind the same Converse-shaped request/response as BedrockClient.
    Key from OPENAI_API_KEY or ~/.openai_key (never written by this code)."""

    URL = "https://api.openai.com/v1/responses"

    def __init__(self, model):
        self.model = model
        self.key = os.environ.get("OPENAI_API_KEY") or (open(os.path.expanduser("~/.openai_key")).read().strip()
                                                         if os.path.exists(os.path.expanduser("~/.openai_key")) else None)
        if not self.key:
            raise RuntimeError("no OpenAI key: set OPENAI_API_KEY or create ~/.openai_key (mode 600) on the machine running run_llm.py")

    @staticmethod
    def _to_openai(request):
        import base64
        items = []
        for m in request["messages"]:
            if m["role"] == "assistant":
                text = " ".join(b["text"] for b in m["content"] if "text" in b)
                if text:
                    items.append({"role": "assistant", "content": [{"type": "output_text", "text": text}]})
                for b in m["content"]:
                    if "toolUse" in b:
                        items.append({"type": "function_call", "call_id": b["toolUse"]["toolUseId"], "name": b["toolUse"]["name"],
                                      "arguments": json.dumps(b["toolUse"]["input"])})
                continue
            parts = []
            for b in m["content"]:
                if "toolResult" in b:
                    items.append({"type": "function_call_output", "call_id": b["toolResult"]["toolUseId"],
                                  "output": " ".join(c.get("text", "") for c in b["toolResult"]["content"])})
                elif "image" in b:
                    b64 = base64.b64encode(b["image"]["source"]["bytes"]).decode()
                    parts.append({"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64}", "detail": "low"})
                else:
                    parts.append({"type": "input_text", "text": b["text"]})
            if parts:
                items.append({"role": "user", "content": parts})
        tools = [{"type": "function", "name": t["toolSpec"]["name"], "description": t["toolSpec"]["description"],
                  "parameters": t["toolSpec"]["inputSchema"]["json"]} for t in request["toolConfig"]["tools"]]
        return {"model": request["modelId"], "instructions": request["system"][0]["text"], "input": items, "tools": tools,
                "max_output_tokens": max(4096, request["inferenceConfig"]["maxTokens"]),  # reasoning tokens count against this
                "store": False}

    @staticmethod
    def _from_openai(resp):
        content = []
        for item in resp.get("output", []):
            if item.get("type") == "message":
                text = " ".join(c.get("text", "") for c in item.get("content", []) if c.get("type") == "output_text")
                if text.strip():
                    content.append({"text": text})
            elif item.get("type") == "function_call":
                try:
                    inp = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:
                    content.append({"text": f"(unparseable tool arguments: {item.get('arguments', '')[:200]})"})
                    continue
                content.append({"toolUse": {"toolUseId": item["call_id"], "name": item["name"], "input": inp}})
        u = resp.get("usage", {})
        usage = {"inputTokens": u.get("input_tokens", 0), "outputTokens": u.get("output_tokens", 0),
                 "cachedInputTokens": (u.get("input_tokens_details") or {}).get("cached_tokens", 0),
                 "reasoningTokens": (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0)}
        if any("toolUse" in b for b in content):
            stop = "tool_use"
        elif resp.get("status") == "incomplete":
            stop = (resp.get("incomplete_details") or {}).get("reason", "incomplete")
        else:
            stop = "end_turn"
        return {"output": {"message": {"role": "assistant", "content": content}}, "usage": usage, "stopReason": stop, "id": resp.get("id")}

    def converse(self, request):
        import urllib.error
        import urllib.request
        body = json.dumps(self._to_openai(request)).encode()
        for attempt in range(5):
            req = urllib.request.Request(self.URL, data=body, headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
            try:
                with urllib.request.urlopen(req, timeout=600) as r:
                    return self._from_openai(json.loads(r.read()))
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:400]
                if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                    wait = 5 * 2 ** attempt
                    print(f"  openai HTTP {e.code}; retrying in {wait}s: {detail[:120]}", flush=True)
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"OpenAI HTTP {e.code}: {detail}") from None


class ClaudeCLIClient:
    """Claude via the local `claude -p` CLI (Claude Code subscription, no API key) behind the same Converse shape.
    Each call is stateless: the trimmed history is flattened into one user message (images inline) and the tool call
    comes back as --json-schema structured output {"call": {"name": ..., "input": ...}}. Cost is the CLI's API-price
    estimate; subscription usage is not billed per call."""

    def __init__(self, model, effort=None):
        import tempfile
        self.model, self.effort, self.n = model, effort, 0
        self.cwd = tempfile.mkdtemp(prefix="claude_cli_")  # empty cwd: no CLAUDE.md / project settings picked up
        self.env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK")}

    @staticmethod
    def _schema(request):
        calls = [{"type": "object", "properties": {"name": {"const": t["toolSpec"]["name"]}, "input": t["toolSpec"]["inputSchema"]["json"]},
                  "required": ["name", "input"], "description": t["toolSpec"]["description"]} for t in request["toolConfig"]["tools"]]
        return {"type": "object", "properties": {"call": {"anyOf": calls}}, "required": ["call"]}

    @staticmethod
    def _to_blocks(request):
        import base64
        blocks = [{"type": "text", "text": "Transcript of the episode so far. You are the assistant; your earlier tool calls are shown as "
                                           "'Tool call'. Give your next tool call as the structured output {call: {name, input}}."}]
        for m in request["messages"]:
            blocks.append({"type": "text", "text": "=== assistant (you) ===" if m["role"] == "assistant" else "=== user ==="})
            for b in m["content"]:
                if "image" in b:
                    blocks.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                               "data": base64.b64encode(b["image"]["source"]["bytes"]).decode()}})
                elif "toolUse" in b:
                    blocks.append({"type": "text", "text": f"Tool call: {b['toolUse']['name']}({json.dumps(b['toolUse']['input'])})"})
                elif "toolResult" in b:
                    blocks.append({"type": "text", "text": "Tool result: " + " ".join(c.get("text", "") for c in b["toolResult"]["content"])})
                elif b.get("text", "").strip():
                    blocks.append({"type": "text", "text": b["text"]})
        return blocks

    def converse(self, request):
        import subprocess
        cmd = ["claude", "-p", "--model", self.model, "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
               "--tools", "", "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence", "--disable-slash-commands",
               "--system-prompt", request["system"][0]["text"], "--json-schema", json.dumps(self._schema(request))]
        if self.effort:
            cmd += ["--effort", self.effort]
        stdin = json.dumps({"type": "user", "message": {"role": "user", "content": self._to_blocks(request)}}) + "\n"
        for attempt in range(3):
            p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, cwd=self.cwd, env=self.env, timeout=900)
            res = next((json.loads(l) for l in p.stdout.splitlines() if l.startswith("{") and json.loads(l).get("type") == "result"), None)
            if res and not res.get("is_error") and res.get("structured_output"):
                break
            detail = (res or {}).get("result") or p.stderr[-400:] or p.stdout[-400:]
            if attempt == 2:
                raise RuntimeError(f"claude -p failed (rc={p.returncode}): {detail}")
            print(f"  claude -p failed (rc={p.returncode}); retrying in {10 * 2 ** attempt}s: {str(detail)[:160]}", flush=True)
            time.sleep(10 * 2 ** attempt)
        self.n += 1
        call = res["structured_output"].get("call") or {}
        content = [{"toolUse": {"toolUseId": f"cli-{self.n}", "name": call.get("name", "?"), "input": call.get("input") or {}}}]
        u = res.get("usage", {})
        usage = {"inputTokens": u.get("input_tokens", 0) + u.get("cache_creation_input_tokens", 0) + u.get("cache_read_input_tokens", 0),
                 "outputTokens": u.get("output_tokens", 0), "cachedInputTokens": u.get("cache_read_input_tokens", 0)}
        return {"output": {"message": {"role": "assistant", "content": content}}, "usage": usage, "stopReason": "tool_use",
                "cost_usd": res.get("total_cost_usd")}


class FakeClient:
    """Replays scripted_policy.scripted_plan (closed-loop on get_state) as tool calls; fakes token counts."""

    def __init__(self, sim, model):
        from scripted_policy import scripted_plan
        self.model, self.plan, self.started, self.n = model, scripted_plan(sim), False, 0

    def converse(self, request):
        last = None
        for b in request["messages"][-1]["content"]:
            if "toolResult" in b:
                last = b["toolResult"]["content"][0]["text"]
        try:
            targets, note = next(self.plan) if not self.started else self.plan.send(last)
            self.started = True
            call = {"name": "move_to", "input": {"targets": targets, "note": note}}
        except StopIteration as e:
            call = {"name": "done", "input": {"summary": e.value or "done"}}
        except RuntimeError as e:
            call = {"name": "give_up", "input": {"reason": str(e)}}
        self.n += 1
        summary = summarize_request(request)
        in_tok = len(json.dumps(summary)) // 4 + summary["_images"] * IMAGE_TOKENS
        out_json = json.dumps(call)
        return {"output": {"message": {"role": "assistant", "content": [
                    {"text": f"[fake LLM] {call['input'].get('note', '')}".strip()},
                    {"toolUse": {"toolUseId": f"fake-{self.n}", **call}}]}},
                "usage": {"inputTokens": in_tok, "outputTokens": len(out_json) // 4}, "stopReason": "tool_use", "fake": True}


def summarize_request(request):
    """Copy of a Converse request with image bytes replaced by a short description (for printing/logging)."""
    n = 0
    out = {k: v for k, v in request.items() if k != "messages"}
    out["messages"] = []
    for m in request["messages"]:
        blocks = []
        for b in m["content"]:
            if "image" in b:
                n += 1
                blocks.append({"image": f"<jpeg {len(b['image']['source']['bytes'])} bytes, 320x240>"})
            else:
                blocks.append(b)
        out["messages"].append({"role": m["role"], "content": blocks})
    out["_images"] = n
    return out


# ------------------------------------------------------------------------------------------ episode
class Episode:
    def __init__(self, sim, llm, args, ep_dir):
        self.sim, self.llm, self.args, self.dir = sim, llm, args, ep_dir
        os.makedirs(os.path.join(ep_dir, "frames"), exist_ok=True)
        self.tf = open(os.path.join(ep_dir, "transcript.jsonl"), "w")
        self.messages = []  # full history; image blocks are {"image_file": path} and expanded when sending
        self.image_bytes = {}
        self.turns = []  # for index.html
        self.usage = {"input": 0, "output": 0, "calls": 0, "cost": 0.0}
        self.price = price_for(args.model)
        self.end_reason = None

    def log(self, rec):
        self.tf.write(json.dumps(rec) + "\n")
        self.tf.flush()

    # ---- observation -> content blocks
    def observe(self, turn):
        obs = self.sim.get_observation(include_third_person=True, marks=not self.args.no_marks, body_map=self.args.body_map)
        files = {}
        for cam, b64 in obs["images"].items():
            fn = f"frames/turn{turn:02d}_{cam}.jpg"
            self.image_bytes[fn] = SimClient.decode_image(b64)
            with open(os.path.join(self.dir, fn), "wb") as f:
                f.write(self.image_bytes[fn])
            files[cam] = fn
        blocks = []
        for cam, label in CAMERAS:
            if cam in files:
                blocks += [{"text": f"Camera: {label}", "_label": True}, {"image_file": files[cam]}]
        blocks.append({"text": f"Robot state (hands in pelvis frame, base in odometry frame): {obs['state_text']}\n"
                               f"Tool calls used: {self.usage['calls']}/{self.args.max_calls}."})
        return blocks, files, obs["state_text"]

    def add(self, role, content, turn):
        self.messages.append({"role": role, "content": content})
        self.log({"type": "message", "role": role, "turn": turn,
                  "content": [{k: v for k, v in b.items() if k != "_label"} for b in content]})

    # ---- history -> trimmed Converse request
    def build_request(self):
        msgs = copy.deepcopy(self.messages)
        first, turns = msgs[0], [msgs[i:i + 2] for i in range(1, len(msgs), 2)]  # (assistant, user) pairs
        if len(turns) > KEEP_FIRST_TURNS + KEEP_LAST_TURNS:
            dropped = len(turns) - KEEP_FIRST_TURNS - KEEP_LAST_TURNS
            turns = turns[:KEEP_FIRST_TURNS] + turns[-KEEP_LAST_TURNS:]
            turns[KEEP_FIRST_TURNS - 1][1]["content"].append({"text": f"[{dropped} earlier turns elided]"})
        msgs = [first] + [m for t in turns for m in t]
        user_idx = [i for i, m in enumerate(msgs) if m["role"] == "user"]
        keep_images = set(user_idx[-KEEP_IMAGE_OBS:])
        for i, m in enumerate(msgs):
            out = []
            n_img = sum("image_file" in b for b in m["content"])
            for b in m["content"]:
                if "image_file" in b:
                    if i in keep_images:
                        out.append({"image": {"format": "jpeg", "source": {"bytes": self.image_bytes[b["image_file"]]}}})
                    elif not (out and out[-1].get("_stub")):
                        out.append({"text": f"[{n_img} camera frames elided]", "_stub": True})
                elif b.get("_label") and i not in keep_images:
                    continue
                else:
                    out.append({k: v for k, v in b.items() if k != "_label"})
            m["content"] = [{k: v for k, v in b.items() if k != "_stub"} for b in out]
        return {"modelId": self.args.model, "system": [{"text": self.system}], "messages": msgs,
                "toolConfig": {"tools": self.tools}, "inferenceConfig": {"maxTokens": self.args.max_tokens}}

    def account(self, resp, latency, summary):
        u = resp.get("usage", {})
        i, o = int(u.get("inputTokens", 0)), int(u.get("outputTokens", 0))
        cost = resp["cost_usd"] if resp.get("cost_usd") is not None else (i * self.price[0] + o * self.price[1]) / 1e6
        self.usage["input"] += i
        self.usage["output"] += o
        self.usage["cost"] += cost
        self.usage["calls"] += 1
        fake = " (fake)" if resp.get("fake") else ""
        print(f"[call {self.usage['calls']}] in={i} out={o} ${cost:.4f}{fake} | total in={self.usage['input']} "
              f"out={self.usage['output']} est ${self.usage['cost']:.4f} | {latency:.1f}s", flush=True)
        self.log({"type": "llm_call", "call": self.usage["calls"], "usage": u, "cost": cost, "total_cost": self.usage["cost"],
                  "latency_s": latency, "stop_reason": resp.get("stopReason"),
                  "request": {"messages": len(summary["messages"]), "images": summary["_images"],
                              "chars": len(json.dumps(summary))}})

    # ---- main loop
    def run(self):
        a = self.args
        record = f"{a.box_root}/{self.dir}/record" if a.box else os.path.abspath(os.path.join(self.dir, "record"))
        self.sim.reset(a.seed, record_dir=record)  # before describe(): the embodiment numbers are computed from the reset pose
        desc = self.sim.describe(marks=not a.no_marks, body_map=a.body_map)
        # the Isaac Lab server has one fixed task; other backends (SIMPLE) name theirs in describe()
        self.task = desc.get("instruction") or TASK
        done_when = "the task is complete" if desc.get("instruction") else "the box rests on the other table"
        if a.playground:
            self.task, done_when = PLAYGROUND_TASK, "you have finished practising"
        self.memory = load_memory(a.memory)
        self.system = SYSTEM.format(n=a.max_calls, notes=desc["text"], done_when=done_when) + memory_text(self.memory)
        self.tools = build_tools(desc["bounds"], **{k: desc[k] for k in ("cameras", "image_size") if k in desc})
        self.extra_tools = {t["toolSpec"]["name"] for t in desc.get("extra_tools", [])}  # backend-specific, dispatched by name
        self.tools += desc.get("extra_tools", [])
        import hashlib
        code_hash = hashlib.sha1(b"".join(open(f, "rb").read() for f in ["run_llm.py", "sim_server_simple.py", "sim_server_wbc.py", "reflect.py"]
                                          if os.path.exists(f))).hexdigest()[:10]
        st0 = self.sim.get_state()
        self.box_start = list(st0["box"]["pos"])
        self.max_lift, self.carried = 0.0, 0.0
        self.log({"type": "meta", "seed": a.seed, "model": a.model, "dry_run": a.dry_run, "max_calls": a.max_calls, "code_hash": code_hash,
                  "playground": a.playground, "memory": a.memory, "memory_lessons": [m["lesson"] for m in self.memory],
                  "max_cost": a.max_cost, "sticky_grasp": desc.get("sticky_grasp"), "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.log({"type": "system", "text": self.system})
        self.log({"type": "tools", "tools": self.tools})
        blocks, files, state_text = self.observe(0)
        self.add("user", [{"text": f"Task: {self.task}"}] + blocks, 0)
        self.turns.append({"turn": 0, "images": files, "state": state_text})
        turn, no_tool_streak = 0, 0
        while self.usage["calls"] < a.max_calls:
            turn += 1
            req = self.build_request()
            summary = summarize_request(req)
            if self.usage["calls"] == 0:
                with open(os.path.join(self.dir, "first_request.json"), "w") as f:
                    json.dump(summary, f, indent=1)
                if a.dry_run or a.show_request:
                    print("===== first Converse request (images summarized) =====\n" + json.dumps(summary, indent=1) + "\n=====", flush=True)
            t0 = time.time()
            try:
                resp = self.llm.converse(req)
            except Exception as e:  # noqa: BLE001 (auth/model-access/network: end the episode, keep the logs)
                self.end_reason = f"LLM error: {type(e).__name__}: {e}"
                print(self.end_reason, flush=True)
                break
            self.account(resp, time.time() - t0, summary)
            content = [b for b in resp["output"]["message"]["content"] if ("toolUse" in b) or b.get("text", "").strip()]
            self.add("assistant", content or [{"text": "(empty response)"}], turn)
            if self.usage["cost"] > a.max_cost:
                self.end_reason = f"cost ${self.usage['cost']:.3f} exceeded --max-cost {a.max_cost}"
                break
            tool_uses = [b["toolUse"] for b in content if "toolUse" in b]
            text = " ".join(b["text"] for b in content if "text" in b)
            rec = {"turn": turn, "text": text, "usage": resp.get("usage", {})}
            self.turns.append(rec)
            if not tool_uses:
                no_tool_streak += 1
                if no_tool_streak >= 2:
                    self.end_reason = "no tool call after re-prompt"
                    break
                self.add("user", [{"text": "Respond with exactly one tool call (move_to, measure, done or give_up)."}], turn)
                rec["result"] = "(no tool call; re-prompted)"
                continue
            no_tool_streak = 0
            tu, extra = tool_uses[0], tool_uses[1:]
            try:
                self._execute(tu, extra, rec, turn)
            except Exception as e:  # noqa: BLE001 (sim RPC / tunnel failure: end the episode, keep the logs)
                self.end_reason = f"sim error: {type(e).__name__}: {e}"
                print(self.end_reason, flush=True)
                break
            if self.end_reason:
                break
        else:
            self.end_reason = f"call budget ({a.max_calls}) exhausted"
        return self.finish()

    def _execute(self, tu, extra, rec, turn):
        """Run one tool call against the sim and append the result + next observation; sets end_reason to stop."""
        a = self.args
        if True:
            rec["tool"] = {"name": tu["name"], "input": tu["input"]}
            rec["note"] = tu["input"].get("note") or tu["input"].get("summary") or tu["input"].get("reason")
            self.log({"type": "tool_call", "call": self.usage["calls"], "name": tu["name"], "input": tu["input"]})
            if tu["name"] in ("done", "give_up"):
                self.end_reason = f"{tu['name']}: {rec['note']}"
                return
            if tu["name"] == "measure":
                try:
                    r = {}
                    result = self.sim.measure(tu["input"]["camera"], tu["input"]["x"], tu["input"]["y"], tu["input"].get("label", ""))["text"]
                except Exception as e:  # noqa: BLE001
                    result, r = f"error: {e}", {}
            elif tu["name"] in self.extra_tools:
                try:
                    r = {}
                    result = self.sim.call(tu["name"], **tu["input"])["text"]
                except Exception as e:  # noqa: BLE001
                    result, r = f"error: {e}", {}
            elif tu["name"] != "move_to":
                result, r = f"error: unknown tool {tu['name']}", {}
            else:
                try:
                    targets = {k: float(v) for k, v in (tu["input"].get("targets") or {}).items()}
                    r = self.sim.move_to(targets)
                    result = r["text"]
                except Exception as e:  # noqa: BLE001 (bad dimension names/values: report, do not advance)
                    result, r = f"error: {e}", {}
            print(f"  {tu['name']}({json.dumps(tu['input'].get('targets', {}))})\n  note: {rec['note']}\n  -> {result}", flush=True)
            for g in r.get("grasp_events", []):
                print(f"  ** sticky grasp {g['event']} ({g['hand']} hand)", flush=True)
            self.log({"type": "tool_result", "call": self.usage["calls"], "text": result, "fallen": r.get("fallen"),
                      "box_on_floor": r.get("box_on_floor"), "grasp_events": r.get("grasp_events", []), "sim_time": r.get("sim_time")})
            blocks, files, state_text = self.observe(turn)
            st = self.sim.get_state()
            self.max_lift = max(self.max_lift, float(st["box"]["lift"]))
            self.carried = max(self.carried, math.dist(st["box"]["pos"][:2], self.box_start[:2]))
            rec.update({"result": result, "images": files, "state": state_text, "stage": st["max_stage"], "sim_time": st["sim_time"]})
            content = [{"toolResult": {"toolUseId": tu["toolUseId"], "content": [{"text": result}]}}]
            content += [{"toolResult": {"toolUseId": e["toolUseId"], "content": [{"text": "ignored: only one tool call per turn"}], "status": "error"}} for e in extra]
            self.add("user", content + blocks, turn)
            if st["fallen"] or r.get("fallen"):
                self.end_reason = "robot fell"
                return
            if (st["box_on_floor"] or r.get("box_on_floor")) and not a.playground:
                self.end_reason = "box hit the floor"
                return

    def finish(self):
        st = self.sim.get_state()
        res = {"seed": self.args.seed, "model": self.args.model, "dry_run": self.args.dry_run, "stage": st["max_stage"],
               "success": st["max_stage"] == 4, "calls": self.usage["calls"], "input_tokens": self.usage["input"],
               "output_tokens": self.usage["output"], "est_cost_usd": round(self.usage["cost"], 4),
               "sticky_grasp_fired": any(e["event"] == "attach" for e in st["grasp_events"]), "grasp_events": st["grasp_events"],
               "fallen": st["fallen"], "box_on_floor": st["box_on_floor"], "task_success": st.get("task_success"), "end_reason": self.end_reason,
               "max_lift_cm": round(100 * getattr(self, "max_lift", 0.0), 1), "carried_cm": round(100 * getattr(self, "carried", 0.0), 1),
               "final_lift_cm": round(100 * float(st["box"]["lift"]), 1),
               "sim_time": st["sim_time"], "wall_time": round(time.time() - self.t_start, 1), "dir": self.dir}
        self.log({"type": "end", **res})
        self.tf.close()
        if self.args.box:  # sim ran remotely: the recording is on the box, so encode there and pull the episode dir back
            import subprocess
            py = (f"from sim_client import make_video, make_episode_videos; d='{self.dir}'; "
                  f"make_video(d + '/record/third_person', d + '/episode.mp4')"
                  + ("; make_episode_videos(d + '/record', d + '/episode')" if self.args.mosaic else ""))
            try:
                subprocess.run(["ssh", self.args.box, f"cd {self.args.box_root} && python3 -c \"{py}\""], check=True)
                subprocess.run(["rsync", "-az", f"{self.args.box}:{self.args.box_root}/{self.dir}/", f"{self.dir}/"], check=True)
                res["video"] = "episode_mosaic.mp4" if self.args.mosaic and os.path.exists(os.path.join(self.dir, "episode_mosaic.mp4")) else "episode.mp4"
            except Exception as e:  # noqa: BLE001
                print(f"remote video encoding / pull failed: {e}", flush=True)
        else:
            try:
                res["video"] = os.path.basename(make_video(os.path.join(self.dir, "record", "third_person"), os.path.join(self.dir, "episode.mp4")))
                if self.args.mosaic:  # per-camera mp4s + 2x2 mosaic (head | third-person / left wrist | right wrist)
                    outs = make_episode_videos(os.path.join(self.dir, "record"), os.path.join(self.dir, "episode"))
                    if "mosaic" in outs:
                        res["video"] = os.path.basename(outs["mosaic"])
            except Exception as e:  # noqa: BLE001
                print(f"video encoding failed: {e}", flush=True)
        with open(os.path.join(self.dir, "result.json"), "w") as f:
            json.dump(res, f, indent=1)
        write_html(self.dir, res, self.turns, self.system)
        return res


def write_html(ep_dir, res, turns, system):
    e = html.escape
    parts = [f"""<!doctype html><html><head><meta charset="utf-8"><title>episode seed {res['seed']}</title><style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px;background:#fafafa;color:#222}}
.turn{{border:1px solid #ddd;border-radius:8px;padding:12px;margin:14px 0;background:#fff}}
.imgs img{{width:320px;height:240px;margin:2px;border:1px solid #ccc}} .imgs span{{display:inline-block;font-size:12px;color:#555;text-align:center}}
pre{{white-space:pre-wrap;background:#f3f3f3;padding:8px;border-radius:4px;font-size:12px}}
.note{{font-style:italic;color:#1a4}} .meta{{color:#777;font-size:12px}} summary{{cursor:pointer}}
</style></head><body><h1>Episode: seed {res['seed']} — stage {res['stage']}{' (success)' if res['success'] else ''}</h1>
<p><b>model</b> {e(res['model'])}{' (dry run, fake LLM)' if res['dry_run'] else ''} · <b>calls</b> {res['calls']} · <b>tokens</b> {res['input_tokens']} in / {res['output_tokens']} out
· <b>est. cost</b> ${res['est_cost_usd']:.4f} · <b>sticky grasp</b> {'fired' if res['sticky_grasp_fired'] else 'no'} · <b>end</b> {e(str(res['end_reason']))}
· <b>sim</b> {res['sim_time']:.1f}s · <b>wall</b> {res['wall_time']:.0f}s</p>"""]
    if res.get("video"):
        parts.append(f'<video controls width="640" src="{res["video"]}"></video>')
    parts.append(f"<details><summary>system prompt</summary><pre>{e(system)}</pre></details>")
    for t in turns:
        parts.append(f"<div class=turn><h3>Turn {t['turn']}</h3>")
        if t.get("text"):
            parts.append(f"<p class=meta>assistant text: {e(t['text'])}</p>")
        if t.get("note"):
            parts.append(f"<p class=note>note: {e(t['note'])}</p>")
        if t.get("tool"):
            parts.append(f"<pre>{e(t['tool']['name'])}({e(json.dumps({k: v for k, v in t['tool']['input'].items() if k != 'note'}))})</pre>")
        if t.get("result"):
            parts.append(f"<p><b>result:</b> {e(t['result'])}</p>")
        if t.get("images"):
            parts.append("<div class=imgs>" + "".join(
                f"<span><img src='{fn}'><br>{cam}{' (not shown to LLM)' if cam == 'third_person' else ''}</span>"
                for cam, fn in t["images"].items()) + "</div>")
        if t.get("state"):
            parts.append(f"<p class=meta>{e(t['state'])}</p>")
        meta = []
        if t.get("usage"):
            meta.append(f"tokens in {t['usage'].get('inputTokens')} / out {t['usage'].get('outputTokens')}")
        if "stage" in t:
            meta.append(f"stage {t['stage']} · sim {t['sim_time']:.1f}s")
        if meta:
            parts.append(f"<p class=meta>{' · '.join(meta)}</p>")
        parts.append("</div>")
    parts.append("</body></html>")
    with open(os.path.join(ep_dir, "index.html"), "w") as f:
        f.write("\n".join(parts))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default=os.environ.get("MODEL_ID", DEFAULT_MODEL))
    p.add_argument("--region", default=os.environ.get("AWS_REGION", "us-west-2"))
    p.add_argument("--provider", choices=["auto", "bedrock", "openai", "claude-cli"], default="auto",
                   help="auto: gpt-* -> OpenAI API, else Bedrock; claude-cli: local `claude -p` (Claude Code subscription)")
    p.add_argument("--effort", default=None, help="claude-cli only: passed to `claude -p --effort`")
    p.add_argument("--memory", default=None, help="lessons.json from reflect.py, injected into the system prompt")
    p.add_argument("--playground", action="store_true", help="free-play session: the practice prompt replaces the task, no task end conditions")
    p.add_argument("--box", default=None, help="ssh host running the sim (e.g. astra) when this client runs elsewhere: the "
                   "recording is written there, encoded there, and the episode dir is pulled back")
    p.add_argument("--box-root", default="/home/ubuntu/astra")
    p.add_argument("--max-calls", type=int, default=40)
    p.add_argument("--max-cost", type=float, default=2.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--dry-run", action="store_true", help="fake LLM replaying scripted_policy; no Bedrock calls")
    p.add_argument("--show-request", action="store_true", help="print the first request payload even when not dry-run")
    p.add_argument("--mosaic", action="store_true", help="also encode per-camera videos and a 4-view mosaic; index.html embeds the mosaic")
    p.add_argument("--no-marks", action="store_true", help="ablation: raw camera images without the kinematic grasp-point marks")
    p.add_argument("--body-map", action="store_true", help="also send the top-down body map image (off by default: found unreadable)")
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default=None, help="episode directory name (default: llm_<timestamp>_seed<seed>[_dry])")
    p.add_argument("--sim-url", default="http://127.0.0.1:8765")
    args = p.parse_args()

    sim = SimClient(args.sim_url)
    sim.health()
    if args.dry_run:
        llm = FakeClient(sim, args.model)
    elif args.provider == "claude-cli":
        llm = ClaudeCLIClient(args.model, args.effort)
    elif args.provider == "openai" or (args.provider == "auto" and args.model.startswith(("gpt-", "o1", "o3", "o4"))):
        llm = OpenAIClient(args.model)
    else:
        llm = BedrockClient(args.model, args.region)
    name = args.name or f"llm_{time.strftime('%Y%m%d_%H%M%S')}_seed{args.seed}{'_dry' if args.dry_run else ''}"
    ep_dir = os.path.join(args.out, name)
    ep = Episode(sim, llm, args, ep_dir)
    ep.t_start = time.time()
    res = ep.run()
    print(json.dumps(res, indent=1), flush=True)
    print(f"report: {os.path.join(ep_dir, 'index.html')}", flush=True)
    print("RUN_LLM_DONE", flush=True)
