"""Outside-agent reflection: reads one episode's transcript (tool calls, results, the agent's notes, end reason), then a fresh
`claude -p` process with no other context edits lessons.json under ExpeL-style rules: lessons only about the robot's own
body/controller/sensors/tools, each ADD must quote its evidence step, new lessons start at count 2, UPVOTE/DOWNVOTE +-1, deleted
at 0, at most MAX_LESSONS (lowest count, then oldest, dropped first).
  python reflect.py --episode runs/play_seed0 --memory lessons.json"""

import argparse
import json
import os
import subprocess
import tempfile
import time

MAX_LESSONS = 10
SYSTEM = """You review one session of a humanoid robot that was controlled by a language model through tool calls. You were not
part of the session. You maintain a SHORT memory of lessons about the ROBOT'S OWN BODY, CONTROLLER, SENSORS AND TOOLS: properties
that would hold in any room with any object. Not task strategies, not facts about particular objects or this scene.
Rules:
1. Evidence first. Every ADD must quote the exact step it comes from (the call number and the relevant part of the tool result).
   A lesson that the tool results do not support is not allowed. (You see the calls and their results only, not the agent's reasoning.)
2. Generality is bounded by the evidence. The condition of a lesson may only name the categories of things actually seen in the
   evidence; if you want to state it more broadly, add
   "(untested beyond <what was seen>)" and keep confidence "may".
3. Phrase each lesson as "When <condition>, <what happens / what to do> because <mechanism>". confidence "may" = seen once,
   "should" = seen repeatedly or mechanistically certain. At most one approximate number per lesson; no ranges, no force values;
   measurements belong in the evidence field.
4. Counter-evidence pass, every time: for each EXISTING lesson, state in observations whether its condition occurred this session
   ("lesson N: tested at call K, held / did not hold" or "lesson N: not tested"), and look for calls where the condition held but the outcome differed.
   If found, DOWNVOTE or EDIT it (quote the call). A lesson the agent believed and acted on without any tool result testing it is
   "untested", not confirmed: do not UPVOTE it.
5. Prefer EDIT or UPVOTE of an existing lesson over adding a similar one. Would the new lesson hold in any room with any object?
   If not, do not add it.
6. Keep the memory small (at most %d lessons) and terse. No advice the transcript does not support.""" % MAX_LESSONS
SCHEMA = {"type": "object", "properties": {
    "observations": {"type": "array", "items": {"type": "object", "properties": {
        "call": {"type": "integer"}, "what_happened": {"type": "string"}}, "required": ["call", "what_happened"]}},
    "ops": {"type": "array", "items": {"type": "object", "properties": {
        "op": {"type": "string", "enum": ["ADD", "EDIT", "UPVOTE", "DOWNVOTE"]},
        "id": {"type": ["integer", "null"]}, "lesson": {"type": ["string", "null"]},
        "confidence": {"type": ["string", "null"], "enum": ["may", "should", None]},
        "evidence": {"type": ["string", "null"]}}, "required": ["op"]}}},
    "required": ["observations", "ops"]}


def digest(ep_dir, max_result=420, notes=False):
    """Compressed transcript: each tool call and its result and how the episode ended. The agent's notes (its own claims) are
    left out by default so the reflector judges from tool results only; notes=True includes them."""
    lines, end = [], None
    for l in open(os.path.join(ep_dir, "transcript.jsonl")):
        d = json.loads(l)
        if d["type"] == "tool_call":
            inp = dict(d["input"])
            note = inp.pop("note", None) or inp.pop("summary", None) or inp.pop("reason", None)
            lines.append(f"call {d['call']}: {d['name']}({json.dumps(inp)})" + (f"  note: {note}" if (note and notes) else ""))
        elif d["type"] == "tool_result":
            lines.append(f"   result: {d['text'][:max_result]}")
        elif d["type"] == "end":
            end = d
    if end:
        reason = str(end.get("end_reason") or "")
        if not notes and ":" in reason:  # keep only the kind of ending (done/give_up/budget), not the agent's summary paragraph
            reason = reason.split(":", 1)[0]
        lines.append(f"END: {reason} (robot fell: {end.get('fallen')})")
    return "\n".join(lines)


def apply_ops(mem, ops):
    next_id = max([m["id"] for m in mem], default=0) + 1
    by_id = {m["id"]: m for m in mem}
    for o in ops:
        if o["op"] == "ADD" and o.get("lesson") and o.get("evidence"):
            mem.append({"id": next_id, "lesson": o["lesson"].strip(), "confidence": o.get("confidence") or "may",
                        "evidence": [o["evidence"]], "count": 2, "added": time.strftime("%Y-%m-%d %H:%M")})
            next_id += 1
        elif o["op"] == "EDIT" and o.get("id") in by_id and o.get("lesson"):
            m = by_id[o["id"]]
            m["lesson"] = o["lesson"].strip()
            if o.get("confidence"):
                m["confidence"] = o["confidence"]
            if o.get("evidence"):
                m["evidence"].append(o["evidence"])
        elif o["op"] in ("UPVOTE", "DOWNVOTE") and o.get("id") in by_id:
            m = by_id[o["id"]]
            m["count"] += 1 if o["op"] == "UPVOTE" else -1
            if o.get("evidence"):
                m["evidence"].append(o["evidence"])
    mem = [m for m in mem if m["count"] > 0]
    mem.sort(key=lambda m: (-m["count"], -m["id"]))  # ties: keep the newest (this session's ADDs), drop the oldest
    mem = mem[:MAX_LESSONS]
    mem.sort(key=lambda m: m["id"])
    return mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--memory", required=True)
    ap.add_argument("--model", default="claude-opus-5-5")
    a = ap.parse_args()
    mem = json.load(open(a.memory)) if os.path.exists(a.memory) else []
    user = ("CURRENT MEMORY (id, count, confidence, lesson):\n" + ("\n".join(f"{m['id']} (count {m['count']}, {m['confidence']}): {m['lesson']}" for m in mem) or "(empty)")
            + "\n\nSESSION TRANSCRIPT:\n" + digest(a.episode)
            + "\n\nList the observations that carry evidence (call number + what the result showed), then the memory operations.")
    cwd = tempfile.mkdtemp(prefix="reflect_")
    env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "CLAUDE_CODE_USE_BEDROCK")}
    cmd = ["claude", "-p", "--model", a.model, "--output-format", "json", "--tools", "", "--setting-sources", "", "--strict-mcp-config",
           "--no-session-persistence", "--disable-slash-commands", "--system-prompt", SYSTEM, "--json-schema", json.dumps(SCHEMA)]
    out = None
    for attempt in range(3):
        try:
            p = subprocess.run(cmd, input=user, capture_output=True, text=True, cwd=cwd, env=env, timeout=900)
        except subprocess.TimeoutExpired:
            print("reflection: claude -p timed out; retrying", flush=True)
            continue
        res = json.loads(p.stdout) if p.returncode == 0 and p.stdout.strip() else {}
        out = res.get("structured_output")
        if out:
            break
        print(f"reflection attempt {attempt + 1} failed (rc={p.returncode}): {(res.get('result') or p.stderr[-300:] or '')[:200]}", flush=True)
        time.sleep(10)
    if not out:
        raise SystemExit("reflection failed after 3 attempts")
    before = [m["lesson"] for m in mem]
    mem = apply_ops(mem, out["ops"])
    with open(os.path.join(a.episode, "reflection.json"), "w") as f:
        json.dump({"output": out, "memory_before": before, "memory_after": mem, "cost_usd": res.get("total_cost_usd")}, f, indent=1)
    with open(a.memory, "w") as f:
        json.dump(mem, f, indent=1)
    print(f"reflection: {len(out['observations'])} observations, {len(out['ops'])} ops -> {len(mem)} lessons (est ${res.get('total_cost_usd', 0):.2f})")
    for m in mem:
        print(f"  [{m['count']}|{m['confidence']}] {m['lesson']}")


if __name__ == "__main__":
    main()
