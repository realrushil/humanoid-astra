"""Dump an episode's full conversation (every message, in order, exactly as recorded) to conversation.md.
Usage: python dump_conversation.py runs/<episode> [...]. Images appear as [image: frames/...] placeholders."""
import json
import os
import sys

for ep in sys.argv[1:]:
    recs = [json.loads(l) for l in open(os.path.join(ep, "transcript.jsonl"))]
    out = [f"# Conversation: {ep}\n"]
    usage = {r["call"]: r for r in recs if r["type"] == "llm_call"}
    for r in recs:
        if r["type"] == "meta":
            out.append(f"model: {r['model']}  seed: {r['seed']}  started: {r['started']}\n")
        elif r["type"] == "system":
            out.append("## SYSTEM PROMPT\n```\n" + r["text"] + "\n```\n")
        elif r["type"] == "tools":
            out.append("## TOOLS\n```json\n" + json.dumps([t["toolSpec"]["name"] + ": " + t["toolSpec"]["description"] for t in r["tools"]], indent=1) + "\n```\n")
        elif r["type"] == "message" and r["role"] == "user":
            lines = []
            for b in r["content"]:
                if "toolResult" in b:
                    lines.append("[tool result] " + " ".join(c.get("text", "") for c in b["toolResult"]["content"]))
                elif "image_file" in b:
                    lines.append(f"[image: {b['image_file']}]")
                else:
                    lines.append(b["text"])
            out.append(f"## USER (turn {r['turn']})\n```\n" + "\n".join(lines) + "\n```\n")
        elif r["type"] == "message" and r["role"] == "assistant":
            u = usage.get(r["turn"], {}).get("usage", {})
            lines = []
            for b in r["content"]:
                if "toolUse" in b:
                    lines.append(f"{b['toolUse']['name']}({json.dumps(b['toolUse']['input'])})")
                elif "text" in b:
                    lines.append(f"(text) {b['text']}")
            out.append(f"## ASSISTANT (turn {r['turn']}, in={u.get('inputTokens')} out={u.get('outputTokens')} tokens)\n```\n" + "\n".join(lines) + "\n```\n")
        elif r["type"] == "end":
            out.append(f"## END\nstage={r['stage']} calls={r['calls']} cost=${r['est_cost_usd']} reason: {r['end_reason']}\n")
    path = os.path.join(ep, "conversation.md")
    open(path, "w").write("\n".join(out))
    print(f"wrote {path} ({len(out)} blocks)")
