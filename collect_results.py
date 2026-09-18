"""Collect every result.json under a directory into a markdown table (stage, calls, tokens, cost, end reason)."""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = []
for p in sorted(glob.glob(os.path.join(root, "**", "result.json"), recursive=True)):
    r = json.load(open(p))
    if "model" not in r:  # scripted-policy results have a different shape
        continue
    d = os.path.relpath(os.path.dirname(p), root)
    model = r["model"].replace("us.anthropic.claude-", "").replace("anthropic.claude-", "").replace("-20251001-v1:0", "")
    rows.append((d, model + (" (fake)" if r.get("dry_run") else ""), r["stage"], r["calls"], r["input_tokens"], r["output_tokens"],
                 r["est_cost_usd"], str(r.get("end_reason", ""))[:60].replace("\n", " ")))
print("| episode | model | stage | calls | tokens in | tokens out | cost $ | end |")
print("|---|---|---|---|---|---|---|---|")
for row in rows:
    print("| " + " | ".join(str(x) for x in row) + " |")
