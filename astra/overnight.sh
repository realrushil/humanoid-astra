#!/bin/bash
# Overnight: scripted ceiling (5 seeds) -> fresh memory (3 play sessions) -> interleaved A/B (5 seeds x {none, memory}) -> stop the box.
cd "$(dirname "$0")"
: "${ASTRA_INSTANCE_ID:?set ASTRA_INSTANCE_ID (EC2 instance to stop at the end)}"
export MAX_CALLS=60
./restart_server.sh carrybox || exit 1
echo "=== scripted probe $(date +%H:%M:%S)"; python3 wbc_probe.py --seeds 0 1 2 3 4 2>&1 | tail -6
./restart_server.sh playground || exit 1
mv -f lessons.json runs/lessons_v3_stale.json 2>/dev/null
./playground_loop.sh lessons.json 20 21 22
cp lessons.json runs/lessons_v4_frozen.json
./restart_server.sh carrybox || exit 1
./transfer_loop.sh ov none 0 runs/lessons_v4_frozen.json 0 none 1 runs/lessons_v4_frozen.json 1 none 2 runs/lessons_v4_frozen.json 2 \
                      none 3 runs/lessons_v4_frozen.json 3 none 4 runs/lessons_v4_frozen.json 4
echo "=== summary $(date +%H:%M:%S)"
python3 - <<'PY'
import json, glob
for f in sorted(glob.glob("runs/ov_*/result.json")):
    r = json.load(open(f)); print(f.split("/")[1], {k: r.get(k) for k in ("stage", "task_success", "max_lift_cm", "carried_cm", "final_lift_cm", "calls")})
PY
ssh astra 'cd ~/astra && for p in $(pgrep -f "^\.venv/bin/python /home/ubuntu/astra/sim_server_"); do kill $p; done'
aws ec2 stop-instances --instance-ids "$ASTRA_INSTANCE_ID" --region us-west-2 --query 'StoppingInstances[0].CurrentState.Name' --output text
echo OVERNIGHT_DONE
