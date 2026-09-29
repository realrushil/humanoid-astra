#!/bin/bash
# A/B deployment runs on the current sim server: ./transfer_loop.sh PREFIX MEMORY_OR_none SEED [MEMORY_OR_none SEED ...]
cd "$(dirname "$0")"
PREFIX=$1; shift
while [ $# -ge 2 ]; do
  MEM=$1; SEED=$2; shift 2
  if [ "$MEM" = "none" ]; then NAME="${PREFIX}_nomem_seed$SEED"; MARGS=""; else NAME="${PREFIX}_mem_seed$SEED"; MARGS="--memory $MEM"; fi
  ./ensure_tunnel.sh || { echo "skipping $NAME: no tunnel"; continue; }
  echo "=== $NAME $(date +%H:%M:%S)"
  python3 -u run_llm.py --provider claude-cli --model claude-opus-5-5 --seed "$SEED" --no-marks --max-calls ${MAX_CALLS:-50} --max-cost 8 \
      --mosaic --box astra --name "$NAME" $MARGS > "runs/$NAME.log" 2>&1
  [ -f "runs/$NAME/result.json" ] || { echo "$NAME produced no result.json"; continue; }
  python3 -c "import json; r=json.load(open('runs/$NAME/result.json')); print({k: r.get(k) for k in ('stage','task_success','calls','end_reason')})" | cut -c1-300
  if [ "$MEM" != "none" ] && [ -n "$REFLECT_TASKS" ]; then python3 reflect.py --episode "runs/$NAME" --memory "$MEM"; fi
done
echo TRANSFER_DONE
