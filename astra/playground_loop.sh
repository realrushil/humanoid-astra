#!/bin/bash
# Free-play sessions with outside-agent reflection between them. Usage: ./playground_loop.sh lessons.json 0 1 2   (memory file, seeds)
# Needs the playground sim server (start_server_wbc.sh --playground) and the 8765 tunnel.
cd "$(dirname "$0")"
MEM=$1; shift
for seed in "$@"; do
  ./ensure_tunnel.sh || { echo "skipping seed $seed: no tunnel"; continue; }
  echo "=== play seed $seed $(date +%H:%M:%S)"
  python3 -u run_llm.py --provider claude-cli --model claude-opus-5-5 --playground --memory "$MEM" --seed "$seed" --no-marks \
      --max-calls 40 --max-cost 6 --mosaic --box astra --name "play_seed$seed" > "runs/play_seed$seed.log" 2>&1
  grep -E '"stage"|end_reason' "runs/play_seed$seed/result.json" | head -2
  [ -f "runs/play_seed$seed/result.json" ] && python3 reflect.py --episode "runs/play_seed$seed" --memory "$MEM" || echo "seed $seed: no result.json, skipping reflection"
done
echo LOOP_DONE
