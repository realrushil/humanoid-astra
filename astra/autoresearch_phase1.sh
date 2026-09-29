#!/bin/bash
# Phase 1: rebuild the memory on the corrected sim. 2 kitchen play sessions + 1 living-room play session, reflect after each.
cd "$(dirname "$0")"
mv -f lessons.json runs/lessons_v1_bad.json 2>/dev/null
./restart_server.sh playground
./playground_loop.sh lessons.json 10 11
./restart_server.sh carrybox
./ensure_tunnel.sh
echo "=== play living room seed 12 $(date +%H:%M:%S)"
python3 -u run_llm.py --provider claude-cli --model claude-opus-5-5 --playground --memory lessons.json --seed 12 --no-marks \
    --max-calls 40 --max-cost 6 --mosaic --box astra --name play_lr_seed12 > runs/play_lr_seed12.log 2>&1
python3 reflect.py --episode runs/play_lr_seed12 --memory lessons.json
cp lessons.json runs/lessons_v2_after_play.json
echo PHASE1_DONE
