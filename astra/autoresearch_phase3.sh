#!/bin/bash
# Phase 3: two more play sessions (coverage rule) continuing memory v2 -> v3; phase 4: memory-side A/B episodes with v3.
cd "$(dirname "$0")"
while ! grep -q PHASE2_DONE runs/autoresearch_phase2.log; do sleep 20; done
cp runs/lessons_v2_frozen.json lessons.json
./restart_server.sh playground || { echo "server restart failed"; exit 1; }
./playground_loop.sh lessons.json 13 14
cp lessons.json runs/lessons_v3_frozen.json
./restart_server.sh carrybox || { echo "server restart failed"; exit 1; }
./transfer_loop.sh carrybox3 runs/lessons_v3_frozen.json 0 runs/lessons_v3_frozen.json 1
echo PHASE3_DONE
