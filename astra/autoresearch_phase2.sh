#!/bin/bash
# Phase 2: frozen-memory A/B on the floor-box -> table task (server already in carrybox mode after phase 1).
cd "$(dirname "$0")"
while ! grep -q PHASE1_DONE runs/autoresearch_phase1.log; do sleep 20; done
cp lessons.json runs/lessons_v2_frozen.json
./transfer_loop.sh carrybox2 none 0 runs/lessons_v2_frozen.json 0 none 1 runs/lessons_v2_frozen.json 1
echo PHASE2_DONE
