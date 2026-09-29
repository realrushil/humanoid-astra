#!/bin/bash
# Runs the walk demo in several SIMPLE environments, one server at a time. Usage: ./simple_walk_batch.sh (on the box)
cd "$(dirname "$0")"
run_env() {  # name env_id task_uid
  echo "=== $1 ($2) $(date +%H:%M:%S)"
  for p in $(pgrep -f "^\.venv/bin/python /home/ubuntu/astra/sim_server_simple.py"); do kill -9 $p; done; sleep 2
  rm -f runs/sim_server.pid runs/sim_server.log runs/start_simple.log
  ./start_server_simple.sh --env-id "$2" --task "$3" --record-every 2 > runs/start_simple.log 2>&1
  if ! grep -q healthy runs/start_simple.log; then echo "server failed for $1"; grep -E "line [0-9]+, in|Error" runs/sim_server.log | tail -5; return; fi
  rm -rf "runs/simple_walk_$1"
  ~/IsaacLab/.venv/bin/python simple_walk_demo.py --out "runs/simple_walk_$1" 2>&1 | grep -vE "Warning" | tail -12
  for p in $(pgrep -f "^\.venv/bin/python /home/ubuntu/astra/sim_server_simple.py"); do kill $p; done; sleep 3
}
run_env livingroom simple/G1WholebodyBendPickMP-v0 g1_wholebody_bend_pick_mp
run_env sofa simple/G1WholebodyBendPickAndPlaceOnSofaMP-v0 g1_wholebody_bend_pick_and_place_on_sofa_mp
run_env scene16 simple/G1WholebodyPickAndBendPlaceMP-v0 g1_wholebody_pick_and_bend_place_mp
echo "BATCH_DONE $(date +%H:%M:%S)"
