#!/bin/bash
# Generalisation check of the adapter's pinch grasp over GraspNet objects (scripted, privileged). Run from the laptop with the
# 8765 tunnel up. Usage: ./simple_grasp_batch.sh "12:apple 13:lemon 71:rubiks_cube 2:soup_can 5:banana"
cd "$(dirname "$0")"
for pair in $1; do
  id=${pair%%:*}; name=${pair#*:}
  echo "=== $name (graspnet1b:$id) $(date +%H:%M:%S)"
  ssh -n astra 'for p in $(pgrep -f "^\.venv/bin/python /home/ubuntu/astra/sim_server_simple.py"); do kill -9 $p; done; sleep 2; rm -f ~/astra/runs/sim_server.pid; cd ~/astra && timeout 580 ./start_server_simple.sh --env-id simple/G1WholebodyTabletopGraspMP-v0 --task g1_wholebody_tabletop_grasp_mp --record-every 2 --target-object graspnet1b:'"$id"' > runs/start_simple.log 2>&1 < /dev/null; tail -1 runs/start_simple.log'
  python3 grasp_calib.py --seeds 0 1 2 --strategies upright --record --out "runs/grasp_calib/$name.jsonl" 2>&1 | grep -v "^ "
done
echo BATCH_DONE
