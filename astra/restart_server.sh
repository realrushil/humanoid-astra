#!/bin/bash
# restart_server.sh MODE  (MODE = playground | carrybox); syncs code first, waits for health, re-opens the tunnel.
cd "$(dirname "$0")"
rsync -az --exclude runs ./ astra:~/astra/
case "$1" in
  playground) ARGS='--record-every 2 --playground --instruction placeholder' ;;
  carrybox)   ARGS='--record-every 2 --instruction "Pick up the cardboard box from the floor and place it on the table."' ;;
  *) echo "unknown mode $1"; exit 1 ;;
esac
ssh -n astra "for p in \$(pgrep -f '^\.venv/bin/python /home/ubuntu/astra/sim_server_'); do kill -9 \$p; done; sleep 1; rm -f ~/astra/runs/sim_server.pid; cd ~/astra && timeout 900 ./start_server_wbc.sh $ARGS > runs/start_wbc.log 2>&1 < /dev/null; tail -1 runs/start_wbc.log"
pkill -f "ssh -f -N .*-L 8765:localhost:8765 astra" 2>/dev/null; sleep 1
./ensure_tunnel.sh || exit 1; curl -s -m 5 localhost:8765/health || exit 1; echo
