#!/bin/bash
# Starts sim_server_wbc.py (SIMPLE backend) in the background on the GPU box and waits for the health check.
cd "$(dirname "$0")"
mkdir -p runs
PORT=${PORT:-8765}
if [ -f runs/sim_server.pid ] && kill -0 "$(cat runs/sim_server.pid)" 2>/dev/null; then
  echo "a sim server is already running (pid $(cat runs/sim_server.pid)); ./stop_server.sh first"; exit 0
fi
export MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES
( cd ~/SIMPLE && nohup .venv/bin/python ~/astra/sim_server_wbc.py --port "$PORT" "$@" > ~/astra/runs/sim_server.log 2>&1 & echo $! > ~/astra/runs/sim_server.pid )
echo "started sim_server_wbc pid $(cat runs/sim_server.pid) (log: runs/sim_server.log); waiting for health ..."
for i in $(seq 1 240); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "healthy after ~$((i*5))s: $(curl -s http://127.0.0.1:$PORT/health)"; exit 0
  fi
  if ! kill -0 "$(cat runs/sim_server.pid)" 2>/dev/null; then
    echo "sim server died; tail of log:"; grep -vE "Warning\]|\[Warning\]" runs/sim_server.log | tail -40; exit 1
  fi
  sleep 5
done
echo "timed out waiting for health"; exit 1
