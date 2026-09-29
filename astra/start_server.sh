#!/bin/bash
# Starts sim_server.py in the background (on the GPU box) and waits for the health check.
cd "$(dirname "$0")"
mkdir -p runs
if [ -f runs/sim_server.pid ] && kill -0 "$(cat runs/sim_server.pid)" 2>/dev/null; then
  echo "sim_server already running (pid $(cat runs/sim_server.pid))"; exit 0
fi
PORT=${PORT:-8765}
nohup ./run_on_box.sh sim_server.py --port "$PORT" > runs/sim_server.log 2>&1 &
echo $! > runs/sim_server.pid
echo "started sim_server pid $! (log: runs/sim_server.log); waiting for health ..."
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "healthy after ~$((i*5))s: $(curl -s http://127.0.0.1:$PORT/health)"; exit 0
  fi
  if ! kill -0 "$(cat runs/sim_server.pid)" 2>/dev/null; then
    echo "sim_server died; tail of log:"; grep -vE "Warning\]" runs/sim_server.log | tail -30; exit 1
  fi
  sleep 5
done
echo "timed out waiting for health"; exit 1
