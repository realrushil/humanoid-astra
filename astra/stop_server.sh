#!/bin/bash
cd "$(dirname "$0")"
if [ -f runs/sim_server.pid ]; then
  PID=$(cat runs/sim_server.pid)
  if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"; sleep 3; kill -9 "$PID" 2>/dev/null
    echo "stopped sim_server pid $PID"
  else
    echo "sim_server not running"
  fi
  rm -f runs/sim_server.pid
else
  pkill -f "sim_server.py" && echo "killed stray sim_server" || echo "no sim_server running"
fi
