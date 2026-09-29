#!/bin/bash
# Re-opens the 8765 tunnel to the box if the local port does not accept connections (a TCP check: the sim server is
# single-threaded, so an HTTP probe during a long move_to would time out and look like a dead tunnel). Exit 1 if it cannot.
up() { python3 -c "import socket,sys; s=socket.socket(); s.settimeout(3); s.connect(('127.0.0.1',8765))" 2>/dev/null; }
if ! up; then
  pkill -f "ssh -f -N .*-L 8765:localhost:8765 astra" 2>/dev/null; sleep 1
  ssh -f -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=20 -o ServerAliveCountMax=6 -L 8765:localhost:8765 astra
  sleep 2; up && echo "tunnel re-opened" || { echo "TUNNEL FAILED"; exit 1; }
fi
