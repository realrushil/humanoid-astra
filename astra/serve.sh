#!/bin/bash
# Serve ./runs on localhost for viewing over an SSH tunnel:  ssh -L 8000:localhost:8000 astra  then open http://localhost:8000/
cd "$(dirname "$0")/runs" && python3 -m http.server "${PORT:-8000}" --bind 127.0.0.1
