#!/usr/bin/env bash
set -euo pipefail
CAP_ROOT="${1:?Pass project root}"
DEP_ROOT="$CAP_ROOT/deps"
REPO="$DEP_ROOT/GR00T-WholeBodyControl"
mkdir -p "$DEP_ROOT/sysroot" "$DEP_ROOT/downloads"
cd "$DEP_ROOT/downloads"
apt-get download libzmq3-dev libmsgpack-dev libgtest-dev nlohmann-json3-dev
for package in ./*.deb; do dpkg-deb -x "$package" "$DEP_ROOT/sysroot"; done
if [ ! -d "$DEP_ROOT/onnxruntime-linux-x64-1.16.3" ]; then
 curl --fail --location --retry 3 --connect-timeout 20 --max-time 120 -o onnxruntime-linux-x64-1.16.3.tgz https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
 tar -xzf onnxruntime-linux-x64-1.16.3.tgz -C "$DEP_ROOT"
fi
if [ ! -x "$CAP_ROOT/.venv-sonic/bin/python" ]; then
 python3 -m venv "$CAP_ROOT/.venv-sonic"
fi
PY="$CAP_ROOT/.venv-sonic/bin/python"
"$PY" -m pip install 'numpy==1.26.4' 'scipy==1.15.3' 'mujoco==3.3.7' 'PyYAML==6.0.2' 'pyzmq==27.1.0' 'msgpack==1.1.1' 'opencv-python==4.11.0.86' 'cyclonedds==0.10.2' 'huggingface_hub==0.34.4'
"$PY" -m pip install --no-deps -e "$REPO/external_dependencies/unitree_sdk2_python"
"$PY" -m pip freeze > "$CAP_ROOT/setup-logs/sonic-python-freeze.txt"
