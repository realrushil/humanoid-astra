#!/usr/bin/env bash
set -euo pipefail
CAP_ROOT="${1:?Pass project root}"
DEP_ROOT="$CAP_ROOT/deps"
REPO="$DEP_ROOT/GR00T-WholeBodyControl"
export TensorRT_ROOT="$DEP_ROOT/TensorRT-10.13.3.9"
export onnxruntime_ROOT="$DEP_ROOT/onnxruntime-linux-x64-1.16.3"
export HAS_ROS2=0
python3 - "$REPO" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1])/'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref'
p=root/'src/g1_deploy_onnx_ref.cpp'
s=p.read_text()
old='ChannelFactory::Instance()->Init(0, networkInterface);'
new='''if (networkInterface != "lo") {
            throw std::runtime_error("G1CAP simulation build requires loopback interface lo");
        }
        ChannelFactory::Instance()->Init(1, networkInterface);'''
if old in s:
 s=s.replace(old,new);p.write_text(s)
elif new not in s: raise SystemExit('unrecognized source; refusing DDS patch')
p=root/'include/dex3_hands.hpp';s=p.read_text();s=s.replace('Init(0,','Init(1,');p.write_text(s)
PY
git -C "$REPO" diff > "$CAP_ROOT/setup-logs/sonic-simulation.patch"
cmake -S "$REPO/gear_sonic_deploy" -B "$REPO/gear_sonic_deploy/build" \
 -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$DEP_ROOT/sysroot/usr" \
 -DZMQ_INCLUDE_DIR="$DEP_ROOT/sysroot/usr/include" -DZMQ_LIBRARY=/usr/lib/x86_64-linux-gnu/libzmq.so.5 \
 -DMSGPACK_INCLUDE_DIR="$DEP_ROOT/sysroot/usr/include" \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build "$REPO/gear_sonic_deploy/build" --target g1_deploy_onnx_ref -j 4
