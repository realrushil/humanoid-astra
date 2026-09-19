#!/usr/bin/env bash
set -euo pipefail
CAP_ROOT="${1:?Pass project root}"
GPU_INDEX="${2:?Pass the available GPU index explicitly}"
PORT="${3:-5556}"
REPO="$CAP_ROOT/deps/GR00T-WholeBodyControl"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export LD_LIBRARY_PATH="$CAP_ROOT/deps/TensorRT-10.13.3.9/lib:$CAP_ROOT/deps/onnxruntime-linux-x64-1.16.3/lib:$REPO/gear_sonic_deploy/thirdparty/unitree_sdk2/thirdparty/lib/x86_64:${LD_LIBRARY_PATH:-}"
cd "$REPO/gear_sonic_deploy"
exec target/release/g1_deploy_onnx_ref lo \
 policy/sonic_v1_1/model_decoder.onnx reference/example/ \
 --obs-config policy/sonic_v1_1/observation_config.yaml \
 --encoder-file policy/sonic_v1_1/model_encoder.onnx \
 --planner-file planner/target_vel/V2/planner_sonic.onnx \
 --input-type zmq_manager --output-type zmq --zmq-host 127.0.0.1 \
 --zmq-port "$PORT" --zmq-out-port "$((PORT+1))" --disable-crc-check \
 --motor-kp-scale 4,10=1.5 --motor-kd-scale 4,10=1.5
