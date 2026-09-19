#!/usr/bin/env bash
# Project-local source and TensorRT staging only. Run on authorized Linux host.
set -euo pipefail
CAP_ROOT="${1:?Pass the project directory}"
DEP_ROOT="$CAP_ROOT/deps"
mkdir -p "$DEP_ROOT/downloads" "$CAP_ROOT/setup-logs"
REPO="$DEP_ROOT/GR00T-WholeBodyControl"
PIN=087f9ac01d46f6d8e4d0b73c01ae64799f292a38
if [ ! -d "$REPO/.git" ]; then
    git clone --depth 1 --filter=blob:none --no-checkout https://github.com/NVlabs/GR00T-WholeBodyControl.git "$REPO"
fi
git -C "$REPO" fetch --depth 1 origin "$PIN"
git -C "$REPO" sparse-checkout set gear_sonic gear_sonic_deploy external_dependencies install_scripts
GIT_LFS_SKIP_SMUDGE=1 git -C "$REPO" -c filter.lfs.required=false -c filter.lfs.process= -c filter.lfs.smudge= checkout --detach "$PIN"
ARCHIVE="$DEP_ROOT/downloads/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz"
if [ ! -f "$ARCHIVE" ]; then
    curl --fail --location --retry 3 --connect-timeout 20 --max-time 1200 --continue-at - \
      https://developer.download.nvidia.com/compute/machine-learning/tensorrt/10.13.3/tars/TensorRT-10.13.3.9.Linux.x86_64-gnu.cuda-12.9.tar.gz \
      --output "$ARCHIVE.part"
    mv "$ARCHIVE.part" "$ARCHIVE"
fi
if [ ! -f "$DEP_ROOT/TensorRT-10.13.3.9/include/NvInfer.h" ]; then
    tar -xzf "$ARCHIVE" -C "$DEP_ROOT"
fi
sha256sum "$ARCHIVE" > "$CAP_ROOT/setup-logs/tensorrt-archive.sha256"
git -C "$REPO" rev-parse HEAD > "$CAP_ROOT/setup-logs/upstream-commit.txt"
du -sh "$DEP_ROOT/TensorRT-10.13.3.9" "$REPO"
