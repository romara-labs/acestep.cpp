#!/bin/bash
# Build with both ROCm/HIP and Vulkan in one binary, so ace-synth --hybrid
# can put each MiniMax Music 3 stage on the device that suits it:
# the AR stage is faster under ROCm, the DiT is ~3x faster under RADV.
#
# Needs ROCm (hipcc + hipBLAS) and the Vulkan SDK (glslc + headers).
# AMDGPU_TARGETS defaults to gfx1100 (RDNA3, RX 7900); override with
#   AMDGPU_TARGETS=gfx1030 ./buildhybrid.sh

set -eu

TARGETS="${AMDGPU_TARGETS:-gfx1100}"
ROCM="${ROCM_PATH:-/opt/rocm}"

rm -rf build-hybrid
mkdir build-hybrid
cd build-hybrid

HIPCXX="$ROCM/lib/llvm/bin/clang++" HIP_PATH="$ROCM" \
cmake .. -DGGML_HIP=ON -DGGML_VULKAN=ON \
         -DAMDGPU_TARGETS="$TARGETS" -DGPU_TARGETS="$TARGETS" \
         -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j "$(nproc)"
