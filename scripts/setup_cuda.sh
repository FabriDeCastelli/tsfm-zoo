#!/usr/bin/env bash
# CUDA 12.8 toolkit for TiRex-2's JIT-built kernels (flashrnn), matching torch's cu128 build.
# The system nvcc is 13.x (torch refuses a major-version mismatch; the driver supports 12.8),
# and the pip nvcc wheel ships no nvcc driver. NVIDIA's redist archives provide nvcc, the
# CUDA runtime and CCCL; cuBLAS headers and libraries come from torch's own pip package.
# Usage: TSFM_ENV=main scripts/setup_cuda.sh   (installs into $CUDA_HOME from .envrc)
set -euo pipefail
cd "$(dirname "$0")/.."
export TSFM_ENV=${TSFM_ENV:-main}
source .envrc
REDIST=https://developer.download.nvidia.com/compute/cuda/redist
ARCHIVES=(
  cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz
  cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz
  cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz
)
mkdir -p "$CUDA_HOME"
for a in "${ARCHIVES[@]}"; do
  curl -fsSL "$REDIST/$a" | tar -xJ -C "$CUDA_HOME" --strip-components=1
done
[ -e "$CUDA_HOME/lib64" ] || ln -s lib "$CUDA_HOME/lib64"
CUBLAS=$("$UV_PROJECT_ENVIRONMENT/bin/python" -c "import nvidia.cublas, os; print(os.path.dirname(nvidia.cublas.__file__))")
ln -sf "$CUBLAS"/include/* "$CUDA_HOME/include/"
for lib in "$CUBLAS"/lib/lib*.so.*; do
  ln -sf "$lib" "$CUDA_HOME/lib/"
  ln -sf "$lib" "$CUDA_HOME/lib/$(basename "${lib%%.so.*}").so"
done
"$CUDA_HOME/bin/nvcc" --version | tail -1
