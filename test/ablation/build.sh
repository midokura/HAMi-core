#!/bin/bash
# Build libvgpu.so from the current branch using Docker.
#
# Why Docker?
#   libvgpu.so is injected via LD_PRELOAD into GPU containers (e.g., gpu_burn,
#   vLLM). These containers typically use older base images (Ubuntu 20.04) with
#   glibc 2.31. Building on the host (Ubuntu 24.04, glibc 2.39) would produce
#   a binary that fails to load inside containers. Docker ensures glibc
#   compatibility.
#
# Usage:
#   bash test/ablation/build.sh [output_path]
#
# Output:
#   /tmp/libvgpu.so (default) or the specified output_path
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIBVGPU_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT="${1:-/tmp/libvgpu.so}"
OUTPUT_DIR="$(dirname "$OUTPUT")"
OUTPUT_NAME="$(basename "$OUTPUT")"

echo "=== Building libvgpu.so ==="
echo "Source: ${LIBVGPU_ROOT}"
echo "Output: ${OUTPUT}"
echo ""

docker run --rm \
  -e DEBIAN_FRONTEND=noninteractive -e TZ=UTC \
  -v "${LIBVGPU_ROOT}:/libvgpu:ro" \
  -v "${OUTPUT_DIR}:/output" \
  nvidia/cuda:12.2.0-devel-ubuntu20.04 \
  bash -c '
    apt-get -y update > /dev/null 2>&1
    apt-get -y install cmake build-essential > /dev/null 2>&1
    mkdir -p /tmp/build && cd /tmp/build
    cmake /libvgpu \
      -DDLSYM_HOOK_ENABLE=1 \
      -DMULTIPROCESS_LIMIT_ENABLE=1 \
      -DHOOK_MEMINFO_ENABLE=1 \
      -DHOOK_NVML_ENABLE=1 \
      -DCMAKE_BUILD_TYPE=Debug \
      > /dev/null 2>&1
    make -j$(nproc) 2>&1 | tail -5
    cp /tmp/build/libvgpu.so /output/'"${OUTPUT_NAME}"'
    echo ""
    echo "BUILD OK"
  '

echo ""
echo "Output: ${OUTPUT}"
echo "md5sum: $(md5sum "${OUTPUT}" | awk '{print $1}')"
echo ""
echo "To deploy:"
echo "  sudo cp ${OUTPUT} /usr/local/vgpu/libvgpu.so"
