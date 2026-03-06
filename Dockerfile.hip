# Build environment for libamvgpu.so (AMD GPU memory virtualization)
#
# Uses Ubuntu 22.04 + ROCm 6.2 dev image to match the target runtime
# (rocm/pytorch:rocm6.2_ubuntu22.04). This ensures glibc compatibility.
#
# Usage:
#   docker build -f Dockerfile.hip -t libamvgpu-builder .
#   docker run --rm -v $(pwd)/dist:/dist libamvgpu-builder
#
# Output: dist/libamvgpu.so
#
# For development (mount source, iterate):
#   docker run --rm -it -v $(pwd):/workspace libamvgpu-builder bash
#   cd /workspace && make -f Makefile.hip clean all
#

FROM rocm/dev-ubuntu-22.04:6.2 AS builder

RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
        cmake build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY . /build/

RUN make -f Makefile.hip clean all && \
    make -f Makefile.hip test

# Verify: no GLIBC_2.38 dependency
RUN ! objdump -T build-hip/libamvgpu.so | grep -q "GLIBC_2.38" && \
    echo "OK: No GLIBC_2.38 dependency" || \
    (echo "ERROR: GLIBC_2.38 dependency found!" && exit 1)

# Verify: LD_AUDIT symbols exported
RUN nm -D build-hip/libamvgpu.so | grep -q "T la_version" && \
    nm -D build-hip/libamvgpu.so | grep -q "T la_symbind64" && \
    echo "OK: LD_AUDIT symbols exported"

# --- Output stage ---
FROM scratch AS output
COPY --from=builder /build/build-hip/libamvgpu.so /libamvgpu.so
COPY --from=builder /build/build-hip/test_memory_limit /test_memory_limit

# Default: copy artifacts to /dist mount
FROM builder AS default
CMD ["sh", "-c", "mkdir -p /dist && cp build-hip/libamvgpu.so build-hip/test_memory_limit /dist/ && echo 'Copied to /dist:' && ls -la /dist/"]
