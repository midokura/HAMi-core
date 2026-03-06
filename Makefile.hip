# Copyright 2024 HAMi Authors
#
# Makefile for libamvgpu - AMD GPU memory virtualization library.
#
# Usage:
#   make -f Makefile.hip                # Build library
#   make -f Makefile.hip test           # Build test program
#   make -f Makefile.hip clean          # Clean build artifacts
#   make -f Makefile.hip install        # Install to /usr/local/lib
#
# Override ROCm path:
#   make -f Makefile.hip ROCM_HOME=/opt/rocm-6.4.0
#

ROCM_HOME ?= /opt/rocm
BUILD_DIR ?= build-hip
BUILD_TYPE ?= Release
# CMakeLists.txt.hip is the AMD/HIP version (CMakeLists.txt is NVIDIA/CUDA)
CMAKE_FILE ?= CMakeLists.txt.hip

.PHONY: all test clean install

all:
	@mkdir -p $(BUILD_DIR)
	@# Copy CMakeLists.txt.hip as CMakeLists.txt into build dir with a symlink
	@# to the source tree, so cmake picks up the HIP build instead of CUDA.
	@ln -snf ../$(CMAKE_FILE) $(BUILD_DIR)/CMakeLists.txt
	@ln -snf ../src $(BUILD_DIR)/src
	cd $(BUILD_DIR) && cmake \
		-DROCM_HOME=$(ROCM_HOME) \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		.
	cd $(BUILD_DIR) && $(MAKE)
	@echo ""
	@echo "Built: $(BUILD_DIR)/libamvgpu.so (LD_AUDIT mode)"
	@echo ""
	@echo "Usage:"
	@echo "  LD_AUDIT=$(BUILD_DIR)/libamvgpu.so \\"
	@echo "  HIP_DEVICE_MEMORY_LIMIT_0=48G \\"
	@echo "  ROC_GLOBAL_CU_MASK=0:0xFFFFFFFFFFFFFFFFFF \\"
	@echo "  ./your_app"

test: all
	$(ROCM_HOME)/bin/hipcc \
		-D__HIP_PLATFORM_AMD__ \
		-I$(ROCM_HOME)/include \
		-x hip src/hip/test_memory_limit.c \
		-o $(BUILD_DIR)/test_memory_limit \
		-ldl
	@echo ""
	@echo "Run test:"
	@echo "  LD_AUDIT=$(BUILD_DIR)/libamvgpu.so \\"
	@echo "  HIP_DEVICE_MEMORY_LIMIT_0=1G \\"
	@echo "  LIBHIP_LOG_LEVEL=3 \\"
	@echo "  $(BUILD_DIR)/test_memory_limit"

clean:
	rm -rf $(BUILD_DIR)

install: all
	install -m 755 $(BUILD_DIR)/libamvgpu.so /usr/local/lib/
	ldconfig
	@echo "Installed libamvgpu.so to /usr/local/lib/"
