#
# Copyright 2024 HAMi Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# CMake build configuration for libamvgpu - AMD GPU memory virtualization.
#
# Build:
#   mkdir build-hip && cd build-hip
#   cmake -DROCM_HOME=/opt/rocm ../ -f CMakeLists.txt.hip
#   make
#
# Output: libamvgpu.so
#

cmake_minimum_required(VERSION 3.10)
project(libamvgpu C)

# ROCm installation path
if(NOT DEFINED ROCM_HOME)
    if(DEFINED ENV{ROCM_HOME})
        set(ROCM_HOME $ENV{ROCM_HOME})
    elseif(EXISTS "/opt/rocm")
        set(ROCM_HOME "/opt/rocm")
    else()
        message(FATAL_ERROR
            "ROCM_HOME not set and /opt/rocm not found. "
            "Set -DROCM_HOME=/path/to/rocm")
    endif()
endif()
message(STATUS "ROCM_HOME: ${ROCM_HOME}")

# Compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -D_GNU_SOURCE -std=gnu11 -Wall -Wextra -Wno-unused-parameter")

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2 -fvisibility=hidden")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0 -DDEBUG")
endif()

# Git version info
execute_process(
    COMMAND git rev-parse --short HEAD
    OUTPUT_VARIABLE GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(GIT_HASH)
    add_definitions(-DGIT_HASH="${GIT_HASH}")
endif()

# LD_AUDIT version (default, works on ROCm 7.x)
set(LIBAMVGPU_AUDIT_SOURCES
    src/hip/libamvgpu_audit.c
    src/multiprocess/hip_multiprocess_memory_limit.c
)

# Legacy LD_PRELOAD version (broken on ROCm 7.x, kept for reference)
set(LIBAMVGPU_PRELOAD_SOURCES
    src/hip/libamvgpu.c
    src/hip/hook.c
    src/hip/memory.c
    src/multiprocess/hip_multiprocess_memory_limit.c
)

# Include directories
set(LIBAMVGPU_INCLUDES
    src/include
    src/multiprocess
    ${ROCM_HOME}/include
    ${ROCM_HOME}/include/hip
)

# --- LD_AUDIT target (primary) ---
add_library(amvgpu SHARED ${LIBAMVGPU_AUDIT_SOURCES})
target_include_directories(amvgpu PRIVATE ${LIBAMVGPU_INCLUDES})
target_link_libraries(amvgpu dl pthread)
set_target_properties(amvgpu PROPERTIES
    OUTPUT_NAME "amvgpu"
    VERSION 1.0.0
    SOVERSION 1
)

# --- Legacy LD_PRELOAD target (optional) ---
add_library(amvgpu_preload SHARED ${LIBAMVGPU_PRELOAD_SOURCES})
target_include_directories(amvgpu_preload PRIVATE ${LIBAMVGPU_INCLUDES})
target_link_libraries(amvgpu_preload dl pthread)
set_target_properties(amvgpu_preload PROPERTIES
    OUTPUT_NAME "amvgpu_preload"
    EXCLUDE_FROM_ALL TRUE
)

# Strip local symbols in release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_custom_command(TARGET amvgpu POST_BUILD
        COMMAND strip -x $<TARGET_FILE:amvgpu>
        COMMENT "Stripping local symbols from libamvgpu.so"
    )
endif()

# Install
install(TARGETS amvgpu
    LIBRARY DESTINATION lib
)

message(STATUS "")
message(STATUS "=== libamvgpu (AMD GPU memory virtualization) ===")
message(STATUS "  ROCm:      ${ROCM_HOME}")
message(STATUS "  Build:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Git:       ${GIT_HASH}")
message(STATUS "")
