/*
 * Copyright 2024 HAMi Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Test program for AMD GPU memory limiting via libamvgpu.
 *
 * Build (standalone, without LD_PRELOAD):
 *   hipcc test_memory_limit.c -o test_memory_limit -ldl
 *
 * Run with memory limit:
 *   LD_PRELOAD=./libamvgpu.so \
 *   HIP_DEVICE_MEMORY_LIMIT_0=1G \
 *   LIBHIP_LOG_LEVEL=3 \
 *   ./test_memory_limit
 */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_HIP(call) \
    do { \
        hipError_t err = (call); \
        if (err != hipSuccess) { \
            printf("  -> %s returned: %s (%d)\n", \
                   #call, hipGetErrorString(err), err); \
            if (err == hipErrorOutOfMemory) \
                printf("  -> OOM as expected (memory limit enforced)\n"); \
            else \
                return 1; \
        } else { \
            printf("  -> %s: OK\n", #call); \
        } \
    } while (0)

int main(void) {
    int device_count = 0;
    hipGetDeviceCount(&device_count);
    printf("Device count: %d\n", device_count);

    if (device_count == 0) {
        printf("No GPU devices found\n");
        return 1;
    }

    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("Device 0: %s\n", prop.name);

    hipSetDevice(0);

    /* Query memory info (should be virtualized if limit is set) */
    size_t free_mem = 0, total_mem = 0;
    hipMemGetInfo(&free_mem, &total_mem);
    printf("\nMemory info:\n");
    printf("  Total: %zu MB\n", total_mem / (1024 * 1024));
    printf("  Free:  %zu MB\n", free_mem / (1024 * 1024));

    /* Test 1: Small allocation (should succeed) */
    printf("\nTest 1: Allocate 256 MB\n");
    void *ptr1 = NULL;
    CHECK_HIP(hipMalloc(&ptr1, 256ULL * 1024 * 1024));

    hipMemGetInfo(&free_mem, &total_mem);
    printf("  Free after alloc: %zu MB\n", free_mem / (1024 * 1024));

    /* Test 2: Another small allocation (should succeed if limit > 512M) */
    printf("\nTest 2: Allocate another 256 MB\n");
    void *ptr2 = NULL;
    CHECK_HIP(hipMalloc(&ptr2, 256ULL * 1024 * 1024));

    hipMemGetInfo(&free_mem, &total_mem);
    printf("  Free after alloc: %zu MB\n", free_mem / (1024 * 1024));

    /* Test 3: Large allocation (may trigger OOM if limit is set) */
    printf("\nTest 3: Allocate 2 GB (may OOM)\n");
    void *ptr3 = NULL;
    hipError_t err = hipMalloc(&ptr3, 2ULL * 1024 * 1024 * 1024);
    if (err == hipErrorOutOfMemory) {
        printf("  -> OOM: memory limit is working correctly\n");
    } else if (err == hipSuccess) {
        printf("  -> Allocated (limit > 2.5 GB or no limit set)\n");
        hipFree(ptr3);
    } else {
        printf("  -> Unexpected error: %d\n", err);
    }

    /* Cleanup */
    printf("\nCleanup:\n");
    if (ptr1) CHECK_HIP(hipFree(ptr1));
    if (ptr2) CHECK_HIP(hipFree(ptr2));

    hipMemGetInfo(&free_mem, &total_mem);
    printf("\nFinal memory:\n");
    printf("  Total: %zu MB\n", total_mem / (1024 * 1024));
    printf("  Free:  %zu MB\n", free_mem / (1024 * 1024));

    printf("\nAll tests passed.\n");
    return 0;
}
