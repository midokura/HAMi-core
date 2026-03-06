/*
 * Copyright 2024 HAMi Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * HIP library entry table initialization.
 * Resolves real HIP function pointers via dlsym(RTLD_NEXT, ...).
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/hip_hook.h"
#include "../include/hip_log_utils.h"

#include <dlfcn.h>
#include <string.h>

/*
 * HIP function entry table.
 * Each entry maps a function name to its resolved pointer.
 * Terminated by a NULL name sentinel.
 *
 * We hook the memory-related functions that need tracking.
 * Non-memory functions are passed through directly.
 */
hip_entry_t hip_library_entry[] = {
    /* Memory allocation */
    {NULL, "hipMalloc"},
    {NULL, "hipFree"},
    {NULL, "hipMallocManaged"},
    {NULL, "hipMallocAsync"},
    {NULL, "hipFreeAsync"},
    {NULL, "hipHostMalloc"},
    {NULL, "hipHostFree"},
    {NULL, "hipMallocPitch"},
    {NULL, "hipExtMallocWithFlags"},
    {NULL, "hipMallocFromPoolAsync"},

    /* Memory pool management */
    {NULL, "hipMemPoolCreate"},
    {NULL, "hipMemPoolDestroy"},

    /* Memory info */
    {NULL, "hipMemGetInfo"},

    /* Array allocation */
    {NULL, "hipMallocArray"},
    {NULL, "hipFreeArray"},
    {NULL, "hipMalloc3DArray"},

    /* Device management */
    {NULL, "hipInit"},
    {NULL, "hipGetDevice"},
    {NULL, "hipSetDevice"},
    {NULL, "hipGetDeviceCount"},
    {NULL, "hipGetDeviceProperties"},
    {NULL, "hipDeviceGetAttribute"},

    /* Sentinel */
    {NULL, NULL}
};

int hip_hook_init(void) {
    int count = 0;

    for (int i = 0; hip_library_entry[i].name != NULL; i++) {
        hip_library_entry[i].fn_ptr = amvgpu_real_dlsym(RTLD_NEXT,
                                             hip_library_entry[i].name);
        if (hip_library_entry[i].fn_ptr == NULL) {
            LOG_DEBUG("Could not resolve %s (may not be available)",
                      hip_library_entry[i].name);
        } else {
            count++;
            LOG_DEBUG("Resolved %s -> %p",
                      hip_library_entry[i].name,
                      hip_library_entry[i].fn_ptr);
        }
    }

    LOG_INFO("Initialized HIP hook table: %d functions resolved", count);
    return count;
}

void *hip_hook_find(const char *name) {
    for (int i = 0; hip_library_entry[i].name != NULL; i++) {
        if (strcmp(hip_library_entry[i].name, name) == 0)
            return hip_library_entry[i].fn_ptr;
    }
    return NULL;
}
