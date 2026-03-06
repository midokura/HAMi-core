/*
 * Copyright 2024 HAMi Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * AMD GPU (HIP) memory virtualization library - LD_PRELOAD entry point.
 *
 * This library intercepts HIP memory allocation calls to enforce
 * per-device memory limits for GPU virtualization. It is loaded via
 * LD_PRELOAD and uses dlsym(RTLD_NEXT) to forward calls to the real
 * HIP runtime after applying memory limit checks.
 *
 * Usage:
 *   LD_PRELOAD=libamvgpu.so HIP_DEVICE_MEMORY_LIMIT_0=48000m ./my_app
 *
 * Environment variables:
 *   HIP_DEVICE_MEMORY_LIMIT       - Global memory limit (with G/M/K suffix)
 *   HIP_DEVICE_MEMORY_LIMIT_{N}   - Per-device memory limit
 *   HIP_DEVICE_MEMORY_SHARED_CACHE - Shared region file path
 *   LIBHIP_LOG_LEVEL              - Log level (1=error, 2=warn, 3=info, 4=debug)
 *   ACTIVE_OOM_KILLER             - Reclaim dead process memory on OOM ("true")
 *   ROC_GLOBAL_CU_MASK            - CU mask (set externally, not managed here)
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/libamvgpu.h"
#include "../include/hip_hook.h"
#include "../include/hip_log_utils.h"
#include "../multiprocess/hip_multiprocess_memory_limit.h"

#include <dlfcn.h>
#include <pthread.h>
#include <stdio.h>
#include <string.h>

/* Initialization state */
static pthread_once_t init_flag = PTHREAD_ONCE_INIT;
static int g_initialized = 0;
static __thread int g_in_init = 0;  /* Prevent dlsym recursion */

/*
 * Library initialization.
 * Called once on the first intercepted HIP call.
 */
static void library_init(void) {
    g_in_init = 1;
    LOG_INFO("libamvgpu initializing (pid %d)", getpid());

    /* Initialize shared memory region for memory tracking */
    if (hip_shrreg_init() != 0) {
        LOG_ERROR("Failed to initialize shared memory region");
        LOG_ERROR("Memory limiting will be disabled");
        g_in_init = 0;
        return;
    }

    /* Initialize HIP hook table */
    int resolved = hip_hook_init();
    if (resolved == 0) {
        LOG_WARN("No HIP functions resolved - is HIP runtime loaded?");
    }

    g_initialized = 1;
    g_in_init = 0;
    LOG_INFO("libamvgpu initialized: %d HIP functions hooked", resolved);
}

/*
 * Ensure library is initialized.
 * Thread-safe via pthread_once.
 * Returns 1 if currently inside initialization (to prevent recursion).
 */
static inline int ensure_initialized(void) {
    if (g_in_init)
        return 1;  /* Recursion guard: skip hook during init */
    pthread_once(&init_flag, library_init);
    return 0;
}

/*
 * GCC constructor - called when the library is loaded.
 * Logs that the library was loaded but defers full initialization
 * to the first HIP call, since HIP runtime may not be ready yet.
 */
__attribute__((constructor(65535)))
static void on_load(void) {
    LOG_INFO("libamvgpu loaded via LD_PRELOAD (pid %d)", getpid());
}

/*
 * GCC destructor - called when the library is unloaded.
 */
__attribute__((destructor))
static void on_unload(void) {
    if (g_initialized) {
        hip_shrreg_cleanup();
        LOG_INFO("libamvgpu unloaded (pid %d)", getpid());
    }
}

/*
 * Note: We do NOT intercept dlsym itself.
 *
 * LD_PRELOAD already causes our wrapper functions (hipMalloc, hipFree, etc.)
 * to override the real HIP symbols. Frameworks like PyTorch that use
 * dlsym(RTLD_DEFAULT, "hipMalloc") will find our wrappers automatically
 * because LD_PRELOAD symbols take precedence.
 *
 * Intercepting dlsym causes problems with HIP runtime initialization
 * on ROCm 7.x, where the HIP runtime uses dlsym internally.
 */
