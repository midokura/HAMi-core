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

/*
 * Library initialization.
 * Called once on the first intercepted HIP call.
 */
static void library_init(void) {
    LOG_INFO("libamvgpu initializing (pid %d)", getpid());

    /* Initialize shared memory region for memory tracking */
    if (hip_shrreg_init() != 0) {
        LOG_ERROR("Failed to initialize shared memory region");
        LOG_ERROR("Memory limiting will be disabled");
        return;
    }

    /* Initialize HIP hook table */
    int resolved = hip_hook_init();
    if (resolved == 0) {
        LOG_WARN("No HIP functions resolved - is HIP runtime loaded?");
    }

    g_initialized = 1;
    LOG_INFO("libamvgpu initialized: %d HIP functions hooked", resolved);
}

/*
 * Ensure library is initialized.
 * Thread-safe via pthread_once.
 */
static inline void ensure_initialized(void) {
    pthread_once(&init_flag, library_init);
}

/*
 * GCC constructor - called when the library is loaded.
 * Logs that the library was loaded but defers full initialization
 * to the first HIP call, since HIP runtime may not be ready yet.
 */
__attribute__((constructor))
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
 * dlsym interception.
 *
 * When applications or frameworks (e.g., PyTorch) use dlsym to
 * look up HIP functions at runtime, we intercept and return our
 * wrapped versions instead of the real ones.
 *
 * This is critical for frameworks that load HIP dynamically.
 */

/* Mapping of function names to our interceptors */
typedef struct {
    const char *name;
    void *wrapper;
} hook_map_entry_t;

/* Forward declarations of our wrappers (defined in memory.c) */
extern hipError_t hipMalloc(void **, size_t);
extern hipError_t hipFree(void *);
extern hipError_t hipMallocManaged(void **, size_t, unsigned int);
extern hipError_t hipMallocAsync(void **, size_t, hipStream_t);
extern hipError_t hipFreeAsync(void *, hipStream_t);
extern hipError_t hipHostMalloc(void **, size_t, unsigned int);
extern hipError_t hipHostFree(void *);
extern hipError_t hipMallocPitch(void **, size_t *, size_t, size_t);
extern hipError_t hipExtMallocWithFlags(void **, size_t, unsigned int);
extern hipError_t hipSetDevice(int);
extern hipError_t hipMemGetInfo(size_t *, size_t *);

static const hook_map_entry_t hook_map[] = {
    {"hipMalloc",              (void *)hipMalloc},
    {"hipFree",                (void *)hipFree},
    {"hipMallocManaged",       (void *)hipMallocManaged},
    {"hipMallocAsync",         (void *)hipMallocAsync},
    {"hipFreeAsync",           (void *)hipFreeAsync},
    {"hipHostMalloc",          (void *)hipHostMalloc},
    {"hipHostFree",            (void *)hipHostFree},
    {"hipMallocPitch",         (void *)hipMallocPitch},
    {"hipExtMallocWithFlags",  (void *)hipExtMallocWithFlags},
    {"hipSetDevice",           (void *)hipSetDevice},
    {"hipMemGetInfo",          (void *)hipMemGetInfo},
    {NULL, NULL}
};

/*
 * Intercept dlsym calls to return our wrappers for HIP functions.
 *
 * When handle is RTLD_NEXT or RTLD_DEFAULT and the symbol matches
 * a hooked function, we return our wrapper. Otherwise, we forward
 * to the real dlsym.
 */
FUNC_ATTR_VISIBLE
void *dlsym(void *handle, const char *symbol) {
    /* Resolve the real dlsym using dlvsym with GLIBC version */
    static void *(*real_dlsym)(void *, const char *) = NULL;
    if (!real_dlsym) {
        /* Try common GLIBC versions */
        real_dlsym = dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.34");
        if (!real_dlsym)
            real_dlsym = dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.17");
        if (!real_dlsym)
            real_dlsym = dlvsym(RTLD_DEFAULT, "dlsym", "GLIBC_2.2.5");
        if (!real_dlsym) {
            fprintf(stderr,
                    "[HAMI-core-hip] FATAL: cannot resolve real dlsym\n");
            return NULL;
        }
    }

    /* Check if this is a HIP function we should intercept */
    if (handle == RTLD_NEXT || handle == RTLD_DEFAULT) {
        for (int i = 0; hook_map[i].name != NULL; i++) {
            if (strcmp(symbol, hook_map[i].name) == 0) {
                ensure_initialized();
                LOG_DEBUG("dlsym intercepted: %s", symbol);
                return hook_map[i].wrapper;
            }
        }
    }

    return real_dlsym(handle, symbol);
}
