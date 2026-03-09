/*
 * Copyright 2024 HAMi Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * AMD GPU (HIP) memory virtualization via LD_AUDIT.
 *
 * This library uses the LD_AUDIT interface (la_symbind64) to intercept
 * HIP memory allocation calls. Unlike LD_PRELOAD, LD_AUDIT intercepts
 * at the dynamic linker level and can selectively redirect only
 * cross-library bindings, leaving HIP-internal calls untouched.
 *
 * This is required for ROCm 7.x, where LD_PRELOAD breaks HIP runtime
 * initialization due to internal dlsym/symbol resolution conflicts.
 *
 * Key design:
 *   - la_objopen: Identifies libamdhip64.so by its cookie ID
 *   - la_symbind64: Captures real function pointers from sym->st_value,
 *     redirects non-HIP callers to our wrappers, passes through HIP
 *     internal calls unchanged
 *   - Multi-process memory tracking via shared memory region (mmap'd file)
 *
 * Usage:
 *   LD_AUDIT=libamvgpu.so HIP_DEVICE_MEMORY_LIMIT_0=48G ./my_app
 *
 * Environment variables:
 *   HIP_DEVICE_MEMORY_LIMIT       - Global memory limit (with G/M/K suffix)
 *   HIP_DEVICE_MEMORY_LIMIT_{N}   - Per-device memory limit
 *   HIP_DEVICE_MEMORY_SHARED_CACHE - Shared region file path
 *   LIBHIP_LOG_LEVEL              - Log level (1=error, 2=warn, 3=info, 4=debug)
 *   ACTIVE_OOM_KILLER             - Reclaim dead process memory on OOM ("true")
 */

#define _GNU_SOURCE

#include "../include/glibc_compat.h"

#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>

#include "../include/hip_log_utils.h"
#include "../multiprocess/hip_multiprocess_memory_limit.h"
#include "alloc_tracker.h"

/* HIP error codes (subset - we don't link against HIP) */
typedef int hipError_t;
#define hipSuccess 0
#define hipErrorOutOfMemory 2
#define hipErrorNotInitialized 3

typedef void *hipStream_t;

/* ====================================================================
 * Real HIP function pointers - captured from la_symbind64
 * ==================================================================== */

static hipError_t (*real_hipMalloc)(void **, size_t) = NULL;
static hipError_t (*real_hipFree)(void *) = NULL;
static hipError_t (*real_hipMemGetInfo)(size_t *, size_t *) = NULL;
static hipError_t (*real_hipSetDevice)(int) = NULL;
static hipError_t (*real_hipMallocManaged)(void **, size_t, unsigned int) = NULL;
static hipError_t (*real_hipMallocAsync)(void **, size_t, hipStream_t) = NULL;
static hipError_t (*real_hipFreeAsync)(void *, hipStream_t) = NULL;
static hipError_t (*real_hipMallocPitch)(void **, size_t *, size_t, size_t) = NULL;
static hipError_t (*real_hipExtMallocWithFlags)(void **, size_t, unsigned int) = NULL;

/* Current device per-thread */
static __thread int current_device = 0;

/* Shared region initialization state */
static int g_shrreg_ready = 0;

/* Pointer-to-size allocation tracker.
 *
 * We use a GCC atomic spinlock instead of pthread_mutex because
 * LD_AUDIT's la_symbind64 intercepts ALL symbol bindings.  When
 * pthread_mutex_lock is lazily resolved via PLT, the dynamic linker
 * re-enters la_symbind64 while holding its internal lock, causing
 * deadlock.  Atomic builtins are compiler intrinsics with no PLT. */
static alloc_tracker_t g_alloc_tracker;
static volatile int g_tracker_initialized = 0;
static volatile int g_tracker_spinlock = 0;

static inline void tracker_lock(void) {
    while (__sync_lock_test_and_set(&g_tracker_spinlock, 1)) {
        while (g_tracker_spinlock)
            __builtin_ia32_pause();
    }
}

static inline void tracker_unlock(void) {
    __sync_lock_release(&g_tracker_spinlock);
}

/* Must be called under tracker_lock() or during single-threaded init */
static inline alloc_tracker_t *get_tracker(void) {
    if (!g_tracker_initialized) {
        alloc_tracker_init(&g_alloc_tracker);
        __sync_synchronize();
        g_tracker_initialized = 1;
    }
    return &g_alloc_tracker;
}

/* ====================================================================
 * Shared region lazy initialization
 *
 * Cannot init in la_version() because the dynamic linker is still
 * resolving symbols at that point. Defer to first memory operation.
 * ==================================================================== */

static void ensure_shrreg_init(void) {
    if (g_shrreg_ready)
        return;
    if (hip_shrreg_init() == 0) {
        g_shrreg_ready = 1;
        LOG_INFO("Shared region initialized for LD_AUDIT mode");
    }
}

/* ====================================================================
 * Wrapper functions
 * ==================================================================== */

static hipError_t wrap_hipMalloc(void **ptr, size_t size) {
    if (!real_hipMalloc) return hipErrorNotInitialized;

    ensure_shrreg_init();
    int dev = current_device;

    if (hip_get_device_memory_limit(dev) == 0)
        return real_hipMalloc(ptr, size);

    if (hip_shrreg_lock() != 0)
        return real_hipMalloc(ptr, size);

    if (hip_oom_check(dev, size) != 0) {
        hip_shrreg_unlock();
        LOG_WARN("hipMalloc: OOM rejected %zu bytes on device %d", size, dev);
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    hipError_t ret = real_hipMalloc(ptr, size);

    if (ret == hipSuccess) {
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        tracker_lock();
        alloc_tracker_insert(get_tracker(), *ptr, size, dev);
        tracker_unlock();
        LOG_DEBUG("hipMalloc: %zu bytes -> %p (dev %d)", size, *ptr, dev);
    }

    return ret;
}

static hipError_t wrap_hipFree(void *ptr) {
    if (!real_hipFree) return hipErrorNotInitialized;
    if (ptr == NULL) return real_hipFree(ptr);

    int dev = current_device;

    if (hip_get_device_memory_limit(dev) == 0)
        return real_hipFree(ptr);

    /* Look up exact allocation size from tracker */
    int alloc_dev = dev;
    tracker_lock();
    size_t tracked_size = alloc_tracker_remove(get_tracker(), ptr, &alloc_dev);
    tracker_unlock();

    hipError_t ret = real_hipFree(ptr);

    if (ret == hipSuccess && tracked_size > 0) {
        if (hip_shrreg_lock() == 0) {
            hip_rm_device_memory_usage(alloc_dev, tracked_size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        LOG_DEBUG("hipFree: %p freed %zu bytes (dev %d, tracked)", ptr, tracked_size, alloc_dev);
    } else if (ret == hipSuccess && tracked_size == 0) {
        LOG_WARN("hipFree: %p not found in tracker (dev %d)", ptr, dev);
    }

    return ret;
}

static hipError_t wrap_hipMemGetInfo(size_t *free_mem, size_t *total_mem) {
    if (!real_hipMemGetInfo) return hipErrorNotInitialized;

    hipError_t ret = real_hipMemGetInfo(free_mem, total_mem);

    if (ret == hipSuccess) {
        ensure_shrreg_init();
        uint64_t limit = hip_get_device_memory_limit(current_device);
        if (limit > 0) {
            uint64_t usage = 0;
            if (hip_shrreg_lock() == 0) {
                usage = hip_get_device_memory_usage(current_device);
                hip_shrreg_unlock();
            }
            *total_mem = (size_t)limit;
            *free_mem = (usage < limit) ? (size_t)(limit - usage) : 0;
            LOG_DEBUG("hipMemGetInfo: free=%zu total=%zu (virtualized)",
                      *free_mem, *total_mem);
        }
    }

    return ret;
}

static hipError_t wrap_hipSetDevice(int device) {
    if (!real_hipSetDevice) return hipErrorNotInitialized;

    hipError_t ret = real_hipSetDevice(device);
    if (ret == hipSuccess) {
        current_device = device;
        LOG_DEBUG("hipSetDevice: %d", device);
    }
    return ret;
}

static hipError_t wrap_hipMallocManaged(void **ptr, size_t size,
                                         unsigned int flags) {
    if (!real_hipMallocManaged) return hipErrorNotInitialized;

    ensure_shrreg_init();
    int dev = current_device;

    if (hip_get_device_memory_limit(dev) == 0)
        return real_hipMallocManaged(ptr, size, flags);

    if (hip_shrreg_lock() != 0)
        return real_hipMallocManaged(ptr, size, flags);

    if (hip_oom_check(dev, size) != 0) {
        hip_shrreg_unlock();
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    hipError_t ret = real_hipMallocManaged(ptr, size, flags);

    if (ret == hipSuccess) {
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        tracker_lock();
        alloc_tracker_insert(get_tracker(), *ptr, size, dev);
        tracker_unlock();
        LOG_DEBUG("hipMallocManaged: %zu bytes -> %p (dev %d)", size, *ptr, dev);
    }

    return ret;
}

static hipError_t wrap_hipMallocAsync(void **ptr, size_t size,
                                       hipStream_t stream) {
    if (!real_hipMallocAsync) return hipErrorNotInitialized;

    ensure_shrreg_init();
    int dev = current_device;

    if (hip_get_device_memory_limit(dev) == 0)
        return real_hipMallocAsync(ptr, size, stream);

    if (hip_shrreg_lock() != 0)
        return real_hipMallocAsync(ptr, size, stream);

    if (hip_oom_check(dev, size) != 0) {
        hip_shrreg_unlock();
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    hipError_t ret = real_hipMallocAsync(ptr, size, stream);

    if (ret == hipSuccess) {
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        tracker_lock();
        alloc_tracker_insert(get_tracker(), *ptr, size, dev);
        tracker_unlock();
        LOG_DEBUG("hipMallocAsync: %zu bytes (dev %d)", size, dev);
    }

    return ret;
}

static hipError_t wrap_hipFreeAsync(void *ptr, hipStream_t stream) {
    if (!real_hipFreeAsync) return hipErrorNotInitialized;
    if (ptr == NULL) return real_hipFreeAsync(ptr, stream);

    int dev = current_device;
    if (hip_get_device_memory_limit(dev) == 0)
        return real_hipFreeAsync(ptr, stream);

    /* Look up size before async free */
    int alloc_dev = dev;
    tracker_lock();
    size_t tracked_size = alloc_tracker_remove(get_tracker(), ptr, &alloc_dev);
    tracker_unlock();

    hipError_t ret = real_hipFreeAsync(ptr, stream);

    if (ret == hipSuccess && tracked_size > 0) {
        if (hip_shrreg_lock() == 0) {
            hip_rm_device_memory_usage(alloc_dev, tracked_size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        LOG_DEBUG("hipFreeAsync: %p freed %zu bytes (dev %d, tracked)", ptr, tracked_size, alloc_dev);
    }

    return ret;
}

static hipError_t wrap_hipMallocPitch(void **ptr, size_t *pitch,
                                       size_t width, size_t height) {
    if (!real_hipMallocPitch) return hipErrorNotInitialized;

    ensure_shrreg_init();
    int dev = current_device;

    if (hip_get_device_memory_limit(dev) == 0)
        return real_hipMallocPitch(ptr, pitch, width, height);

    size_t estimated = width * height;

    if (hip_shrreg_lock() != 0)
        return real_hipMallocPitch(ptr, pitch, width, height);

    if (hip_oom_check(dev, estimated) != 0) {
        hip_shrreg_unlock();
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    hipError_t ret = real_hipMallocPitch(ptr, pitch, width, height);

    if (ret == hipSuccess && pitch != NULL) {
        size_t actual = (*pitch) * height;
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, actual, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        tracker_lock();
        alloc_tracker_insert(get_tracker(), *ptr, actual, dev);
        tracker_unlock();
        LOG_DEBUG("hipMallocPitch: %zux%zu (pitch=%zu, total=%zu) dev %d",
                  width, height, *pitch, actual, dev);
    }

    return ret;
}

static hipError_t wrap_hipExtMallocWithFlags(void **ptr, size_t size,
                                              unsigned int flags) {
    if (!real_hipExtMallocWithFlags) return hipErrorNotInitialized;

    ensure_shrreg_init();
    int dev = current_device;

    if (hip_get_device_memory_limit(dev) == 0)
        return real_hipExtMallocWithFlags(ptr, size, flags);

    if (hip_shrreg_lock() != 0)
        return real_hipExtMallocWithFlags(ptr, size, flags);

    if (hip_oom_check(dev, size) != 0) {
        hip_shrreg_unlock();
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    hipError_t ret = real_hipExtMallocWithFlags(ptr, size, flags);

    if (ret == hipSuccess) {
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        tracker_lock();
        alloc_tracker_insert(get_tracker(), *ptr, size, dev);
        tracker_unlock();
    }

    return ret;
}

/* ====================================================================
 * LD_AUDIT interface
 * ==================================================================== */

__attribute__((visibility("default")))
unsigned int la_version(unsigned int version) {
    LOG_INFO("libamvgpu LD_AUDIT loaded (pid %d, LAV %u)", getpid(), version);
    return LAV_CURRENT;
}

/* Cookie to identify libamdhip64.so */
static int hip_cookie = -1;

__attribute__((visibility("default")))
unsigned int la_objopen(struct link_map *map, Lmid_t lmid,
                        uintptr_t *cookie) {
    static int counter = 0;
    int id = counter++;
    *cookie = (uintptr_t)id;

    if (map->l_name && strstr(map->l_name, "libamdhip64")) {
        hip_cookie = id;
        LOG_DEBUG("Identified HIP library: %s (cookie %d)", map->l_name, id);
    }

    return LA_FLG_BINDTO | LA_FLG_BINDFROM;
}

/*
 * Symbol binding interceptor.
 *
 * For each HIP function we care about:
 *   1. Capture the real function pointer from sym->st_value (first time only)
 *   2. If the caller is HIP itself (from_hip), pass through unchanged
 *      to avoid breaking HIP internal initialization
 *   3. If the caller is external (PyTorch, user app, etc.), redirect to wrapper
 *
 * If no memory limit is configured (all limits == 0), skip all interception.
 */
#if __ELF_NATIVE_CLASS == 64
__attribute__((visibility("default")))
uintptr_t la_symbind64(Elf64_Sym *sym, unsigned int ndx,
                        uintptr_t *refcook, uintptr_t *defcook,
                        unsigned int *flags, const char *symname)
#else
__attribute__((visibility("default")))
uintptr_t la_symbind32(Elf32_Sym *sym, unsigned int ndx,
                        uintptr_t *refcook, uintptr_t *defcook,
                        unsigned int *flags, const char *symname)
#endif
{
    if (!symname)
        return sym->st_value;

    /* Skip interception if caller is HIP itself */
    int from_hip = ((int)(uintptr_t)*refcook == hip_cookie);

    /*
     * Macro to handle each intercepted function:
     * - Capture real pointer on first encounter
     * - Return wrapper for external callers, real for HIP-internal
     */
    #define INTERCEPT(func_name, real_ptr, wrapper_fn) \
        if (strcmp(symname, #func_name) == 0) { \
            if (!(real_ptr)) \
                (real_ptr) = (void *)sym->st_value; \
            return from_hip ? sym->st_value : (uintptr_t)(wrapper_fn); \
        }

    INTERCEPT(hipMalloc,             real_hipMalloc,             wrap_hipMalloc)
    INTERCEPT(hipFree,               real_hipFree,               wrap_hipFree)
    INTERCEPT(hipMemGetInfo,         real_hipMemGetInfo,         wrap_hipMemGetInfo)
    INTERCEPT(hipSetDevice,          real_hipSetDevice,          wrap_hipSetDevice)
    INTERCEPT(hipMallocManaged,      real_hipMallocManaged,      wrap_hipMallocManaged)
    INTERCEPT(hipMallocAsync,        real_hipMallocAsync,        wrap_hipMallocAsync)
    INTERCEPT(hipFreeAsync,          real_hipFreeAsync,          wrap_hipFreeAsync)
    INTERCEPT(hipMallocPitch,        real_hipMallocPitch,        wrap_hipMallocPitch)
    INTERCEPT(hipExtMallocWithFlags, real_hipExtMallocWithFlags, wrap_hipExtMallocWithFlags)

    #undef INTERCEPT

    return sym->st_value;
}
