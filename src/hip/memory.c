/*
 * Copyright 2024 HAMi Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * HIP memory allocation interception.
 *
 * Intercepts hipMalloc, hipFree, and related functions to track GPU
 * memory usage and enforce per-device memory limits. Each allocation
 * is checked against the configured limit before being forwarded to
 * the real HIP runtime.
 *
 * Thread safety: All shared region access is protected by the
 * semaphore-based lock in hip_multiprocess_memory_limit.c.
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
#include <string.h>

#include "alloc_tracker.h"

/* Pointer-to-size allocation tracker */
static alloc_tracker_t g_alloc_tracker;
static int g_tracker_initialized = 0;
static pthread_mutex_t g_tracker_mutex = PTHREAD_MUTEX_INITIALIZER;

static inline alloc_tracker_t *get_tracker(void) {
    if (!g_tracker_initialized) {
        pthread_mutex_lock(&g_tracker_mutex);
        if (!g_tracker_initialized) {
            alloc_tracker_init(&g_alloc_tracker);
            g_tracker_initialized = 1;
        }
        pthread_mutex_unlock(&g_tracker_mutex);
    }
    return &g_alloc_tracker;
}

/* Current device index for this thread */
static __thread int current_device = 0;

/* Ensure shared region is initialized before any memory operation */
static int g_mem_init_result = -1;

static void do_mem_init(void) {
    g_mem_init_result = hip_shrreg_init();
}

static inline int ensure_initialized(void) {
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, do_mem_init);
    return g_mem_init_result;
}

/* ====================================================================
 * HIP Memory Function Interceptors
 * ==================================================================== */

/*
 * hipMalloc - Primary device memory allocation.
 */
FUNC_ATTR_VISIBLE
hipError_t hipMalloc(void **ptr, size_t size) {
    REAL_FUNC(hipMalloc);

    if (hip_get_device_memory_limit(current_device) == 0) {
        return real_hipMalloc(ptr, size);
    }

    hipError_t ret;
    int dev = current_device;

    if (hip_shrreg_lock() != 0)
        return real_hipMalloc(ptr, size);

    if (hip_oom_check(dev, size) != 0) {
        hip_shrreg_unlock();
        LOG_WARN("hipMalloc: OOM rejected %zu bytes on device %d", size, dev);
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    ret = real_hipMalloc(ptr, size);

    if (ret == hipSuccess) {
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        pthread_mutex_lock(&g_tracker_mutex);
        alloc_tracker_insert(get_tracker(), *ptr, size, dev);
        pthread_mutex_unlock(&g_tracker_mutex);
        LOG_DEBUG("hipMalloc: %zu bytes -> %p (dev %d)", size, *ptr, dev);
    }

    return ret;
}

/*
 * hipFree - Device memory deallocation.
 *
 * Uses pointer-to-size tracking hash table to determine the exact
 * allocation size, avoiding the inaccurate hipMemGetInfo delta method.
 */
FUNC_ATTR_VISIBLE
hipError_t hipFree(void *ptr) {
    REAL_FUNC(hipFree);

    if (ptr == NULL)
        return real_hipFree(ptr);

    if (hip_get_device_memory_limit(current_device) == 0)
        return real_hipFree(ptr);

    int dev = current_device;
    int alloc_dev = dev;

    pthread_mutex_lock(&g_tracker_mutex);
    size_t tracked_size = alloc_tracker_remove(get_tracker(), ptr, &alloc_dev);
    pthread_mutex_unlock(&g_tracker_mutex);

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

/*
 * hipMallocManaged - Unified memory allocation.
 */
FUNC_ATTR_VISIBLE
hipError_t hipMallocManaged(void **ptr, size_t size, unsigned int flags) {
    typedef hipError_t (*fn_t)(void **, size_t, unsigned int);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)amvgpu_real_dlsym(RTLD_NEXT, "hipMallocManaged");
    if (!real_fn)
        return hipErrorNotInitialized;

    if (hip_get_device_memory_limit(current_device) == 0)
        return real_fn(ptr, size, flags);

    int dev = current_device;

    if (hip_shrreg_lock() != 0)
        return real_fn(ptr, size, flags);

    if (hip_oom_check(dev, size) != 0) {
        hip_shrreg_unlock();
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    hipError_t ret = real_fn(ptr, size, flags);

    if (ret == hipSuccess) {
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        pthread_mutex_lock(&g_tracker_mutex);
        alloc_tracker_insert(get_tracker(), *ptr, size, dev);
        pthread_mutex_unlock(&g_tracker_mutex);
        LOG_DEBUG("hipMallocManaged: %zu bytes -> %p (dev %d)", size, *ptr, dev);
    }

    return ret;
}

/*
 * hipMallocAsync - Asynchronous device memory allocation.
 */
FUNC_ATTR_VISIBLE
hipError_t hipMallocAsync(void **ptr, size_t size, hipStream_t stream) {
    typedef hipError_t (*fn_t)(void **, size_t, hipStream_t);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)amvgpu_real_dlsym(RTLD_NEXT, "hipMallocAsync");
    if (!real_fn)
        return hipErrorNotInitialized;

    if (hip_get_device_memory_limit(current_device) == 0)
        return real_fn(ptr, size, stream);

    int dev = current_device;

    if (hip_shrreg_lock() != 0)
        return real_fn(ptr, size, stream);

    if (hip_oom_check(dev, size) != 0) {
        hip_shrreg_unlock();
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    hipError_t ret = real_fn(ptr, size, stream);

    if (ret == hipSuccess) {
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        pthread_mutex_lock(&g_tracker_mutex);
        alloc_tracker_insert(get_tracker(), *ptr, size, dev);
        pthread_mutex_unlock(&g_tracker_mutex);
        LOG_DEBUG("hipMallocAsync: %zu bytes (dev %d)", size, dev);
    }

    return ret;
}

/*
 * hipFreeAsync - Asynchronous device memory deallocation.
 */
FUNC_ATTR_VISIBLE
hipError_t hipFreeAsync(void *ptr, hipStream_t stream) {
    typedef hipError_t (*fn_t)(void *, hipStream_t);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)amvgpu_real_dlsym(RTLD_NEXT, "hipFreeAsync");
    if (!real_fn)
        return hipErrorNotInitialized;

    if (ptr == NULL)
        return real_fn(ptr, stream);

    if (hip_get_device_memory_limit(current_device) == 0)
        return real_fn(ptr, stream);

    int alloc_dev = current_device;
    pthread_mutex_lock(&g_tracker_mutex);
    size_t tracked_size = alloc_tracker_remove(get_tracker(), ptr, &alloc_dev);
    pthread_mutex_unlock(&g_tracker_mutex);

    hipError_t ret = real_fn(ptr, stream);

    if (ret == hipSuccess && tracked_size > 0) {
        if (hip_shrreg_lock() == 0) {
            hip_rm_device_memory_usage(alloc_dev, tracked_size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        LOG_DEBUG("hipFreeAsync: %p freed %zu bytes (dev %d, tracked)", ptr, tracked_size, alloc_dev);
    }

    return ret;
}

/*
 * hipHostMalloc - Pinned host memory allocation.
 * Tracked because it pins GPU-accessible memory.
 */
FUNC_ATTR_VISIBLE
hipError_t hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
    typedef hipError_t (*fn_t)(void **, size_t, unsigned int);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)amvgpu_real_dlsym(RTLD_NEXT, "hipHostMalloc");
    if (!real_fn)
        return hipErrorNotInitialized;

    /* Pinned host memory does not consume device VRAM,
     * pass through without limit check */
    return real_fn(ptr, size, flags);
}

/*
 * hipHostFree - Pinned host memory deallocation.
 */
FUNC_ATTR_VISIBLE
hipError_t hipHostFree(void *ptr) {
    typedef hipError_t (*fn_t)(void *);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)amvgpu_real_dlsym(RTLD_NEXT, "hipHostFree");
    if (!real_fn)
        return hipErrorNotInitialized;

    return real_fn(ptr);
}

/*
 * hipMallocPitch - Pitched device memory allocation.
 */
FUNC_ATTR_VISIBLE
hipError_t hipMallocPitch(void **ptr, size_t *pitch,
                          size_t width, size_t height) {
    typedef hipError_t (*fn_t)(void **, size_t *, size_t, size_t);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)amvgpu_real_dlsym(RTLD_NEXT, "hipMallocPitch");
    if (!real_fn)
        return hipErrorNotInitialized;

    if (hip_get_device_memory_limit(current_device) == 0)
        return real_fn(ptr, pitch, width, height);

    /* Estimate allocation size (actual pitch may be larger) */
    size_t estimated = width * height;
    int dev = current_device;

    if (hip_shrreg_lock() != 0)
        return real_fn(ptr, pitch, width, height);

    if (hip_oom_check(dev, estimated) != 0) {
        hip_shrreg_unlock();
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    hipError_t ret = real_fn(ptr, pitch, width, height);

    if (ret == hipSuccess && pitch != NULL) {
        size_t actual = (*pitch) * height;
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, actual, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        pthread_mutex_lock(&g_tracker_mutex);
        alloc_tracker_insert(get_tracker(), *ptr, actual, dev);
        pthread_mutex_unlock(&g_tracker_mutex);
        LOG_DEBUG("hipMallocPitch: %zux%zu (pitch=%zu, total=%zu) dev %d",
                  width, height, *pitch, actual, dev);
    }

    return ret;
}

/*
 * hipExtMallocWithFlags - Extended allocation with flags.
 */
FUNC_ATTR_VISIBLE
hipError_t hipExtMallocWithFlags(void **ptr, size_t size, unsigned int flags) {
    typedef hipError_t (*fn_t)(void **, size_t, unsigned int);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)amvgpu_real_dlsym(RTLD_NEXT, "hipExtMallocWithFlags");
    if (!real_fn)
        return hipErrorNotInitialized;

    if (hip_get_device_memory_limit(current_device) == 0)
        return real_fn(ptr, size, flags);

    int dev = current_device;

    if (hip_shrreg_lock() != 0)
        return real_fn(ptr, size, flags);

    if (hip_oom_check(dev, size) != 0) {
        hip_shrreg_unlock();
        return hipErrorOutOfMemory;
    }

    hip_shrreg_unlock();

    hipError_t ret = real_fn(ptr, size, flags);

    if (ret == hipSuccess) {
        if (hip_shrreg_lock() == 0) {
            hip_add_device_memory_usage(dev, size, MEM_TYPE_DATA);
            hip_shrreg_unlock();
        }
        pthread_mutex_lock(&g_tracker_mutex);
        alloc_tracker_insert(get_tracker(), *ptr, size, dev);
        pthread_mutex_unlock(&g_tracker_mutex);
    }

    return ret;
}

/*
 * hipSetDevice - Track current device for subsequent allocations.
 */
FUNC_ATTR_VISIBLE
hipError_t hipSetDevice(int device) {
    typedef hipError_t (*fn_t)(int);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)amvgpu_real_dlsym(RTLD_NEXT, "hipSetDevice");
    if (!real_fn)
        return hipErrorNotInitialized;

    hipError_t ret = real_fn(device);
    if (ret == hipSuccess) {
        current_device = device;
        LOG_DEBUG("hipSetDevice: %d", device);
    }

    return ret;
}

/*
 * hipMemGetInfo - Override to report virtualized memory limits.
 * When a memory limit is configured, report the limit as total
 * and adjust free memory accordingly.
 */
FUNC_ATTR_VISIBLE
hipError_t hipMemGetInfo(size_t *free_mem, size_t *total_mem) {
    typedef hipError_t (*fn_t)(size_t *, size_t *);
    static fn_t real_fn = NULL;
    if (!real_fn)
        real_fn = (fn_t)amvgpu_real_dlsym(RTLD_NEXT, "hipMemGetInfo");
    if (!real_fn)
        return hipErrorNotInitialized;

    hipError_t ret = real_fn(free_mem, total_mem);

    if (ret == hipSuccess) {
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
