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
 * AMD GPU (HIP) memory virtualization library for HAMi.
 * Intercepts HIP memory allocation calls via LD_PRELOAD and enforces
 * per-device memory limits using shared memory IPC.
 */

#ifndef LIBAMVGPU_H
#define LIBAMVGPU_H

#include <dlfcn.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* HIP error codes (subset) */
typedef enum {
    hipSuccess = 0,
    hipErrorOutOfMemory = 2,
    hipErrorNotInitialized = 3,
    hipErrorInvalidValue = 11,
    hipErrorInvalidDevice = 101,
} hipError_t;

typedef void *hipDeviceptr_t;
typedef void *hipStream_t;
typedef void *hipMemPool_t;

/* Maximum number of GPU devices */
#define HIP_DEVICE_MAX_COUNT 16

/* Environment variable names - matching HAMi convention */
#define HIP_DEVICE_MEMORY_LIMIT_ENV     "HIP_DEVICE_MEMORY_LIMIT"
#define HIP_DEVICE_MEMORY_LIMIT_ENV_FMT "HIP_DEVICE_MEMORY_LIMIT_%d"
#define HIP_DEVICE_MEMORY_SHARED_CACHE  "HIP_DEVICE_MEMORY_SHARED_CACHE"
#define LIBHIP_LOG_LEVEL_ENV            "LIBHIP_LOG_LEVEL"
#define ACTIVE_OOM_KILLER_ENV           "ACTIVE_OOM_KILLER"

/* Default shared cache path */
#define DEFAULT_SHARED_CACHE_PATH       "/tmp/hipdevshr.cache"

/* Function entry table types */
typedef hipError_t (*hip_sym_t)();

typedef struct {
    void *fn_ptr;
    const char *name;
} hip_entry_t;

/* GCC visibility for LD_PRELOAD exported symbols */
#define FUNC_ATTR_VISIBLE __attribute__((visibility("default")))

/* dlsym hook macro - find real function via dlsym */
#define REAL_FUNC(name) \
    static __typeof__(name) *real_##name = NULL; \
    if (!real_##name) { \
        real_##name = (__typeof__(name) *)dlsym(RTLD_NEXT, #name); \
        if (!real_##name) { \
            LOG_ERROR("Failed to resolve real " #name); \
            return hipErrorNotInitialized; \
        } \
    }

/* HIP call forwarding with entry table */
#define HIP_FIND_ENTRY(table, sym) ({ \
    void *_entry = NULL; \
    for (int _i = 0; table[_i].name != NULL; _i++) { \
        if (strcmp(table[_i].name, #sym) == 0) { \
            _entry = table[_i].fn_ptr; \
            break; \
        } \
    } \
    _entry; \
})

#define HIP_OVERRIDE_CALL(table, sym, ...) \
    ({ hip_sym_t _fn = (hip_sym_t)HIP_FIND_ENTRY(table, sym); \
       _fn ? _fn(__VA_ARGS__) : hipErrorNotInitialized; })

/* Global entry tables */
extern hip_entry_t hip_library_entry[];

#ifdef __cplusplus
}
#endif

#endif /* LIBAMVGPU_H */
