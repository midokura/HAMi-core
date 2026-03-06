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
 * Multi-process memory limiting for AMD GPUs via shared memory IPC.
 *
 * This module manages a shared memory region (mmap'd file) that tracks
 * GPU memory usage across all processes sharing the same GPU device.
 * Each process registers itself in a slot and updates its memory usage
 * atomically. When a process attempts to allocate GPU memory that would
 * exceed the configured limit, the allocation is rejected with OOM.
 *
 * The design follows HAMi's multiprocess_memory_limit.c for NVIDIA GPUs,
 * adapted for AMD HIP runtime.
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "../include/glibc_compat.h"
#include "hip_multiprocess_memory_limit.h"
#include "../include/hip_log_utils.h"
#include "../include/libamvgpu.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

/* Global shared region pointer */
static hip_shared_region_t *g_shrreg = NULL;

/* Current process slot index (-1 = not registered) */
static int g_proc_slot = -1;

/* Initialization guard */
static pthread_once_t g_init_once = PTHREAD_ONCE_INIT;

/* Flag to track if cleanup was already called */
static volatile int g_cleanup_done = 0;

/*
 * Parse a memory size string with optional suffix.
 * Supports: G/g (GiB), M/m (MiB), K/k (KiB), or plain bytes.
 * Returns size in bytes, or 0 on parse error.
 */
static uint64_t parse_memory_size(const char *str) {
    char *end = NULL;
    uint64_t val;

    if (str == NULL || *str == '\0')
        return 0;

    val = strtoull(str, &end, 10);
    if (end == str)
        return 0;

    switch (*end) {
    case 'G': case 'g':
        val *= 1024ULL * 1024 * 1024;
        break;
    case 'M': case 'm':
        val *= 1024ULL * 1024;
        break;
    case 'K': case 'k':
        val *= 1024ULL;
        break;
    case '\0': case '\n':
        /* Already in bytes */
        break;
    default:
        LOG_WARN("Unknown memory size suffix '%c' in '%s'", *end, str);
        break;
    }

    return val;
}

/*
 * Read per-device memory limits from environment variables.
 * Format: HIP_DEVICE_MEMORY_LIMIT_{idx}=<size>[G|M|K]
 * Falls back to HIP_DEVICE_MEMORY_LIMIT for global limit.
 */
static void read_memory_limits(hip_shared_region_t *region) {
    char env_name[64];
    const char *env_val;
    uint64_t global_limit = 0;

    /* Global limit as fallback */
    env_val = getenv(HIP_DEVICE_MEMORY_LIMIT_ENV);
    if (env_val != NULL) {
        global_limit = parse_memory_size(env_val);
        LOG_INFO("Global memory limit: %lu bytes (%s)", global_limit, env_val);
    }

    for (int i = 0; i < HIP_MAX_DEVICES; i++) {
        snprintf(env_name, sizeof(env_name),
                 HIP_DEVICE_MEMORY_LIMIT_ENV_FMT, i);
        env_val = getenv(env_name);
        if (env_val != NULL) {
            region->limit[i] = parse_memory_size(env_val);
            LOG_INFO("Device %d memory limit: %lu bytes (%s)",
                     i, region->limit[i], env_val);
        } else {
            region->limit[i] = global_limit;
        }
    }
}

/*
 * Check if a process is still alive by reading /proc/<pid>/stat.
 * Returns 1 if alive, 0 if dead.
 */
static int is_process_alive(pid_t pid) {
    char path[32];
    struct stat st;

    if (pid <= 0)
        return 0;

    snprintf(path, sizeof(path), "/proc/%d/stat", pid);
    return (stat(path, &st) == 0) ? 1 : 0;
}

/*
 * Find or allocate a process slot for the current process.
 * Must be called with the lock held.
 * Returns slot index, or -1 if no slots available.
 */
static int find_or_create_proc_slot(hip_shared_region_t *region) {
    pid_t mypid = getpid();
    int free_slot = -1;

    for (int i = 0; i < HIP_MAX_PROCS; i++) {
        if (region->procs[i].pid == mypid) {
            return i;
        }
        if (free_slot < 0 && region->procs[i].pid == 0) {
            free_slot = i;
        }
    }

    if (free_slot < 0) {
        /* Try to reclaim dead process slots */
        for (int i = 0; i < HIP_MAX_PROCS; i++) {
            if (region->procs[i].pid != 0 &&
                !is_process_alive(region->procs[i].pid)) {
                LOG_INFO("Reclaiming dead process slot %d (pid %d)",
                         i, region->procs[i].pid);
                memset(&region->procs[i], 0, sizeof(hip_shrreg_proc_slot_t));
                region->proc_num--;
                if (free_slot < 0) {
                    free_slot = i;
                }
            }
        }
    }

    if (free_slot < 0) {
        LOG_ERROR("No available process slots (max %d)", HIP_MAX_PROCS);
        return -1;
    }

    memset(&region->procs[free_slot], 0, sizeof(hip_shrreg_proc_slot_t));
    region->procs[free_slot].pid = mypid;
    region->procs[free_slot].hostpid = mypid;
    region->procs[free_slot].status = 1;  /* Running */
    region->proc_num++;

    LOG_INFO("Registered process %d in slot %d (total: %d)",
             mypid, free_slot, region->proc_num);

    return free_slot;
}

/*
 * Clear the current process's slot in the shared region.
 * Must be called with the lock held.
 */
static void clear_proc_slot_locked(hip_shared_region_t *region, int slot) {
    if (slot < 0 || slot >= HIP_MAX_PROCS)
        return;

    pid_t pid = region->procs[slot].pid;
    memset(&region->procs[slot], 0, sizeof(hip_shrreg_proc_slot_t));
    if (region->proc_num > 0)
        region->proc_num--;

    LOG_INFO("Cleared process slot %d (pid %d, remaining: %d)",
             slot, pid, region->proc_num);
}

/*
 * Initialize shared memory region from cache file.
 */
static void do_shrreg_init(void) {
    const char *cache_path;
    int fd;
    int created = 0;
    struct stat st;

    cache_path = getenv(HIP_DEVICE_MEMORY_SHARED_CACHE);
    if (cache_path == NULL)
        cache_path = DEFAULT_SHARED_CACHE_PATH;

    LOG_INFO("Shared cache: %s", cache_path);

    fd = open(cache_path, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        LOG_ERROR("Failed to open shared cache '%s': %s",
                  cache_path, strerror(errno));
        return;
    }

    /* Ensure file is large enough */
    if (fstat(fd, &st) == 0 && st.st_size < (off_t)sizeof(hip_shared_region_t)) {
        if (ftruncate(fd, sizeof(hip_shared_region_t)) < 0) {
            LOG_ERROR("Failed to resize shared cache: %s", strerror(errno));
            close(fd);
            return;
        }
        created = 1;
    }

    g_shrreg = (hip_shared_region_t *)mmap(
        NULL, sizeof(hip_shared_region_t),
        PROT_READ | PROT_WRITE, MAP_SHARED,
        fd, 0);
    close(fd);

    if (g_shrreg == MAP_FAILED) {
        LOG_ERROR("Failed to mmap shared cache: %s", strerror(errno));
        g_shrreg = NULL;
        return;
    }

    /* Initialize on first creation */
    if (created || g_shrreg->initialized_flag != HIP_SHRREG_MAGIC) {
        memset(g_shrreg, 0, sizeof(hip_shared_region_t));

        if (sem_init(&g_shrreg->sem, 1 /* pshared */, 1 /* initial */) < 0) {
            LOG_ERROR("Failed to init semaphore: %s", strerror(errno));
            munmap(g_shrreg, sizeof(hip_shared_region_t));
            g_shrreg = NULL;
            return;
        }

        read_memory_limits(g_shrreg);

        __sync_synchronize();
        g_shrreg->initialized_flag = HIP_SHRREG_MAGIC;

        LOG_INFO("Created new shared region");
    } else {
        LOG_INFO("Attached to existing shared region (procs: %d)",
                 g_shrreg->proc_num);
    }

    /* Register current process */
    if (hip_shrreg_lock() == 0) {
        g_proc_slot = find_or_create_proc_slot(g_shrreg);
        hip_shrreg_unlock();
    }

    /* Register cleanup on process exit */
    atexit(hip_shrreg_cleanup);

    LOG_INFO("Shared region initialized, slot=%d", g_proc_slot);
}

/*
 * Reinitialize after fork in child process.
 */
static void child_reinit(void) {
    g_init_once = PTHREAD_ONCE_INIT;
    g_proc_slot = -1;
    g_cleanup_done = 0;

    if (g_shrreg != NULL && hip_shrreg_lock() == 0) {
        g_proc_slot = find_or_create_proc_slot(g_shrreg);
        hip_shrreg_unlock();
    }
}

/* Register fork handler during library load */
__attribute__((constructor))
static void register_fork_handler(void) {
    pthread_atfork(NULL, NULL, child_reinit);
}

int hip_shrreg_init(void) {
    pthread_once(&g_init_once, do_shrreg_init);
    return (g_shrreg != NULL) ? 0 : -1;
}

int hip_shrreg_lock(void) {
    struct timespec ts;
    int retries = 0;

    if (g_shrreg == NULL)
        return -1;

    while (retries < SEM_LOCK_MAX_RETRIES) {
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += SEM_LOCK_TIMEOUT_SEC;

        if (sem_timedwait(&g_shrreg->sem, &ts) == 0) {
            __sync_synchronize();
            g_shrreg->owner_pid = (size_t)getpid();
            return 0;
        }

        if (errno != ETIMEDOUT) {
            LOG_ERROR("sem_timedwait failed: %s", strerror(errno));
            return -1;
        }

        /* Deadlock detection: check if lock owner is still alive */
        size_t owner = g_shrreg->owner_pid;
        if (owner > 0 && !is_process_alive((pid_t)owner)) {
            LOG_WARN("Lock owner pid %zu is dead, recovering semaphore",
                     owner);
            sem_post(&g_shrreg->sem);
            /* Retry immediately */
            continue;
        }

        retries++;
        LOG_WARN("Lock timeout (attempt %d/%d, owner: %zu)",
                 retries, SEM_LOCK_MAX_RETRIES, owner);
    }

    LOG_ERROR("Failed to acquire lock after %d retries", SEM_LOCK_MAX_RETRIES);
    return -1;
}

void hip_shrreg_unlock(void) {
    if (g_shrreg == NULL)
        return;

    g_shrreg->owner_pid = 0;
    __sync_synchronize();
    sem_post(&g_shrreg->sem);
}

uint64_t hip_get_device_memory_limit(int dev) {
    if (g_shrreg == NULL || dev < 0 || dev >= HIP_MAX_DEVICES)
        return 0;
    return g_shrreg->limit[dev];
}

uint64_t hip_get_device_memory_usage(int dev) {
    uint64_t total = 0;

    if (g_shrreg == NULL || dev < 0 || dev >= HIP_MAX_DEVICES)
        return 0;

    for (int i = 0; i < HIP_MAX_PROCS; i++) {
        if (g_shrreg->procs[i].pid > 0 &&
            is_process_alive(g_shrreg->procs[i].pid)) {
            total += g_shrreg->procs[i].used[dev].total;
        }
    }

    return total;
}

int hip_oom_check(int dev, uint64_t addon) {
    uint64_t limit, usage;

    limit = hip_get_device_memory_limit(dev);
    if (limit == 0)
        return 0;  /* No limit configured */

    usage = hip_get_device_memory_usage(dev) + addon;

    if (usage > limit) {
        LOG_ERROR("Device %d OOM: usage %lu + alloc %lu = %lu > limit %lu",
                  dev, usage - addon, addon, usage, limit);

        /* Optionally kill dead processes to reclaim memory */
        const char *oom_killer = getenv(ACTIVE_OOM_KILLER_ENV);
        if (oom_killer != NULL && strcmp(oom_killer, "true") == 0) {
            for (int i = 0; i < HIP_MAX_PROCS; i++) {
                if (g_shrreg->procs[i].pid > 0 &&
                    !is_process_alive(g_shrreg->procs[i].pid)) {
                    LOG_INFO("OOM: reclaiming dead slot %d (pid %d)",
                             i, g_shrreg->procs[i].pid);
                    clear_proc_slot_locked(g_shrreg, i);
                }
            }
            /* Re-check after cleanup */
            usage = hip_get_device_memory_usage(dev) + addon;
            if (usage <= limit) {
                LOG_INFO("OOM resolved after reclaiming dead processes");
                return 0;
            }
        }

        return -1;
    }

    return 0;
}

int hip_add_device_memory_usage(int dev, uint64_t size, int type) {
    if (g_shrreg == NULL || g_proc_slot < 0)
        return -1;
    if (dev < 0 || dev >= HIP_MAX_DEVICES)
        return -1;

    hip_shrreg_proc_slot_t *slot = &g_shrreg->procs[g_proc_slot];
    hip_device_memory_t *mem = &slot->used[dev];

    switch (type) {
    case MEM_TYPE_CONTEXT:
        mem->context_size += size;
        break;
    case MEM_TYPE_MODULE:
        mem->module_size += size;
        break;
    case MEM_TYPE_DATA:
        mem->data_size += size;
        break;
    default:
        LOG_WARN("Unknown memory type %d", type);
        mem->data_size += size;
        break;
    }

    mem->total = mem->context_size + mem->module_size + mem->data_size;
    __sync_synchronize();

    LOG_DEBUG("Device %d: added %lu bytes (type %d), total: %lu",
              dev, size, type, mem->total);

    return 0;
}

void hip_rm_device_memory_usage(int dev, uint64_t size, int type) {
    if (g_shrreg == NULL || g_proc_slot < 0)
        return;
    if (dev < 0 || dev >= HIP_MAX_DEVICES)
        return;

    hip_shrreg_proc_slot_t *slot = &g_shrreg->procs[g_proc_slot];
    hip_device_memory_t *mem = &slot->used[dev];

    switch (type) {
    case MEM_TYPE_CONTEXT:
        mem->context_size = (mem->context_size >= size) ?
                            mem->context_size - size : 0;
        break;
    case MEM_TYPE_MODULE:
        mem->module_size = (mem->module_size >= size) ?
                           mem->module_size - size : 0;
        break;
    case MEM_TYPE_DATA:
        mem->data_size = (mem->data_size >= size) ?
                         mem->data_size - size : 0;
        break;
    default:
        mem->data_size = (mem->data_size >= size) ?
                         mem->data_size - size : 0;
        break;
    }

    mem->total = mem->context_size + mem->module_size + mem->data_size;
    __sync_synchronize();

    LOG_DEBUG("Device %d: removed %lu bytes (type %d), total: %lu",
              dev, size, type, mem->total);
}

void hip_shrreg_cleanup(void) {
    struct timespec ts;

    if (g_cleanup_done || g_shrreg == NULL || g_proc_slot < 0)
        return;

    g_cleanup_done = 1;

    /* Try to acquire lock with a short timeout for cleanup */
    clock_gettime(CLOCK_REALTIME, &ts);
    ts.tv_sec += 3;

    if (sem_timedwait(&g_shrreg->sem, &ts) == 0) {
        clear_proc_slot_locked(g_shrreg, g_proc_slot);
        g_shrreg->owner_pid = 0;
        __sync_synchronize();
        sem_post(&g_shrreg->sem);
    } else {
        LOG_WARN("Cleanup: could not acquire lock, slot may leak");
    }

    g_proc_slot = -1;
}
