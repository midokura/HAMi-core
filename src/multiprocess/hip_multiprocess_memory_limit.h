/*
 * Copyright 2024 HAMi Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Shared memory region for multi-process GPU memory tracking (AMD/HIP).
 * Follows the same IPC pattern as HAMi's multiprocess_memory_limit.h.
 */

#ifndef HIP_MULTIPROCESS_MEMORY_LIMIT_H
#define HIP_MULTIPROCESS_MEMORY_LIMIT_H

#include <semaphore.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Shared region magic number for version detection */
#define HIP_SHRREG_MAGIC 0x484D4931  /* "HMI1" */

/* Maximum number of concurrent processes sharing GPUs */
#define HIP_MAX_PROCS 1024

/* Maximum GPU devices */
#define HIP_MAX_DEVICES 16

/* Memory category types */
#define MEM_TYPE_CONTEXT  0
#define MEM_TYPE_MODULE   1
#define MEM_TYPE_DATA     2

/* Per-device memory tracking for a single process */
typedef struct {
    uint64_t context_size;
    uint64_t module_size;
    uint64_t data_size;
    uint64_t total;
} hip_device_memory_t;

/* Process slot in shared region */
typedef struct {
    int32_t pid;
    int32_t hostpid;
    hip_device_memory_t used[HIP_MAX_DEVICES];
    uint64_t monitorused[HIP_MAX_DEVICES];
    int32_t status;  /* 0=inactive, 1=running */
} hip_shrreg_proc_slot_t;

/* Shared memory region layout */
typedef struct {
    int32_t initialized_flag;              /* HIP_SHRREG_MAGIC when valid */
    sem_t sem;                             /* Synchronization semaphore */
    size_t owner_pid;                      /* Current lock holder */
    uint64_t device_num;                   /* Number of GPU devices */
    uint64_t limit[HIP_MAX_DEVICES];       /* Memory limit per device (bytes) */
    hip_shrreg_proc_slot_t procs[HIP_MAX_PROCS];
    int proc_num;                          /* Active process count */
} hip_shared_region_t;

/* Semaphore lock timeout in seconds */
#define SEM_LOCK_TIMEOUT_SEC 10

/* Maximum lock retry count */
#define SEM_LOCK_MAX_RETRIES 30

/*
 * Initialize the shared memory region.
 * Creates or opens the shared cache file and mmaps it.
 * Returns 0 on success, -1 on failure.
 */
int hip_shrreg_init(void);

/*
 * Acquire exclusive access to the shared region.
 * Uses sem_timedwait with deadlock detection.
 * Returns 0 on success, -1 on failure.
 */
int hip_shrreg_lock(void);

/*
 * Release exclusive access to the shared region.
 */
void hip_shrreg_unlock(void);

/*
 * Get the memory limit for a specific device.
 * Returns the limit in bytes, or 0 if unlimited.
 */
uint64_t hip_get_device_memory_limit(int dev);

/*
 * Get current aggregate memory usage for a device across all processes.
 * Must be called with the lock held.
 */
uint64_t hip_get_device_memory_usage(int dev);

/*
 * Add memory usage for the current process on a specific device.
 * type: MEM_TYPE_CONTEXT, MEM_TYPE_MODULE, or MEM_TYPE_DATA
 * Returns 0 on success, -1 if OOM.
 */
int hip_add_device_memory_usage(int dev, uint64_t size, int type);

/*
 * Remove memory usage for the current process on a specific device.
 */
void hip_rm_device_memory_usage(int dev, uint64_t size, int type);

/*
 * Check if allocating additional bytes would exceed the device memory limit.
 * Must be called with the lock held.
 * Returns 0 if allocation is allowed, -1 if it would exceed the limit.
 */
int hip_oom_check(int dev, uint64_t addon);

/*
 * Clean up process slot on exit.
 * Registered via atexit().
 */
void hip_shrreg_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* HIP_MULTIPROCESS_MEMORY_LIMIT_H */
