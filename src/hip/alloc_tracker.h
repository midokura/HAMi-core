/*
 * Copyright 2024 HAMi Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Pointer-to-size tracking hash table for GPU memory allocations.
 *
 * Tracks (pointer, size, device) tuples so that hipFree can determine
 * the exact allocation size without relying on hipMemGetInfo delta.
 *
 * Uses open addressing with linear probing and tombstone markers
 * for deletion to maintain probe chain integrity.
 *
 * Thread safety: All access must be externally synchronized (e.g.,
 * the caller holds a pthread_mutex).
 */

#ifndef ALLOC_TRACKER_H
#define ALLOC_TRACKER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Maximum number of tracked allocations.
 * Must be a power of 2 for fast modular hashing. */
#define ALLOC_TRACKER_CAPACITY 16384

/* Sentinel value marking a deleted (tombstone) slot.
 * Using 0x1 which is never a valid aligned pointer. */
#define ALLOC_TOMBSTONE ((void *)0x1)

typedef struct {
    void   *ptr;    /* NULL = empty, TOMBSTONE = deleted, else = live */
    size_t  size;   /* Allocation size in bytes */
    int     device; /* Device index */
} alloc_entry_t;

typedef struct {
    alloc_entry_t entries[ALLOC_TRACKER_CAPACITY];
    size_t count;
} alloc_tracker_t;

/* Initialize the tracker (zero all entries). */
static inline void alloc_tracker_init(alloc_tracker_t *t) {
    t->count = 0;
    for (size_t i = 0; i < ALLOC_TRACKER_CAPACITY; i++) {
        t->entries[i].ptr = NULL;
        t->entries[i].size = 0;
        t->entries[i].device = 0;
    }
}

/* Hash a pointer to a table index. */
static inline size_t alloc_hash(const void *ptr) {
    uintptr_t h = (uintptr_t)ptr >> 4;
    h *= 0x9E3779B97F4A7C15ULL;
    return h & (ALLOC_TRACKER_CAPACITY - 1);
}

/*
 * Record an allocation.
 * Returns 0 on success, -1 if the table is full.
 */
static inline int alloc_tracker_insert(alloc_tracker_t *t,
                                        void *ptr, size_t size, int device) {
    if (ptr == NULL || ptr == ALLOC_TOMBSTONE ||
        t->count >= ALLOC_TRACKER_CAPACITY)
        return -1;

    size_t idx = alloc_hash(ptr);
    size_t first_tombstone = (size_t)-1;

    for (size_t probe = 0; probe < ALLOC_TRACKER_CAPACITY; probe++) {
        size_t i = (idx + probe) & (ALLOC_TRACKER_CAPACITY - 1);

        if (t->entries[i].ptr == NULL) {
            /* Empty slot - use tombstone slot if we passed one */
            size_t slot = (first_tombstone != (size_t)-1) ? first_tombstone : i;
            t->entries[slot].ptr = ptr;
            t->entries[slot].size = size;
            t->entries[slot].device = device;
            t->count++;
            return 0;
        }
        if (t->entries[i].ptr == ALLOC_TOMBSTONE) {
            if (first_tombstone == (size_t)-1)
                first_tombstone = i;
            continue;
        }
        if (t->entries[i].ptr == ptr) {
            /* Duplicate - update in place */
            t->entries[i].size = size;
            t->entries[i].device = device;
            return 0;
        }
    }

    /* Table full but we have a tombstone */
    if (first_tombstone != (size_t)-1) {
        t->entries[first_tombstone].ptr = ptr;
        t->entries[first_tombstone].size = size;
        t->entries[first_tombstone].device = device;
        t->count++;
        return 0;
    }

    return -1;
}

/*
 * Look up and remove an allocation by pointer.
 * Returns the allocation size, or 0 if not found.
 * Also sets *out_device if non-NULL.
 */
static inline size_t alloc_tracker_remove(alloc_tracker_t *t,
                                           void *ptr, int *out_device) {
    if (ptr == NULL)
        return 0;

    size_t idx = alloc_hash(ptr);
    for (size_t probe = 0; probe < ALLOC_TRACKER_CAPACITY; probe++) {
        size_t i = (idx + probe) & (ALLOC_TRACKER_CAPACITY - 1);

        if (t->entries[i].ptr == ptr) {
            size_t size = t->entries[i].size;
            if (out_device)
                *out_device = t->entries[i].device;
            /* Replace with tombstone to preserve probe chains */
            t->entries[i].ptr = ALLOC_TOMBSTONE;
            t->entries[i].size = 0;
            t->count--;
            return size;
        }
        if (t->entries[i].ptr == NULL) {
            return 0; /* Empty slot = end of probe chain */
        }
        /* TOMBSTONE: continue probing */
    }
    return 0;
}

/*
 * Look up an allocation size without removing it.
 * Returns the allocation size, or 0 if not found.
 */
static inline size_t alloc_tracker_lookup(const alloc_tracker_t *t,
                                           const void *ptr) {
    if (ptr == NULL)
        return 0;

    size_t idx = alloc_hash(ptr);
    for (size_t probe = 0; probe < ALLOC_TRACKER_CAPACITY; probe++) {
        size_t i = (idx + probe) & (ALLOC_TRACKER_CAPACITY - 1);
        if (t->entries[i].ptr == ptr)
            return t->entries[i].size;
        if (t->entries[i].ptr == NULL)
            return 0; /* End of probe chain */
        /* TOMBSTONE: continue probing */
    }
    return 0;
}

#ifdef __cplusplus
}
#endif

#endif /* ALLOC_TRACKER_H */
