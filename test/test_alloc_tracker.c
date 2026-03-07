/*
 * Unit tests for alloc_tracker.h - pointer-to-size tracking hash table.
 *
 * Build and run:
 *   gcc -o test_alloc_tracker test_alloc_tracker.c -I../src/hip && ./test_alloc_tracker
 */

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "../src/hip/alloc_tracker.h"

static alloc_tracker_t tracker;

static void test_init(void) {
    alloc_tracker_init(&tracker);
    assert(tracker.count == 0);
    printf("  PASS: init\n");
}

static void test_insert_and_lookup(void) {
    alloc_tracker_init(&tracker);

    void *p1 = (void *)0x1000;
    void *p2 = (void *)0x2000;
    void *p3 = (void *)0x3000;

    assert(alloc_tracker_insert(&tracker, p1, 100, 0) == 0);
    assert(alloc_tracker_insert(&tracker, p2, 200, 0) == 0);
    assert(alloc_tracker_insert(&tracker, p3, 300, 1) == 0);
    assert(tracker.count == 3);

    assert(alloc_tracker_lookup(&tracker, p1) == 100);
    assert(alloc_tracker_lookup(&tracker, p2) == 200);
    assert(alloc_tracker_lookup(&tracker, p3) == 300);
    assert(alloc_tracker_lookup(&tracker, (void *)0x9999) == 0);
    printf("  PASS: insert_and_lookup\n");
}

static void test_remove(void) {
    alloc_tracker_init(&tracker);

    void *p1 = (void *)0x1000;
    void *p2 = (void *)0x2000;

    alloc_tracker_insert(&tracker, p1, 100, 0);
    alloc_tracker_insert(&tracker, p2, 200, 1);

    int dev = -1;
    size_t size = alloc_tracker_remove(&tracker, p1, &dev);
    assert(size == 100);
    assert(dev == 0);
    assert(tracker.count == 1);
    assert(alloc_tracker_lookup(&tracker, p1) == 0); /* Gone */
    assert(alloc_tracker_lookup(&tracker, p2) == 200); /* Still there */

    size = alloc_tracker_remove(&tracker, p2, &dev);
    assert(size == 200);
    assert(dev == 1);
    assert(tracker.count == 0);

    /* Remove non-existent */
    size = alloc_tracker_remove(&tracker, (void *)0x9999, NULL);
    assert(size == 0);
    printf("  PASS: remove\n");
}

static void test_null_ptr(void) {
    alloc_tracker_init(&tracker);

    assert(alloc_tracker_insert(&tracker, NULL, 100, 0) == -1);
    assert(alloc_tracker_lookup(&tracker, NULL) == 0);
    assert(alloc_tracker_remove(&tracker, NULL, NULL) == 0);
    printf("  PASS: null_ptr\n");
}

static void test_duplicate_insert(void) {
    alloc_tracker_init(&tracker);

    void *p = (void *)0x5000;
    alloc_tracker_insert(&tracker, p, 100, 0);
    assert(tracker.count == 1);

    /* Insert same pointer with different size - should update */
    alloc_tracker_insert(&tracker, p, 200, 1);
    assert(tracker.count == 1);
    assert(alloc_tracker_lookup(&tracker, p) == 200);
    printf("  PASS: duplicate_insert\n");
}

static void test_alloc_free_cycle(void) {
    alloc_tracker_init(&tracker);

    /* Simulate alloc-free-alloc cycle */
    void *p = (void *)0x6000;
    alloc_tracker_insert(&tracker, p, 100, 0);
    alloc_tracker_remove(&tracker, p, NULL);
    assert(tracker.count == 0);

    /* Re-insert same pointer (simulates reuse of address) */
    alloc_tracker_insert(&tracker, p, 200, 0);
    assert(tracker.count == 1);
    assert(alloc_tracker_lookup(&tracker, p) == 200);
    printf("  PASS: alloc_free_cycle\n");
}

static void test_many_entries(void) {
    alloc_tracker_init(&tracker);

    /* Insert many entries */
    int n = 1000;
    for (int i = 0; i < n; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x1000);
        assert(alloc_tracker_insert(&tracker, p, (size_t)(i * 64), 0) == 0);
    }
    assert(tracker.count == (size_t)n);

    /* Verify all present */
    for (int i = 0; i < n; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x1000);
        assert(alloc_tracker_lookup(&tracker, p) == (size_t)(i * 64));
    }

    /* Remove all */
    for (int i = 0; i < n; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x1000);
        size_t s = alloc_tracker_remove(&tracker, p, NULL);
        assert(s == (size_t)(i * 64));
    }
    assert(tracker.count == 0);
    printf("  PASS: many_entries (%d)\n", n);
}

static void test_device_tracking(void) {
    alloc_tracker_init(&tracker);

    void *p0 = (void *)0xA000;
    void *p1 = (void *)0xB000;

    alloc_tracker_insert(&tracker, p0, 1024, 0);
    alloc_tracker_insert(&tracker, p1, 2048, 3);

    int dev = -1;
    alloc_tracker_remove(&tracker, p0, &dev);
    assert(dev == 0);

    alloc_tracker_remove(&tracker, p1, &dev);
    assert(dev == 3);
    printf("  PASS: device_tracking\n");
}

static void test_hash_collision(void) {
    alloc_tracker_init(&tracker);

    /* Insert many pointers that may collide (close addresses) */
    int n = 200;
    for (int i = 0; i < n; i++) {
        /* Pointers spaced 16 bytes apart - after >>4 in hash, these are
         * consecutive integers, likely to cause collisions */
        void *p = (void *)(uintptr_t)(0x10000 + i * 16);
        assert(alloc_tracker_insert(&tracker, p, (size_t)(i + 1) * 100, 0) == 0);
    }
    assert(tracker.count == (size_t)n);

    /* Verify all lookups work despite collisions */
    for (int i = 0; i < n; i++) {
        void *p = (void *)(uintptr_t)(0x10000 + i * 16);
        assert(alloc_tracker_lookup(&tracker, p) == (size_t)(i + 1) * 100);
    }

    /* Remove every other entry */
    for (int i = 0; i < n; i += 2) {
        void *p = (void *)(uintptr_t)(0x10000 + i * 16);
        size_t s = alloc_tracker_remove(&tracker, p, NULL);
        assert(s == (size_t)(i + 1) * 100);
    }
    assert(tracker.count == (size_t)(n / 2));

    /* Remaining entries (odd indices) must still be findable
     * through tombstone chains */
    for (int i = 1; i < n; i += 2) {
        void *p = (void *)(uintptr_t)(0x10000 + i * 16);
        assert(alloc_tracker_lookup(&tracker, p) == (size_t)(i + 1) * 100);
    }

    /* Removed entries must not be found */
    for (int i = 0; i < n; i += 2) {
        void *p = (void *)(uintptr_t)(0x10000 + i * 16);
        assert(alloc_tracker_lookup(&tracker, p) == 0);
    }

    printf("  PASS: hash_collision (%d entries)\n", n);
}

static void test_tombstone_reinsert(void) {
    alloc_tracker_init(&tracker);

    void *p1 = (void *)0xD000;
    void *p2 = (void *)0xE000;
    void *p3 = (void *)0xF000;

    alloc_tracker_insert(&tracker, p1, 100, 0);
    alloc_tracker_insert(&tracker, p2, 200, 0);
    alloc_tracker_insert(&tracker, p3, 300, 0);

    /* Remove middle entry - creates tombstone */
    alloc_tracker_remove(&tracker, p2, NULL);
    assert(tracker.count == 2);

    /* Insert new pointer - should reuse tombstone slot */
    void *p4 = (void *)0x1F000;
    alloc_tracker_insert(&tracker, p4, 400, 1);
    assert(tracker.count == 3);

    /* All entries accessible */
    assert(alloc_tracker_lookup(&tracker, p1) == 100);
    assert(alloc_tracker_lookup(&tracker, p3) == 300);
    assert(alloc_tracker_lookup(&tracker, p4) == 400);
    assert(alloc_tracker_lookup(&tracker, p2) == 0); /* Removed */

    /* Re-insert same pointer that was removed */
    alloc_tracker_insert(&tracker, p2, 999, 2);
    assert(tracker.count == 4);
    assert(alloc_tracker_lookup(&tracker, p2) == 999);

    printf("  PASS: tombstone_reinsert\n");
}

static void test_capacity_limit(void) {
    alloc_tracker_init(&tracker);

    /* Fill to capacity */
    for (int i = 0; i < ALLOC_TRACKER_CAPACITY; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x100);
        int rc = alloc_tracker_insert(&tracker, p, 64, 0);
        assert(rc == 0);
    }
    assert(tracker.count == ALLOC_TRACKER_CAPACITY);

    /* Next insert should fail */
    void *overflow = (void *)0xFFFFFF00;
    assert(alloc_tracker_insert(&tracker, overflow, 64, 0) == -1);

    /* Remove one entry */
    void *first = (void *)0x100;
    size_t s = alloc_tracker_remove(&tracker, first, NULL);
    assert(s == 64);
    assert(tracker.count == ALLOC_TRACKER_CAPACITY - 1);

    /* Now insert should succeed (tombstone reuse) */
    assert(alloc_tracker_insert(&tracker, overflow, 128, 0) == 0);
    assert(tracker.count == ALLOC_TRACKER_CAPACITY);
    assert(alloc_tracker_lookup(&tracker, overflow) == 128);

    printf("  PASS: capacity_limit (%d entries)\n", ALLOC_TRACKER_CAPACITY);
}

static void test_interleaved_alloc_free(void) {
    alloc_tracker_init(&tracker);

    /* Simulate PyTorch-like pattern:
     * allocate a batch, free some, allocate more, free all */
    int total_ops = 0;

    /* Phase 1: allocate 500 tensors */
    for (int i = 0; i < 500; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x1000);
        alloc_tracker_insert(&tracker, p, (size_t)(i + 1) * 1024, 0);
    }
    assert(tracker.count == 500);
    total_ops += 500;

    /* Phase 2: free first 200 (forward pass intermediates) */
    for (int i = 0; i < 200; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x1000);
        size_t s = alloc_tracker_remove(&tracker, p, NULL);
        assert(s == (size_t)(i + 1) * 1024);
    }
    assert(tracker.count == 300);
    total_ops += 200;

    /* Phase 3: allocate 300 more (backward pass) with recycled addresses.
     * Addresses 1-200 were freed in phase 2, so they are new inserts (+200).
     * Addresses 201-300 still exist, so they are duplicate updates (+0).
     * count = 300 + 200 = 500 */
    for (int i = 0; i < 300; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x1000); /* reuse low addrs */
        alloc_tracker_insert(&tracker, p, (size_t)(i + 1) * 2048, 0);
    }
    assert(tracker.count == 500);
    total_ops += 300;

    /* Phase 4: verify all current allocations */
    /* Addresses 1-300: should have new sizes (2048 * i) */
    for (int i = 0; i < 300; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x1000);
        assert(alloc_tracker_lookup(&tracker, p) == (size_t)(i + 1) * 2048);
    }
    /* Addresses 301-500: should have original sizes (1024 * i) */
    for (int i = 300; i < 500; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x1000);
        assert(alloc_tracker_lookup(&tracker, p) == (size_t)(i + 1) * 1024);
    }

    /* Phase 5: free everything.
     * After phase 3, we have entries at addresses 1-500
     * (1-300 updated, 301-500 original). Freeing 1-500 → count = 0 */
    for (int i = 0; i < 500; i++) {
        void *p = (void *)(uintptr_t)((i + 1) * 0x1000);
        alloc_tracker_remove(&tracker, p, NULL);
    }
    assert(tracker.count == 0);
    total_ops += 500;

    printf("  PASS: interleaved_alloc_free (%d ops)\n", total_ops);
}

int main(void) {
    printf("Running alloc_tracker tests...\n");
    test_init();
    test_insert_and_lookup();
    test_remove();
    test_null_ptr();
    test_duplicate_insert();
    test_alloc_free_cycle();
    test_many_entries();
    test_device_tracking();
    test_hash_collision();
    test_tombstone_reinsert();
    test_capacity_limit();
    test_interleaved_alloc_free();
    printf("All alloc_tracker tests passed!\n");
    return 0;
}
