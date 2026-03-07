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
    printf("All alloc_tracker tests passed!\n");
    return 0;
}
