/*
 * Copyright 2024 HAMi Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Logging utilities for AMD GPU memory virtualization library.
 * Follows the same pattern as HAMi's log_utils.h for NVIDIA.
 */

#ifndef HIP_LOG_UTILS_H
#define HIP_LOG_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/syscall.h>

/* Log levels - matching HAMi convention */
#define LOG_LEVEL_ERROR 1
#define LOG_LEVEL_WARN  2
#define LOG_LEVEL_INFO  3
#define LOG_LEVEL_DEBUG 4

static inline int get_log_level(void) {
    static int level = -1;
    if (level < 0) {
        const char *env = getenv("LIBHIP_LOG_LEVEL");
        level = env ? atoi(env) : LOG_LEVEL_WARN;
    }
    return level;
}

#define LOG_FMT "[HAMI-core-hip %s(pid:%d tid:%ld %s:%d)]: "

#define LOG_IMPL(level_str, level_val, fmt, ...) \
    do { \
        if (get_log_level() >= (level_val)) { \
            fprintf(stderr, LOG_FMT fmt "\n", \
                    level_str, getpid(), syscall(SYS_gettid), \
                    __FILE__, __LINE__, ##__VA_ARGS__); \
        } \
    } while (0)

#define LOG_ERROR(fmt, ...) LOG_IMPL("Error", LOG_LEVEL_ERROR, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  LOG_IMPL("Warn",  LOG_LEVEL_WARN,  fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  LOG_IMPL("Info",  LOG_LEVEL_INFO,  fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) LOG_IMPL("Debug", LOG_LEVEL_DEBUG, fmt, ##__VA_ARGS__)

#endif /* HIP_LOG_UTILS_H */
