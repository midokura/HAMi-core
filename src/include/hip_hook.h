/*
 * Copyright 2024 HAMi Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * HIP function entry table declaration for dlsym interception.
 */

#ifndef HIP_HOOK_H
#define HIP_HOOK_H

#include "libamvgpu.h"

/* Number of HIP functions in the entry table */
#define HIP_ENTRY_COUNT 32

/* Initialize the HIP library entry table with real function pointers */
int hip_hook_init(void);

/* Lookup a function in the entry table */
void *hip_hook_find(const char *name);

#endif /* HIP_HOOK_H */
