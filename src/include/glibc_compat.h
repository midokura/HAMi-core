/*
 * glibc compatibility - ensure libamvgpu.so works on glibc 2.35+.
 *
 * _GNU_SOURCE implies _ISOC2X_SOURCE on glibc 2.38+, which causes
 * strtoull/strtol to redirect to __isoc23_* variants. This creates
 * a hard GLIBC_2.38 dependency that breaks in containers with older
 * glibc (e.g., Ubuntu 22.04 rocm/pytorch images use glibc 2.35).
 *
 * Fix: Include <features.h> to process _GNU_SOURCE, then override
 * the ISOC2X flag before any stdlib headers use it for __asm__ renames.
 *
 * IMPORTANT: This header MUST be included BEFORE <stdlib.h>, <stdio.h>,
 * or any other header that uses strtoull/strtol/atoi.
 */

#ifndef GLIBC_COMPAT_H
#define GLIBC_COMPAT_H

/* Force features.h to process _GNU_SOURCE now */
#include <features.h>

/* Override: disable C2X strtol redirection */
#ifdef __GLIBC_USE_C2X_STRTOL
#undef __GLIBC_USE_C2X_STRTOL
#define __GLIBC_USE_C2X_STRTOL 0
#endif

#endif /* GLIBC_COMPAT_H */
