/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */
#ifndef __GDD_TYPE_H__
#define __GDD_TYPE_H__

/*
 * CUDA 13.0 deprecates legacy vector types (double4, long4, ulong4,
 * longlong4, ulonglong4) in favor of the *_16a / *_32a aligned variants.
 * Since this library uses double4 as the underlying type of gqd_real,
 * we globally silence the deprecation diagnostic. The legacy types are
 * scheduled for removal in CUDA 14.0; migrating to double4_16a will be
 * required at that point.
 *
 * This macro must be defined BEFORE <vector_types.h> is included.
 */
#ifndef __NV_NO_VECTOR_DEPRECATION_DIAG
#define __NV_NO_VECTOR_DEPRECATION_DIAG
#endif

#include <vector_types.h>


/* compiler switch */
/**
 * ALL_MATH will include advanced math functions, including
 * atan, acos, asin, sinh, cosh, tanh, asinh, acosh, atanh
 * WARNING: these functions take long time to compile,
 * e.g., several hours
 *
 * gdtq 0.0.4: only for builds with GDTQ_LEGACY_ELEMENTARY (the 0.0.3
 * implementations); by default all functions of all classes come from the
 * multi-word kernels (gqd_elem.cu / gqs_elem.cu) and ALL_MATH is ignored.
 * */
//#define ALL_MATH


/* type definition */
/* double-precision based (original GQD) */
typedef double2 gdd_real; // DD
typedef double3 gtd_real; // TD
typedef double4 gqd_real; // QD

/* float-precision based (added 2026, mirrors dtq-0.0.2 ds/ts/qs_real) */
/* Full precision holds only while the lowest limb (~|x|*2^-24 per word)
 * stays in the float normal range (>= 2^-126); below the per-type
 * threshold, precision degrades by about one bit per halving
 * (representation limit of the format itself, independent of the
 * algorithm):  |x| >= 2^-102 (gds) / 2^-78 (gts) / 2^-54 (gqs). */
typedef float2  gds_real; // DS = double-single (2 floats, ~14 digits)
typedef float3  gts_real; // TS = triple-single (3 floats, ~21 digits)
typedef float4  gqs_real; // QS = quadruple-single (4 floats, ~28 digits)


/* initialization functions, these can be called by hosts */
void GDDStart(const int device = 0);
void GDDEnd();
void GTDStart(const int device = 0);
void GTDEnd();
void GQDStart(const int device = 0);
void GQDEnd();

/* float-based initialization functions */
void GDSStart(const int device = 0);
void GDSEnd();
void GTSStart(const int device = 0);
void GTSEnd();
void GQSStart(const int device = 0);
void GQSEnd();

#endif /*__GDD_GQD_TYPE_H__*/
