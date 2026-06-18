/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Float-overload error-free transformations.  These mirror the double
 * versions in inline.cuh but operate on `float`.  They coexist with the
 * double overloads via C++ overload resolution, so kernel code that
 * uses gds_real / gts_real / gqs_real picks them up automatically.
 */
#ifndef __GDS_GQS_INLINE_CUH__
#define __GDS_GQS_INLINE_CUH__

extern __host__ __device__
float quick_two_sum( float a, float b, float &err );

extern __host__ __device__
float two_sum( float a, float b, float &err );

extern __host__ __device__
float quick_two_diff( float a, float b, float &err );

extern __host__ __device__
float two_diff( float a, float b, float &err );

extern __host__ __device__
void split(float a, float &hi, float &lo);

extern __device__
float two_prod(float a, float b, float &err);

extern __host__ __device__
float two_sqr(float a, float &err);

extern __host__ __device__
float nint(float d);

extern __host__ __device__
void three_sum(float &a, float &b, float &c);

extern __host__ __device__
void three_sum2(float &a, float &b, float &c);

extern __host__ __device__
void renormalize3(float x0, float x1, float x2, float x3,
                  float &r0, float &r1, float &r2);

#endif /* __GDS_GQS_INLINE_CUH__ */
