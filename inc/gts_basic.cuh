/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 * Triple-float layer (gts_real) modelled on CAMPARY's TD operators
 * (Joldes-Muller-Popescu, "Tight and rigorous error bounds for basic
 * building blocks of float-word arithmetic"). */

#ifndef __GTS_BASIC_CUH__
#define __GTS_BASIC_CUH__

#include "gqs.cuh"

/* renormalization */
__host__ __device__
void renorm( gts_real &x );

__host__ __device__
gts_real make_ts_renorm(float x0, float x1, float x2, float x3);

/* additions */
__host__ __device__
gts_real negative(const gts_real &a);

__host__ __device__
gts_real operator+(const gts_real &a, float b);

__host__ __device__
gts_real operator+(float a, const gts_real &b);

__host__ __device__
gts_real standard_add(const gts_real &a, const gts_real &b);

__host__ __device__
gts_real bf_add(const gts_real &a, const gts_real &b);

__host__ __device__
gts_real operator+(const gts_real &a, const gts_real &b);

/* subtractions */
__host__ __device__
gts_real operator-(const gts_real &a, float b);

__host__ __device__
gts_real operator-(float a, const gts_real &b);

__host__ __device__
gts_real operator-(const gts_real &a, const gts_real &b);

/* multiplications */
__host__ __device__
gts_real mul_pwr2(const gts_real &a, float b);

__device__
gts_real operator*(const gts_real &a, float b);

__device__
gts_real operator*(float a, const gts_real &b);

__device__
gts_real standard_mul(const gts_real &a, const gts_real &b);

__device__
gts_real bf_mul(const gts_real &a, const gts_real &b);

__device__
gts_real operator*(const gts_real &a, const gts_real &b);

__device__
gts_real sqr(const gts_real &a);

/* divisions */
__device__
gts_real operator/(const gts_real &a, const gts_real &b);

__device__
gts_real operator/(const gts_real &a, float b);

__device__
gts_real operator/(float a, const gts_real &b);

__device__
gts_real inv(const gts_real &a);

/* miscellaneous */
__host__ __device__
gts_real abs(const gts_real &a);

__host__ __device__
gts_real fabs(const gts_real &a);

__host__ __device__
float to_float(const gts_real &a);

__host__ __device__
gts_real ldexp(const gts_real &a, int n);

__device__
gts_real nint(const gts_real &a);

/* comparisons */
__host__ __device__ bool is_zero    (const gts_real &a);
__host__ __device__ bool is_one     (const gts_real &a);
__host__ __device__ bool is_positive(const gts_real &a);
__host__ __device__ bool is_negative(const gts_real &a);

__host__ __device__ bool operator==(const gts_real &a, const gts_real &b);
__host__ __device__ bool operator==(const gts_real &a, float b);
__host__ __device__ bool operator==(float a, const gts_real &b);

__host__ __device__ bool operator!=(const gts_real &a, const gts_real &b);
__host__ __device__ bool operator!=(const gts_real &a, float b);
__host__ __device__ bool operator!=(float a, const gts_real &b);

__host__ __device__ bool operator< (const gts_real &a, const gts_real &b);
__host__ __device__ bool operator< (const gts_real &a, float b);
__host__ __device__ bool operator< (float a, const gts_real &b);

__host__ __device__ bool operator<=(const gts_real &a, const gts_real &b);
__host__ __device__ bool operator<=(const gts_real &a, float b);
__host__ __device__ bool operator<=(float a, const gts_real &b);

__host__ __device__ bool operator> (const gts_real &a, const gts_real &b);
__host__ __device__ bool operator> (const gts_real &a, float b);
__host__ __device__ bool operator> (float a, const gts_real &b);

__host__ __device__ bool operator>=(const gts_real &a, const gts_real &b);
__host__ __device__ bool operator>=(const gts_real &a, float b);
__host__ __device__ bool operator>=(float a, const gts_real &b);

#endif /* __GTS_BASIC_CUH__ */
