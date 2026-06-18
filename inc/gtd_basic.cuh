/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 * Triple-double layer (gtd_real) modelled on CAMPARY's TD operators
 * (Joldes-Muller-Popescu, "Tight and rigorous error bounds for basic
 * building blocks of double-word arithmetic"). */

#ifndef __GTD_BASIC_CUH__
#define __GTD_BASIC_CUH__

#include "gqd.cuh"

/* renormalization */
__host__ __device__
void renorm( gtd_real &x );

__host__ __device__
gtd_real make_td_renorm(double x0, double x1, double x2, double x3);

/* additions */
__host__ __device__
gtd_real negative(const gtd_real &a);

__host__ __device__
gtd_real operator+(const gtd_real &a, double b);

__host__ __device__
gtd_real operator+(double a, const gtd_real &b);

__host__ __device__
gtd_real standard_add(const gtd_real &a, const gtd_real &b);

__host__ __device__
gtd_real bf_add(const gtd_real &a, const gtd_real &b);

__host__ __device__
gtd_real operator+(const gtd_real &a, const gtd_real &b);

/* subtractions */
__host__ __device__
gtd_real operator-(const gtd_real &a, double b);

__host__ __device__
gtd_real operator-(double a, const gtd_real &b);

__host__ __device__
gtd_real operator-(const gtd_real &a, const gtd_real &b);

/* multiplications */
__host__ __device__
gtd_real mul_pwr2(const gtd_real &a, double b);

__device__
gtd_real operator*(const gtd_real &a, double b);

__device__
gtd_real operator*(double a, const gtd_real &b);

__device__
gtd_real standard_mul(const gtd_real &a, const gtd_real &b);

__device__
gtd_real bf_mul(const gtd_real &a, const gtd_real &b);

__device__
gtd_real operator*(const gtd_real &a, const gtd_real &b);

__device__
gtd_real sqr(const gtd_real &a);

/* divisions */
__device__
gtd_real operator/(const gtd_real &a, const gtd_real &b);

__device__
gtd_real operator/(const gtd_real &a, double b);

__device__
gtd_real operator/(double a, const gtd_real &b);

__device__
gtd_real inv(const gtd_real &a);

/* miscellaneous */
__host__ __device__
gtd_real abs(const gtd_real &a);

__host__ __device__
gtd_real fabs(const gtd_real &a);

__host__ __device__
double to_double(const gtd_real &a);

__host__ __device__
gtd_real ldexp(const gtd_real &a, int n);

__device__
gtd_real nint(const gtd_real &a);

/* comparisons */
__host__ __device__ bool is_zero    (const gtd_real &a);
__host__ __device__ bool is_one     (const gtd_real &a);
__host__ __device__ bool is_positive(const gtd_real &a);
__host__ __device__ bool is_negative(const gtd_real &a);

__host__ __device__ bool operator==(const gtd_real &a, const gtd_real &b);
__host__ __device__ bool operator==(const gtd_real &a, double b);
__host__ __device__ bool operator==(double a, const gtd_real &b);

__host__ __device__ bool operator!=(const gtd_real &a, const gtd_real &b);
__host__ __device__ bool operator!=(const gtd_real &a, double b);
__host__ __device__ bool operator!=(double a, const gtd_real &b);

__host__ __device__ bool operator< (const gtd_real &a, const gtd_real &b);
__host__ __device__ bool operator< (const gtd_real &a, double b);
__host__ __device__ bool operator< (double a, const gtd_real &b);

__host__ __device__ bool operator<=(const gtd_real &a, const gtd_real &b);
__host__ __device__ bool operator<=(const gtd_real &a, double b);
__host__ __device__ bool operator<=(double a, const gtd_real &b);

__host__ __device__ bool operator> (const gtd_real &a, const gtd_real &b);
__host__ __device__ bool operator> (const gtd_real &a, double b);
__host__ __device__ bool operator> (double a, const gtd_real &b);

__host__ __device__ bool operator>=(const gtd_real &a, const gtd_real &b);
__host__ __device__ bool operator>=(const gtd_real &a, double b);
__host__ __device__ bool operator>=(double a, const gtd_real &b);

#endif /* __GTD_BASIC_CUH__ */
