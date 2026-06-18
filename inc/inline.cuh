/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */
#ifndef __GDD_GQD_INLINE_CUH__
#define __GDD_GQD_INLINE_CUH__


//#define _GQD_SPLITTER            (134217729.0)                   // = 2^27 + 1
//#define _GQD_SPLIT_THRESH        (6.69692879491417e+299)         // = 2^996

// check CUDA_FMA
#ifdef CUDA_FMA
	#define GD_FMA(a, b, c) fma((a), (b), (c)) // a * b + c
	#define GD_FMS(a, b, c) GD_FMA((a), (b), -(c)) // a * b - c
	#define GQD_FMA(a, b, c) GD_FMA((a), (b), (c)) // a * b + c
	#define GQD_FMS(a, b, c) GQD_FMA((a), (b), -(c)) // a * b - c
#endif // CUDA_FMA

//extern "C" {

/****************Basic Funcitons *********************/

//computs fl( a + b ) and err( a + b ), assumes |a| > |b|
extern __host__ __device__
double quick_two_sum( double a , double b, double &err );

extern __host__ __device__
double two_sum( double a, double b, double &err );

//computes fl( a - b ) and err( a - b ), assumes |a| >= |b|
extern __host__ __device__
double quick_two_diff( double a, double b, double &err );

//computes fl( a - b ) and err( a - b )
extern __host__ __device__
double two_diff( double a, double b, double &err );

// Computes high word and lo word of a 
extern __host__ __device__
void split(double a, double &hi, double &lo);

/* Computes fl(a*b) and err(a*b). */
extern  __device__
double two_prod(double a, double b, double &err);

/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
extern __host__ __device__
double two_sqr(double a, double &err);

/* Computes the nearest integer to d. */
extern __host__ __device__
double nint(double d);

/* three_sum: a, b, c become the three components of the exact sum a+b+c
 * arranged in decreasing magnitude (no rounding error in the sum).
 * Used by QD multiplication/addition and by the triple-double layer. */
extern __host__ __device__
void three_sum(double &a, double &b, double &c);

/* three_sum2: like three_sum but the lowest component is collapsed
 * (b receives t2 + t3 directly), saving one two_sum.  Used when the
 * lowest term will be discarded after one more renormalization step. */
extern __host__ __device__
void three_sum2(double &a, double &b, double &c);

/* Triple-double renormalization (CAMPARY / Joldes-Muller-Popescu,
 * "Algorithm 4: renormalization for ulp-nonoverlapping numbers").
 * Takes four input doubles (x0 most significant) and produces three
 * doubles (r0, r1, r2) that form a renormalized triple-double. */
extern __host__ __device__
void renormalize3(double x0, double x1, double x2, double x3,
                  double &r0, double &r1, double &r2);

//} // extern "C"

#endif /* __GDD_GQD_INLINE_CUH__ */
