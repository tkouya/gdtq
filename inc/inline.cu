/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GDD_GQD_INLINE_CU__
#define __GDD_GQD_INLINE_CU__

//#include "inline.cuh"

#define _GQD_SPLITTER            (134217729.0)                   // = 2^27 + 1
#define _GQD_SPLIT_THRESH        (6.69692879491417e+299)         // = 2^996

//extern "C" {

/****************Basic Funcitons *********************/

//computs fl( a + b ) and err( a + b ), assumes |a| > |b|
__host__ __device__
double quick_two_sum( double a , double b, double &err )
{

	if(b == 0.0) {
		err = 0.0;
		return (a + b);
	}

	double s = a + b;
	err = b - (s - a);

	return s;
}

__host__ __device__
double two_sum( double a, double b, double &err )
{

	if( (a == 0.0) || (b == 0.0) ) {
		err = 0.0;
		return (a + b);
	}

	double s = a + b;
	double bb = s - a;
	err = (a - (s - bb)) + (b - bb);
	
	return s;
}


//computes fl( a - b ) and err( a - b ), assumes |a| >= |b|
__host__ __device__
double quick_two_diff( double a, double b, double &err )
{
	if(a == b) {
		err = 0.0;
		return 0.0;
	}

	double s;
        
	/*
	if(fabs((a-b)/a) < GPU_D_EPS) {
                s = 0.0;
                err = 0.0;
                return s;
        }
	*/

	s = a - b;
	err = (a - s) - b;
	return s;
}

//computes fl( a - b ) and err( a - b )
__host__ __device__
double two_diff( double a, double b, double &err )
{
	if(a == b) {
		err = 0.0;
		return 0.0;
	}

	double s = a - b;
	
	/*
	if(fabs((a-b)/a) < GPU_D_EPS) {
		s = 0.0;
		err = 0.0;
		return s;
	}
	*/	

	double bb = s - a;
	err = (a - (s - bb)) - (b + bb);
	return s;
}

// Computes high word and lo word of a 
__host__ __device__
void split(double a, double &hi, double &lo) 
{
	double temp;
	if (a > _GQD_SPLIT_THRESH || a < -_GQD_SPLIT_THRESH)
	{
		a *= 3.7252902984619140625e-09;  // 2^-28
		temp = _GQD_SPLITTER * a;
		hi = temp - (temp - a);
		lo = a - hi;
		hi *= 268435456.0;          // 2^28
		lo *= 268435456.0;          // 2^28
	} else 	{
		temp = _GQD_SPLITTER * a;
		hi = temp - (temp - a);
		lo = a - hi;
	}
}

/* Computes fl(a*b) and err(a*b). */

/*
 * CUDA-13 / Blackwell note on FMA contraction
 * -------------------------------------------
 * The classical Dekker two_prod algorithm in the #else branch below
 * computes the rounding error of a*b with
 *
 *     err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo);
 *
 * This relies on every *, +, and - being a *separate* IEEE-754
 * double-precision rounded operation.  nvcc's default (-fmad=true),
 * especially on modern architectures such as Blackwell sm_12x,
 * aggressively contracts `A*B + C` into an fma() instruction.  When
 * this happens the rounding error computed above is no longer the
 * true error of `a*b`, which silently destroys QD multiplication
 * (and everything built on it: qd_mul, qd_sqr, qd_sqrt, sin, cos,
 * tan, ...).  Empirically the QD precision collapses from ~1e-63 to
 * ~1e-16 on a DGX Spark / GB10.
 *
 * The correct fix is to compute the error term with a *single* FMA:
 *
 *     err = fma(a, b, -p)      ==      a*b - p (exactly)
 *
 * which is not only immune to further contraction but also faster
 * and more accurate than the split-based method.  Defining GQD_FMS
 * below activates this path in two_prod / two_sqr.  __fma_rn is the
 * CUDA device intrinsic for round-to-nearest FMA in double precision.
 */
#ifndef GQD_FMS
#  ifdef __CUDA_ARCH__
     /* Device path: use the CUDA round-to-nearest FMA intrinsic. */
#    define GQD_FMS(a, b, p) __fma_rn((a), (b), -(p))
#  else
     /* Host path: <cmath> / <math.h> provides the C99 `fma()` which has
        the same semantics (round-to-nearest double-precision FMA).
        Needed because two_sqr is declared __host__ __device__. */
#    include <cmath>
#    define GQD_FMS(a, b, p) std::fma((a), (b), -(p))
#  endif
#endif

 __device__
double two_prod(double a, double b, double &err) 
{
#ifdef GQD_FMS
	double p = a * b;

	/* err is passed by reference, not by pointer: no dereference needed. */
	err = GQD_FMS(a, b, p);

	return p;
#else // GQD_FMS
	
	double a_hi, a_lo, b_hi, b_lo;
	double p = a * b;
	split(a, a_hi, a_lo);
	split(b, b_hi, b_lo);
	
	//err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo); 
	err = (a_hi*b_hi) - p + (a_hi*b_lo) + (a_lo*b_hi) + (a_lo*b_lo); 

	return p;
#endif // GQD_FMS
}

/* Computes fl(a*a) and err(a*a).  Faster than the above method. */
__host__ __device__
double two_sqr(double a, double &err) 
{
#ifdef GQD_FMS
	double p = a * a;

	/* err is passed by reference, not by pointer: no dereference needed.
	   The macro name is GQD_FMS (originally mistyped as QD_FMS). */
	err = GQD_FMS(a, a, p);

	return p;
#else // GQD_FMS
	double hi, lo;
	double q = a * a;
	split(a, hi, lo);
	err = ((hi * hi - q) + 2.0 * hi * lo) + lo * lo;
	return q;
#endif // GQD_FMS
}

/* Computes the nearest integer to d. */
__host__ __device__
double nint(double d)
{
	if (d == floor(d))
		return d;
	return floor(d + 0.5);
}

/* three_sum / three_sum2: hoisted from gqd_basic.cu so the triple-double
 * layer (gtd_basic.cu) can share them without an ODR clash. */
__host__ __device__
void three_sum(double &a, double &b, double &c)
{
	double t1, t2, t3;
	t1 = two_sum(a, b, t2);
	a  = two_sum(c, t1, t3);
	b  = two_sum(t2, t3, c);
}

__host__ __device__
void three_sum2(double &a, double &b, double &c)
{
	double t1, t2, t3;
	t1 = two_sum(a, b, t2);
	a  = two_sum(c, t1, t3);
	b  = (t2 + t3);
}

/* CAMPARY-style triple-double renormalization
 * (Joldes-Muller-Popescu 2017, Algorithm 4 "renormalize_3").
 * Inputs are four ulp-nonoverlapping doubles in decreasing magnitude;
 * outputs are a normalized triple. */
__host__ __device__
void renormalize3(double x0, double x1, double x2, double x3,
                  double &r0, double &r1, double &r2)
{
	double s, e1, e2, e3;
	/* sweep error from least to most significant */
	s  = quick_two_sum(x2, x3, e3);
	s  = quick_two_sum(x1, s,  e2);
	r0 = quick_two_sum(x0, s,  e1);
	/* combine the error chain */
	s  = quick_two_sum(e2, e3, e3);
	r1 = quick_two_sum(e1, s,  e2);
	r2 = e2 + e3;
}

//} // extern "C"

#endif /* __GDD_GQD_INLINE_CU__ */
