/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GDD_SQRT_CU__
#define __GDD_SQRT_CU__

//#include "common.cuh"
//#include "gdd_sqrt.cuh"
#include "gqd.cuh"

/* Computes the square root of the double-double number dd.
   NOTE: dd must be a non-negative number.                   */
__device__
gdd_real sqrt(const gdd_real &a)
{
	if (is_zero(a))
    		return make_dd(0.0);

  	//TODO: should make an error
  	if (is_negative(a)) {
    		//return _nan;
         	 return make_dd( 0.0 );
  	}

  	double x = 1.0 / sqrt(a.x);
  	double ax = a.x * x;

  	/* d = high word of (a - ax*ax).  two_sqr(ax) is exact, and
  	 * a.x - p is exact by Sterbenz (p is within a factor of two of
  	 * a.x), so no double-double subtraction is needed here.  This is
  	 * the same error-free product the branch-free FMA is built on. */
  	double e;
  	double p = two_sqr(ax, e);
  	double d = ((a.x - p) - e) + a.y;

  	return dd_add(ax, d * (x * 0.5));
}

/* Reference (0.0.2) square root, kept so that the benchmark can measure
   what the branch-free FMA buys us. */
__device__
gdd_real sqrt_legacy(const gdd_real &a)
{
	if (is_zero(a))
		return make_dd(0.0);

	if (is_negative(a))
		return make_dd(0.0);

	double x = 1.0 / sqrt(a.x);
	double ax = a.x * x;

	/* sqr_d (not sqr) so this works whether or not QD's inline.h is
	 * also in scope — see gdd_basic.cuh for the rationale. */
	return dd_add(ax, (a - sqr_d(ax)).x * (x * 0.5));
}

#endif /* __GDD_SQRT_CU__ */


