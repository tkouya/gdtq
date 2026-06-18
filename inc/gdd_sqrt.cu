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

  	/* sqr_d (not sqr) so this works whether or not QD's inline.h is
  	 * also in scope — see gdd_basic.cuh for the rationale. */
  	return dd_add(ax, (a - sqr_d(ax)).x * (x * 0.5));
  	//return a - sqr_d(ax);
}

#endif /* __GDD_SQRT_CU__ */


