/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GDS_SQRT_CU__
#define __GDS_SQRT_CU__

//#include "common.cuh"
//#include "gdd_sqrt.cuh"
#include "gqs.cuh"

/* Computes the square root of the float-float number dd.
   NOTE: dd must be a non-negative number.                   */
__device__
gds_real sqrt(const gds_real &a)
{
	if (is_zero(a))
    		return make_ds(0.0);

  	//TODO: should make an error
  	if (is_negative(a)) {
    		//return _nan;
         	 return make_ds( 0.0 );
  	}

  	float x = 1.0 / sqrt(a.x);
  	float ax = a.x * x;

  	return ds_add(ax, (a - sqr(ax)).x * (x * 0.5));
  	//return a - sqr(ax);
}

#endif /* __GDS_SQRT_CU__ */


