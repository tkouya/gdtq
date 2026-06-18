/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GTD_SQRT_CU__
#define __GTD_SQRT_CU__

#include "gqd.cuh"

/* sqrt for triple-double via Newton iteration on x = 1/sqrt(a):
 *   x_{n+1} = x_n + (1/2 - a/2 * x_n^2) * x_n
 * One double seed gives ~53 bits; two iterations bring us to ~159 bits
 * (TD precision).  A final multiply by `a` gives sqrt(a). */
__device__
gtd_real sqrt(const gtd_real &a)
{
	if (is_zero(a))
		return make_td(0.0);

	if (is_negative(a))
		return make_td(0.0);   /* TODO: signal NaN */

	gtd_real r = make_td(1.0 / sqrt(a.x));
	gtd_real h = mul_pwr2(a, 0.5);

	/* two refinement passes are enough for ~159 bits */
	r = r + ((0.5 - h * sqr(r)) * r);
	r = r + ((0.5 - h * sqr(r)) * r);

	r = r * a;
	return r;
}

#endif /* __GTD_SQRT_CU__ */
