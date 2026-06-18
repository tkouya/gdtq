/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GTS_SQRT_CU__
#define __GTS_SQRT_CU__

#include "gqs.cuh"

/* sqrt for triple-float via Newton iteration on x = 1/sqrt(a):
 *   x_{n+1} = x_n + (1/2 - a/2 * x_n^2) * x_n
 * One float seed gives ~53 bits; two iterations bring us to ~159 bits
 * (TD precision).  A final multiply by `a` gives sqrt(a). */
__device__
gts_real sqrt(const gts_real &a)
{
	if (is_zero(a))
		return make_ts(0.0);

	if (is_negative(a))
		return make_ts(0.0);   /* TODO: signal NaN */

	gts_real r = make_ts(1.0 / sqrt(a.x));
	gts_real h = mul_pwr2(a, 0.5);

	/* two refinement passes are enough for ~159 bits */
	r = r + ((0.5 - h * sqr(r)) * r);
	r = r + ((0.5 - h * sqr(r)) * r);

	r = r * a;
	return r;
}

#endif /* __GTS_SQRT_CU__ */
