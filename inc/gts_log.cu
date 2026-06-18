/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GTS_LOG_CU__
#define __GTS_LOG_CU__

#include "gqs.cuh"

/* log(a) by Newton's method on f(x) = exp(x) - a:
 *   x_{n+1} = x_n + a * exp(-x_n) - 1
 * Quadratic convergence: 53 -> 106 -> 159 bits in two iterations. */
__device__
gts_real log(const gts_real &a)
{
	if (is_one(a))    return make_ts(0.0);
	if (a.x <= 0.0)   return make_ts(0.0);   /* TODO: signal NaN/inf */

	gts_real x = make_ts(log(a.x));

	x = x + a * exp(negative(x)) - 1.0;
	x = x + a * exp(negative(x)) - 1.0;

	return x;
}

#endif /* __GTS_LOG_CU__ */
