/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GTS_LOG_CU__
#define __GTS_LOG_CU__

#include <math_constants.h>

#include "gqs.cuh"

/* log(a) by Newton's method on f(x) = exp(x) - a:
 *   x_{n+1} = x_n + a * exp(-x_n) - 1
 * Newton doubles the number of correct digits per iteration, so start
 * from the ~14-digit float-float logarithm (one gds_real exp) instead
 * of the 7-digit float one: a single triple-float iteration -- one
 * gts_real exp instead of two -- then reaches full precision. */
__device__
gts_real log(const gts_real &a)
{
	if (is_one(a))    return make_ts(0.0);

	/* MPFR/IEEE semantics, silently: log(0) = -inf, log(negative) = nan. */
	if (is_zero(a))   return make_ts(-CUDART_INF_F);
	if (a.x < 0.0f)   return make_ts(CUDART_NAN_F);

	gds_real x0 = log(make_ds(a.x, a.y));
	gts_real x = make_ts(x0.x, x0.y, 0.0f);

	x = x + a * exp(negative(x)) - 1.0;

	return x;
}

/* Base-10 logarithm. */
__device__
gts_real log10(const gts_real &a)
{
	return log(a) / _ts_log10;
}

#endif /* __GTS_LOG_CU__ */
