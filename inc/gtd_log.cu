/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GTD_LOG_CU__
#define __GTD_LOG_CU__

#include <math_constants.h>

#include "gqd.cuh"

#if defined(GDTQ_LEGACY_ELEMENTARY)  /* 0.0.3 implementation; default: mw kernels in gqd_elem.cu */

/* log(a) by Newton's method on f(x) = exp(x) - a:
 *   x_{n+1} = x_n + a * exp(-x_n) - 1
 * Newton doubles the number of correct digits per iteration, so start
 * from the ~32-digit double-double logarithm (one gdd_real exp) instead
 * of the 16-digit double one: a single triple-double iteration -- one
 * gtd_real exp instead of two -- then reaches full precision. */
__device__
gtd_real log(const gtd_real &a)
{
	if (is_one(a))    return make_td(0.0);

	/* MPFR/IEEE semantics, silently: log(0) = -inf, log(negative) = nan. */
	if (is_zero(a))   return make_td(-CUDART_INF);
	if (a.x < 0.0)    return make_td(CUDART_NAN);

	gdd_real x0 = log(make_dd(a.x, a.y));
	gtd_real x = make_td(x0.x, x0.y, 0.0);

	x = x + a * exp(negative(x)) - 1.0;

	return x;
}

/* Base-10 logarithm. */
__device__
gtd_real log10(const gtd_real &a)
{
	return log(a) / _td_log10;
}

#endif /* GDTQ_LEGACY_ELEMENTARY */

#endif /* __GTD_LOG_CU__ */
