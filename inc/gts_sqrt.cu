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

	/* Each Newton step runs at the cheapest precision that can hold
	 * its result: ~7 -> ~14 digits in gds_real, then the final step in
	 * gts_real.  Both are pairs of branch-free FMAs, so neither
	 * h*r^2 nor the following multiply-and-add is renormalized on its
	 * own.  The 0.0.2 version ran two full triple-single iterations
	 * built from separate multiplies and adds. */
	gds_real ad = make_ds(a.x, a.y);
	gds_real hd = mul_pwr2(ad, 0.5f);
	gds_real rd = make_ds(1.0f / sqrt(a.x));
	rd = dw_fma(dw_fma(negative(hd), sqr(rd), make_ds(0.5f)), rd, rd);

	gts_real h = mul_pwr2(a, 0.5f);
	gts_real r = make_ts(rd.x, rd.y, 0.0f);

	r = tw_fma(tw_fma(negative(h), sqr(r), make_ts(0.5f)), r, r);

	r = r * a;
	return r;
}

/* Reference (0.0.2) square root, kept for benchmarking. */
__device__
gts_real sqrt_legacy(const gts_real &a)
{
	if (is_zero(a))
		return make_ts(0.0);

	if (is_negative(a))
		return make_ts(0.0);

	gts_real r = make_ts(1.0 / sqrt(a.x));
	gts_real h = mul_pwr2(a, 0.5);

	r = r + ((0.5 - h * sqr(r)) * r);
	r = r + ((0.5 - h * sqr(r)) * r);

	r = r * a;
	return r;
}

#endif /* __GTS_SQRT_CU__ */
