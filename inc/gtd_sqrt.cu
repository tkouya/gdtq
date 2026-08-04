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

	/* Newton doubles the number of correct digits at every step, so
	 * each step runs at the cheapest precision that can hold its
	 * result: ~16 -> ~32 digits in gdd_real, then the final step in
	 * gtd_real.  Both are pairs of branch-free FMAs, so neither
	 * h*r^2 nor the following multiply-and-add is renormalized on its
	 * own.  The 0.0.2 version ran two full triple-double iterations
	 * built from separate multiplies and adds. */
	gdd_real ad = make_dd(a.x, a.y);
	gdd_real hd = mul_pwr2(ad, 0.5);
	gdd_real rd = make_dd(1.0 / sqrt(a.x));
	rd = dw_fma(dw_fma(negative(hd), sqr(rd), make_dd(0.5)), rd, rd);

	gtd_real h = mul_pwr2(a, 0.5);
	gtd_real r = make_td(rd.x, rd.y, 0.0);

	r = tw_fma(tw_fma(negative(h), sqr(r), make_td(0.5)), r, r);

	r = r * a;
	return r;
}

/* Reference (0.0.2) square root, kept for benchmarking. */
__device__
gtd_real sqrt_legacy(const gtd_real &a)
{
	if (is_zero(a))
		return make_td(0.0);

	if (is_negative(a))
		return make_td(0.0);

	gtd_real r = make_td(1.0 / sqrt(a.x));
	gtd_real h = mul_pwr2(a, 0.5);

	r = r + ((0.5 - h * sqr(r)) * r);
	r = r + ((0.5 - h * sqr(r)) * r);

	r = r * a;
	return r;
}

#endif /* __GTD_SQRT_CU__ */
