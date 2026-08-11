/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GTS_EXP_CU__
#define __GTS_EXP_CU__

#include <math_constants.h>

#include "gqs.cuh"

/* Taylor series for exp(r), |r| < 1/k, with argument reduction
 *   exp(a) = 2^m * (exp(r))^k,    m = round(a/log2),  r = (a - m*log2)/k
 * k = 2^16 keeps |r| < ~6e-6, then 16 self-squarings undo the scaling.
 * Each term is accumulated with one fused multiply-add (tw_fma); the
 * convergence test uses the leading-word product, so the term itself
 * need not be materialized.  The overflow/underflow bounds are those
 * of IEEE-754 single precision (the leading limb is a float). */
__device__
gts_real exp(const gts_real &a)
{
	const float k = ldexp(1.0, 16);
	const float inv_k = 1.0 / k;

	if (a.x <= -103.98f) return make_ts(0.0);
	if (a.x >=  88.73f)  return make_ts(CUDART_INF_F);

	if (is_zero(a)) return make_ts(1.0);

	if (is_one(a))  return _ts_e;

	float m = floor(a.x / _ts_log2.x + 0.5);
	gts_real r = mul_pwr2(a - _ts_log2 * m, inv_k);

	gts_real s, p;
	float thresh = inv_k * _ts_eps;
	float tmag;

	p = sqr(r);
	s = r + mul_pwr2(p, 0.5);
	int i = 0;
	do {
		p = p * r;
		tmag = fabs(to_float(p) * ts_inv_fact[i].x);
		s = tw_fma(p, ts_inv_fact[i], s);
		++i;
	} while (tmag > thresh && i < n_ts_inv_fact);

	for (int j = 0; j < 16; j++)
		s = mul_pwr2(s, 2.0) + sqr(s);

	s = s + 1.0;

	return ldexp(s, int(m));
}


/* exp(a) - 1, accurate for small |a|.
   For |a| <= log(2)/2 the ln2 reduction in exp() is a no-op (m = 0), and
   the repeated-squaring ladder  s <- 2s + s^2  is exactly the doubling
   rule for u = exp(t) - 1.  So expm1 is exp() without the final +1.
   For larger |a|, exp(a) - 1 suffers no cancellation.                  */
__device__
gts_real expm1(const gts_real &a)
{
	const float k = ldexp(1.0, 16);
	const float inv_k = 1.0 / k;

	if (a.x <= -88.0f)  return make_ts(-1.0);
	if (a.x >=  88.73f) return make_ts(CUDART_INF_F);

	if (is_zero(a)) return make_ts(0.0);

	if (fabs(a.x) > 0.346573590279972655f)   /* log(2)/2 */
		return exp(a) - 1.0;

	gts_real r = mul_pwr2(a, inv_k);
	gts_real s, p;
	float thresh = inv_k * _ts_eps;
	float tmag;

	p = sqr(r);
	s = r + mul_pwr2(p, 0.5);
	int i = 0;
	do {
		p = p * r;
		tmag = fabs(to_float(p) * ts_inv_fact[i].x);
		s = tw_fma(p, ts_inv_fact[i], s);
		++i;
	} while (tmag > thresh && i < 9);

	for (int j = 0; j < 16; j++)
		s = mul_pwr2(s, 2.0) + sqr(s);

	return s;      /* = exp(a) - 1, no +1 */
}

#endif /* __GTS_EXP_CU__ */
