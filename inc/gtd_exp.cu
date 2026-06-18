/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GTD_EXP_CU__
#define __GTD_EXP_CU__

#include "gqd.cuh"

/* Taylor series for exp(r), |r| < 1/k, with argument reduction
 *   exp(a) = 2^m * (exp(r))^k,    m = round(a/log2),  r = (a - m*log2)/k
 * k = 2^16 keeps |r| < ~6e-6, then 16 self-squarings undo the scaling.
 * Truncation point ~ 8 inverse-factorial terms achieves TD precision. */
__device__
gtd_real exp(const gtd_real &a)
{
	gtd_real r;
	const double k = ldexp(1.0, 16);
	const double inv_k = 1.0 / k;

	if (a.x <= -709.0) { r.x = r.y = r.z = 0.0; return r; }
	if (a.x >=  709.0) { r.x = r.y = r.z = 0.0; return r; }

	if (is_zero(a)) { r.x = 1.0; r.y = r.z = 0.0; return r; }

	if (is_one(a))  return _td_e;

	double m = floor(a.x / _td_log2.x + 0.5);
	r = mul_pwr2(a - _td_log2 * m, inv_k);

	gtd_real s, p, t;
	double thresh = inv_k * _td_eps;

	p = sqr(r);
	s = r + mul_pwr2(p, 0.5);
	int i = 0;
	do {
		p = p * r;
		t = p * td_inv_fact[i++];
		s = s + t;
	} while ((fabs(to_double(t)) > thresh) && (i < n_td_inv_fact));

	for (int j = 0; j < 16; j++)
		s = mul_pwr2(s, 2.0) + sqr(s);

	s = s + 1.0;

	return ldexp(s, int(m));
}

#endif /* __GTD_EXP_CU__ */
