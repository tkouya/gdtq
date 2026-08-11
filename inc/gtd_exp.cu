/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */

#ifndef __GTD_EXP_CU__
#define __GTD_EXP_CU__

#include <math_constants.h>

#include "gqd.cuh"

/* Same two-level table-driven reduction as the gqd_real version,

       a = m log 2 + j1/64 + j2/8192 + r,   |r| <= 2^-14 + eps,

   with the exp tables truncated to triple-double (see GTDStart()). */
__device__
gtd_real exp(const gtd_real &a)
{
	if (a.x <= -709.0) return make_td(0.0);
	if (a.x >=  709.0) return make_td(CUDART_INF);

	if (is_zero(a)) return make_td(1.0);

	if (is_one(a))  return _td_e;

	double m = floor(a.x / _td_log2.x + 0.5);
	gtd_real r = a - _td_log2 * m;

	int j1 = (int)(floor(r.x * 64.0 + 0.5));
	r = r - j1 * (1.0 / 64.0);                  /* exact */
	int j2 = (int)(floor(r.x * 8192.0 + 0.5));
	r = r - j2 * (1.0 / 8192.0);                /* exact */

	gtd_real s, p;
	double thresh = _td_eps;
	double tmag;

	p = sqr(r);
	s = 1.0 + r + mul_pwr2(p, 0.5);
	int i = 0;
	do {
		p = p * r;
		/* s += p / i!, fused; the term magnitude for the convergence
		   test is estimated from the leading words. */
		tmag = fabs(to_double(p) * td_inv_fact[i].x);
		s = tw_fma(p, td_inv_fact[i], s);
		++i;
	} while (tmag > thresh && i < 10);

	s = s * d_td_exp_table_64[j1 + 23];
	s = s * d_td_exp_table_8192[j2 + 65];

	return ldexp(s, int(m));
}


/* exp(a) - 1, accurate for small |a|.
   For |a| <= log(2)/2 the ln2 reduction in exp() is a no-op (m = 0), and
   the repeated-squaring ladder  s <- 2s + s^2  is exactly the doubling
   rule for u = exp(t) - 1.  So expm1 is exp() without the final +1.
   For larger |a|, exp(a) - 1 suffers no cancellation.                  */
__device__
gtd_real expm1(const gtd_real &a)
{
	const double k = ldexp(1.0, 16);
	const double inv_k = 1.0 / k;

	if (a.x <= -709.0) return make_td(-1.0);
	if (a.x >=  709.0) return make_td(CUDART_INF);

	if (is_zero(a)) return make_td(0.0);

	if (fabs(a.x) > 0.346573590279972655)   /* log(2)/2 */
		return exp(a) - 1.0;

	gtd_real r = mul_pwr2(a, inv_k);
	gtd_real s, p;
	double thresh = inv_k * _td_eps;
	double tmag;

	p = sqr(r);
	s = r + mul_pwr2(p, 0.5);
	int i = 0;
	do {
		p = p * r;
		tmag = fabs(to_double(p) * td_inv_fact[i].x);
		s = tw_fma(p, td_inv_fact[i], s);
		++i;
	} while (tmag > thresh && i < 9);

	for (int j = 0; j < 16; j++)
		s = mul_pwr2(s, 2.0) + sqr(s);

	return s;      /* = exp(a) - 1, no +1 */
}

#endif /* __GTD_EXP_CU__ */
