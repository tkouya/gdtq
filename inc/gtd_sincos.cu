/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Triple-double sin / cos / tan.  Same range-reduction strategy as the
 * QD code:
 *   1. reduce mod 2*pi
 *   2. reduce mod pi/2  -> remember quadrant j in [-2,2]
 *   3. reduce mod pi/1024 using the precomputed sin/cos tables
 *      (d_td_sin_table, d_td_cos_table; populated by GTDStart()) and a
 *      short Taylor series.
 */

#ifndef __GTD_SIN_COS_CU__
#define __GTD_SIN_COS_CU__

#include "gqd.cuh"

__device__
void sincos_taylor(const gtd_real &a, gtd_real &sin_a, gtd_real &cos_a)
{
	const double thresh = 0.5 * _td_eps * fabs(to_double(a));
	gtd_real p, s, t, x;

	if (is_zero(a)) {
		sin_a = make_td(0.0);
		cos_a = make_td(1.0);
		return;
	}

	x = negative(sqr(a));
	s = a;
	p = a;
	int i = 0;
	do {
		p = p * x;
		t = p * td_inv_fact[i];
		s = s + t;
		i = i + 2;
	} while (i < n_td_inv_fact && fabs(to_double(t)) > thresh);

	sin_a = s;
	cos_a = sqrt(make_td(1.0) - sqr(s));
}

__device__
gtd_real sin_taylor(const gtd_real &a)
{
	const double thresh = 0.5 * _td_eps * fabs(to_double(a));
	gtd_real p, s, t, x;

	if (is_zero(a)) return make_td(0.0);

	x = negative(sqr(a));
	s = a;
	p = a;
	int i = 0;
	do {
		p = p * x;
		t = p * td_inv_fact[i];
		s = s + t;
		i += 2;
	} while (i < n_td_inv_fact && fabs(to_double(t)) > thresh);

	return s;
}

__device__
gtd_real cos_taylor(const gtd_real &a)
{
	const double thresh = 0.5 * _td_eps;
	gtd_real p, s, t, x;

	if (is_zero(a)) return make_td(1.0);

	x = negative(sqr(a));
	s = make_td(1.0) + mul_pwr2(x, 0.5);
	p = x;
	int i = 1;
	do {
		p = p * x;
		t = p * td_inv_fact[i];
		s = s + t;
		i += 2;
	} while (i < n_td_inv_fact && fabs(to_double(t)) > thresh);

	return s;
}

__device__
gtd_real sin(const gtd_real &a)
{
	gtd_real z, r;
	if (is_zero(a)) return make_td(0.0);

	z = nint(a / _td_2pi);
	r = a - _td_2pi * z;

	double q = floor(r.x / _td_pi2.x + 0.5);
	gtd_real t = r - _td_pi2 * q;
	int j = (int)q;
	q = floor(t.x / _td_pi1024.x + 0.5);
	t = t - _td_pi1024 * q;
	int k = (int)q;
	int abs_k = abs(k);

	if (j < -2 || j > 2)  return make_td(0.0);
	if (abs_k > 256)      return make_td(0.0);

	if (k == 0) {
		switch (j) {
		case  0: return sin_taylor(t);
		case  1: return cos_taylor(t);
		case -1: return negative(cos_taylor(t));
		default: return negative(sin_taylor(t));
		}
	}

	gtd_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	gtd_real u = d_td_cos_table[abs_k - 1];
	gtd_real v = d_td_sin_table[abs_k - 1];

	if (j == 0) {
		if (k > 0) return u * sin_t + v * cos_t;
		else       return u * sin_t - v * cos_t;
	} else if (j == 1) {
		if (k > 0) return u * cos_t - v * sin_t;
		else       return u * cos_t + v * sin_t;
	} else if (j == -1) {
		if (k > 0) return v * sin_t - u * cos_t;
		else       return negative(u * cos_t) - v * sin_t;
	} else {
		if (k > 0) return negative(u * sin_t) - v * cos_t;
		else       return v * cos_t - u * sin_t;
	}
}

__device__
gtd_real cos(const gtd_real &a)
{
	if (is_zero(a)) return make_td(1.0);

	gtd_real z = nint(a / _td_2pi);
	gtd_real r = a - _td_2pi * z;

	double q = floor(r.x / _td_pi2.x + 0.5);
	gtd_real t = r - _td_pi2 * q;
	int j = (int)q;
	q = floor(t.x / _td_pi1024.x + 0.5);
	t = t - _td_pi1024 * q;
	int k = (int)q;
	int abs_k = abs(k);

	if (j < -2 || j > 2)  return make_td(0.0);
	if (abs_k > 256)      return make_td(0.0);

	if (k == 0) {
		switch (j) {
		case  0: return cos_taylor(t);
		case  1: return negative(sin_taylor(t));
		case -1: return sin_taylor(t);
		default: return negative(cos_taylor(t));
		}
	}

	gtd_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	gtd_real u = d_td_cos_table[abs_k - 1];
	gtd_real v = d_td_sin_table[abs_k - 1];

	if (j == 0) {
		if (k > 0) return u * cos_t - v * sin_t;
		else       return u * cos_t + v * sin_t;
	} else if (j == 1) {
		if (k > 0) return negative(u * sin_t) - v * cos_t;
		else       return v * cos_t - u * sin_t;
	} else if (j == -1) {
		if (k > 0) return u * sin_t + v * cos_t;
		else       return u * sin_t - v * cos_t;
	} else {
		if (k > 0) return v * sin_t - u * cos_t;
		else       return negative(u * cos_t) - v * sin_t;
	}
}

__device__
void sincos(const gtd_real &a, gtd_real &sin_a, gtd_real &cos_a)
{
	if (is_zero(a)) {
		sin_a = make_td(0.0);
		cos_a = make_td(1.0);
		return;
	}

	gtd_real z = nint(a / _td_2pi);
	gtd_real t = a - _td_2pi * z;

	double q = floor(t.x / _td_pi2.x + 0.5);
	t = t - _td_pi2 * q;
	int j = (int)q;
	q = floor(t.x / _td_pi1024.x + 0.5);
	t = t - _td_pi1024 * q;
	int k = (int)q;
	int abs_k = abs(k);

	if (j < -2 || j > 2)  { sin_a = cos_a = make_td(0.0); return; }
	if (abs_k > 256)      { sin_a = cos_a = make_td(0.0); return; }

	gtd_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	if (k == 0) {
		if      (j ==  0) { sin_a = sin_t;            cos_a = cos_t; }
		else if (j ==  1) { sin_a = cos_t;            cos_a = negative(sin_t); }
		else if (j == -1) { sin_a = negative(cos_t);  cos_a = sin_t; }
		else              { sin_a = negative(sin_t);  cos_a = negative(cos_t); }
		return;
	}

	gtd_real u = d_td_cos_table[abs_k - 1];
	gtd_real v = d_td_sin_table[abs_k - 1];

	if (j == 0) {
		if (k > 0) { sin_a = u*sin_t + v*cos_t;            cos_a = u*cos_t - v*sin_t; }
		else       { sin_a = u*sin_t - v*cos_t;            cos_a = u*cos_t + v*sin_t; }
	} else if (j == 1) {
		if (k > 0) { cos_a = negative(u*sin_t) - v*cos_t;  sin_a = u*cos_t - v*sin_t; }
		else       { cos_a = v*cos_t - u*sin_t;            sin_a = u*cos_t + v*sin_t; }
	} else if (j == -1) {
		if (k > 0) { cos_a = u*sin_t + v*cos_t;            sin_a = v*sin_t - u*cos_t; }
		else       { cos_a = u*sin_t - v*cos_t;            sin_a = negative(u*cos_t) - v*sin_t; }
	} else {
		if (k > 0) { sin_a = negative(u*sin_t) - v*cos_t;  cos_a = v*sin_t - u*cos_t; }
		else       { sin_a = v*cos_t - u*sin_t;            cos_a = negative(u*cos_t) - v*sin_t; }
	}
}

__device__
gtd_real tan(const gtd_real &a)
{
	gtd_real s, c;
	sincos(a, s, c);
	return s / c;
}

#endif /* __GTD_SIN_COS_CU__ */
