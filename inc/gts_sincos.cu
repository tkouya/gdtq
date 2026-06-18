/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Triple-float sin / cos / tan.  Same range-reduction strategy as the
 * QD code:
 *   1. reduce mod 2*pi
 *   2. reduce mod pi/2  -> remember quadrant j in [-2,2]
 *   3. reduce mod pi/1024 using the precomputed sin/cos tables
 *      (d_ts_sin_table, d_ts_cos_table; populated by GTDStart()) and a
 *      short Taylor series.
 */

#ifndef __GTS_SIN_COS_CU__
#define __GTS_SIN_COS_CU__

#include "gqs.cuh"

__device__
void sincos_taylor(const gts_real &a, gts_real &sin_a, gts_real &cos_a)
{
	const float thresh = 0.5 * _ts_eps * fabs(to_float(a));
	gts_real p, s, t, x;

	if (is_zero(a)) {
		sin_a = make_ts(0.0);
		cos_a = make_ts(1.0);
		return;
	}

	x = negative(sqr(a));
	s = a;
	p = a;
	int i = 0;
	do {
		p = p * x;
		t = p * ts_inv_fact[i];
		s = s + t;
		i = i + 2;
	} while (i < n_ts_inv_fact && fabs(to_float(t)) > thresh);

	sin_a = s;
	cos_a = sqrt(make_ts(1.0) - sqr(s));
}

__device__
gts_real sin_taylor(const gts_real &a)
{
	const float thresh = 0.5 * _ts_eps * fabs(to_float(a));
	gts_real p, s, t, x;

	if (is_zero(a)) return make_ts(0.0);

	x = negative(sqr(a));
	s = a;
	p = a;
	int i = 0;
	do {
		p = p * x;
		t = p * ts_inv_fact[i];
		s = s + t;
		i += 2;
	} while (i < n_ts_inv_fact && fabs(to_float(t)) > thresh);

	return s;
}

__device__
gts_real cos_taylor(const gts_real &a)
{
	const float thresh = 0.5 * _ts_eps;
	gts_real p, s, t, x;

	if (is_zero(a)) return make_ts(1.0);

	x = negative(sqr(a));
	s = make_ts(1.0) + mul_pwr2(x, 0.5);
	p = x;
	int i = 1;
	do {
		p = p * x;
		t = p * ts_inv_fact[i];
		s = s + t;
		i += 2;
	} while (i < n_ts_inv_fact && fabs(to_float(t)) > thresh);

	return s;
}

__device__
gts_real sin(const gts_real &a)
{
	gts_real z, r;
	if (is_zero(a)) return make_ts(0.0);

	z = nint(a / _ts_2pi);
	r = a - _ts_2pi * z;

	float q = floor(r.x / _ts_pi2.x + 0.5);
	gts_real t = r - _ts_pi2 * q;
	int j = (int)q;
	q = floor(t.x / _ts_pi1024.x + 0.5);
	t = t - _ts_pi1024 * q;
	int k = (int)q;
	int abs_k = abs(k);

	if (j < -2 || j > 2)  return make_ts(0.0);
	if (abs_k > 256)      return make_ts(0.0);

	if (k == 0) {
		switch (j) {
		case  0: return sin_taylor(t);
		case  1: return cos_taylor(t);
		case -1: return negative(cos_taylor(t));
		default: return negative(sin_taylor(t));
		}
	}

	gts_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	gts_real u = d_ts_cos_table[abs_k - 1];
	gts_real v = d_ts_sin_table[abs_k - 1];

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
gts_real cos(const gts_real &a)
{
	if (is_zero(a)) return make_ts(1.0);

	gts_real z = nint(a / _ts_2pi);
	gts_real r = a - _ts_2pi * z;

	float q = floor(r.x / _ts_pi2.x + 0.5);
	gts_real t = r - _ts_pi2 * q;
	int j = (int)q;
	q = floor(t.x / _ts_pi1024.x + 0.5);
	t = t - _ts_pi1024 * q;
	int k = (int)q;
	int abs_k = abs(k);

	if (j < -2 || j > 2)  return make_ts(0.0);
	if (abs_k > 256)      return make_ts(0.0);

	if (k == 0) {
		switch (j) {
		case  0: return cos_taylor(t);
		case  1: return negative(sin_taylor(t));
		case -1: return sin_taylor(t);
		default: return negative(cos_taylor(t));
		}
	}

	gts_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	gts_real u = d_ts_cos_table[abs_k - 1];
	gts_real v = d_ts_sin_table[abs_k - 1];

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
void sincos(const gts_real &a, gts_real &sin_a, gts_real &cos_a)
{
	if (is_zero(a)) {
		sin_a = make_ts(0.0);
		cos_a = make_ts(1.0);
		return;
	}

	gts_real z = nint(a / _ts_2pi);
	gts_real t = a - _ts_2pi * z;

	float q = floor(t.x / _ts_pi2.x + 0.5);
	t = t - _ts_pi2 * q;
	int j = (int)q;
	q = floor(t.x / _ts_pi1024.x + 0.5);
	t = t - _ts_pi1024 * q;
	int k = (int)q;
	int abs_k = abs(k);

	if (j < -2 || j > 2)  { sin_a = cos_a = make_ts(0.0); return; }
	if (abs_k > 256)      { sin_a = cos_a = make_ts(0.0); return; }

	gts_real sin_t, cos_t;
	sincos_taylor(t, sin_t, cos_t);

	if (k == 0) {
		if      (j ==  0) { sin_a = sin_t;            cos_a = cos_t; }
		else if (j ==  1) { sin_a = cos_t;            cos_a = negative(sin_t); }
		else if (j == -1) { sin_a = negative(cos_t);  cos_a = sin_t; }
		else              { sin_a = negative(sin_t);  cos_a = negative(cos_t); }
		return;
	}

	gts_real u = d_ts_cos_table[abs_k - 1];
	gts_real v = d_ts_sin_table[abs_k - 1];

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
gts_real tan(const gts_real &a)
{
	gts_real s, c;
	sincos(a, s, c);
	return s / c;
}

#endif /* __GTS_SIN_COS_CU__ */
