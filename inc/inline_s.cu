/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Float-overload error-free transformations (Two-Sum / Two-Prod / split / ...)
 * for use by gds_real / gts_real / gqs_real on the device.
 *
 * Splitter constants follow Veltkamp at the float granularity:
 *   _GQS_SPLITTER     = 2^12 + 1 = 4097
 *   _GQS_SPLIT_THRESH = 2^114 ~ 2.0769e+34
 *
 * The host path uses std::fma() for the float overload; the device path
 * uses __fmaf_rn for round-to-nearest IEEE-754 single FMA.  Activated by
 * defining GQS_FMS, which is set up below in lockstep with GQD_FMS.
 */
#ifndef __GDS_GQS_INLINE_CU__
#define __GDS_GQS_INLINE_CU__

#define _GQS_SPLITTER       (4097.0f)             /* = 2^12 + 1 */
#define _GQS_SPLIT_THRESH   (2.0769187e+34f)      /* = 2^114    */

#ifndef GQS_FMS
#  ifdef __CUDA_ARCH__
#    define GQS_FMS(a, b, p) __fmaf_rn((a), (b), -(p))
#  else
#    include <cmath>
#    define GQS_FMS(a, b, p) std::fma((a), (b), -(p))
#  endif
#endif

/****************Basic Functions (float overloads) *********************/

__host__ __device__
float quick_two_sum( float a, float b, float &err )
{
	if (b == 0.0f) {
		err = 0.0f;
		return (a + b);
	}

	float s = a + b;
	err = b - (s - a);
	return s;
}

__host__ __device__
float two_sum( float a, float b, float &err )
{
	if ((a == 0.0f) || (b == 0.0f)) {
		err = 0.0f;
		return (a + b);
	}

	float s  = a + b;
	float bb = s - a;
	err = (a - (s - bb)) + (b - bb);
	return s;
}

__host__ __device__
float quick_two_diff( float a, float b, float &err )
{
	if (a == b) {
		err = 0.0f;
		return 0.0f;
	}

	float s = a - b;
	err = (a - s) - b;
	return s;
}

__host__ __device__
float two_diff( float a, float b, float &err )
{
	if (a == b) {
		err = 0.0f;
		return 0.0f;
	}

	float s  = a - b;
	float bb = s - a;
	err = (a - (s - bb)) - (b + bb);
	return s;
}

__host__ __device__
void split(float a, float &hi, float &lo)
{
	float temp;
	if (a > _GQS_SPLIT_THRESH || a < -_GQS_SPLIT_THRESH) {
		a *= 1.220703125e-4f;          /* 2^-13 */
		temp = _GQS_SPLITTER * a;
		hi = temp - (temp - a);
		lo = a - hi;
		hi *= 8192.0f;                 /* 2^13 */
		lo *= 8192.0f;
	} else {
		temp = _GQS_SPLITTER * a;
		hi = temp - (temp - a);
		lo = a - hi;
	}
}

__device__
float two_prod(float a, float b, float &err)
{
#ifdef GQS_FMS
	float p = a * b;
	err = GQS_FMS(a, b, p);
	return p;
#else
	float a_hi, a_lo, b_hi, b_lo;
	float p = a * b;
	split(a, a_hi, a_lo);
	split(b, b_hi, b_lo);
	err = (a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi + a_lo * b_lo;
	return p;
#endif
}

__host__ __device__
float two_sqr(float a, float &err)
{
#ifdef GQS_FMS
	float p = a * a;
	err = GQS_FMS(a, a, p);
	return p;
#else
	float hi, lo;
	float q = a * a;
	split(a, hi, lo);
	err = ((hi * hi - q) + 2.0f * hi * lo) + lo * lo;
	return q;
#endif
}

__host__ __device__
float nint(float d)
{
	if (d == floorf(d))
		return d;
	return floorf(d + 0.5f);
}

__host__ __device__
void three_sum(float &a, float &b, float &c)
{
	float t1, t2, t3;
	t1 = two_sum(a, b, t2);
	a  = two_sum(c, t1, t3);
	b  = two_sum(t2, t3, c);
}

__host__ __device__
void three_sum2(float &a, float &b, float &c)
{
	float t1, t2, t3;
	t1 = two_sum(a, b, t2);
	a  = two_sum(c, t1, t3);
	b  = (t2 + t3);
}

__host__ __device__
void renormalize3(float x0, float x1, float x2, float x3,
                  float &r0, float &r1, float &r2)
{
	float s, e1, e2, e3;
	s  = quick_two_sum(x2, x3, e3);
	s  = quick_two_sum(x1, s,  e2);
	r0 = quick_two_sum(x0, s,  e1);
	s  = quick_two_sum(e2, e3, e3);
	r1 = quick_two_sum(e1, s,  e2);
	r2 = e2 + e3;
}

#endif /* __GDS_GQS_INLINE_CU__ */
