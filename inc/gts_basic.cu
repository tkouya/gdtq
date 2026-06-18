/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Triple-float basic operators (gts_real = (x, y, z) with
 * |x| > |y| > |z|, ulp-nonoverlapping).  Algorithms follow CAMPARY
 * (Joldes-Muller-Popescu, "Tight and rigorous error bounds for basic
 * building blocks of float-word arithmetic", ACM TOMS 2017) and the
 * standard Bailey/Hida QD truncation pattern collapsed from 4 to 3
 * components.
 */

#ifndef __GTS_BASIC_CU__
#define __GTS_BASIC_CU__

#include "gqs.cuh"

/*=================================================================
 * renormalization
 *=================================================================*/
__host__ __device__
void renorm(gts_real &x)
{
	float r0, r1, r2;
	renormalize3(x.x, x.y, x.z, 0.0, r0, r1, r2);
	x.x = r0;  x.y = r1;  x.z = r2;
}

__host__ __device__
gts_real make_ts_renorm(float x0, float x1, float x2, float x3)
{
	float r0, r1, r2;
	renormalize3(x0, x1, x2, x3, r0, r1, r2);
	return make_ts(r0, r1, r2);
}

/*=================================================================
 * type construction helpers (forward to make_ts defined in common.cu)
 *=================================================================*/

/*=================================================================
 * additions
 *=================================================================*/

/* td + float */
__host__ __device__
gts_real operator+(const gts_real &a, float b)
{
	float c0, c1, c2, e;
	c0 = two_sum(a.x, b, e);
	c1 = two_sum(a.y, e, e);
	c2 = a.z + e;
	return make_ts_renorm(c0, c1, c2, 0.0);
}

__host__ __device__
gts_real operator+(float a, const gts_real &b) { return b + a; }

/* td + td  (CAMPARY Algorithm 6 specialised to 3 limbs) */
//__host__ __device__
//gts_real operator+(const gts_real &a, const gts_real &b)
__host__ __device__
gts_real standard_add(const gts_real &a, const gts_real &b)
{
	float s0, s1, s2;
	float e0, e1, t1;

	s0 = two_sum(a.x, b.x, e0);
	s1 = two_sum(a.y, b.y, e1);
	/* fold e0 into the second column */
	s1 = two_sum(s1, e0, t1);
	/* third column: a.z + b.z + e1 + t1 */
	s2 = a.z + b.z + e1 + t1;

	return make_ts_renorm(s0, s1, s2, 0.0);
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
//void Add3(const float x[3], const float y[3], float z[3]) {
//static inline void c_td_add_bf(float *a, float *b, float *c)
__host__ __device__
gts_real bf_add(const gts_real &a, const gts_real &b)
{
  float a0, b0, c0, d0, e0, f0;
  float a1, b1, c1, d1, e1, f1;
  float a2, b2, c2, d2, e2;
  float a3, b3, c3, d3;
  float c4;
  float c5, d5;
  float b6, c6;
  float a7, b7, c7;
  float b8, c8;

  a0 = a.x;
  b0 = b.x;
  c0 = a.y;
  d0 = b.y;
  e0 = a.z;
  f0 = b.z;
  a1 = two_sum(a0, b0, b1);
  c1 = two_sum(c0, d0, d1);
  e1 = two_sum(e0, f0, f1);
  a2 = quick_two_sum(a1, c1, c2);
  b2 = b1 + f1;
  d2 = two_sum(d1, e1, e2);
  a3 = quick_two_sum(a2, d2, d3);
  b3 = two_sum(b2, c2, c3);
  c4 = c3 + e2;
  c5 = two_sum(c4, d3, d5);
  b6 = two_sum(b3, c5, c6);
  a7 = quick_two_sum(a3, b6, b7);
  c7 = c6 + d5;
  b8 = quick_two_sum(b7, c7, c8);

  return make_ts(a7, b8, c8);
}

/* triple-float + triple-float
 *
 * NOTE: not `inline`, and explicitly `__host__ __device__`.  With nvcc
 * -rdc=true an inline function is emitted as a weak symbol only when it
 * is actually used in the TU, so a TU that merely declares it (e.g.
 * matmul_strassen_general_gts.cu / gtslinear.cu via the gqs.cuh header
 * chain) cannot resolve the device-side call at -dlink time when the
 * only TU that compiles gts_basic.cu (gdslinear.cu) does not itself
 * call gts_real + gts_real from its own kernels.  A non-inline
 * __host__ __device__ definition forces a strong external device symbol
 * from gdslinear's TU, which is what nvcc -dlink needs.  All sibling
 * operators (-, *, /, ...) are already non-inline. */
__host__ __device__
gts_real operator+(const gts_real &a, const gts_real &b)
{
#ifdef USE_STANDARD_ADD
  return standard_add(a, b);
#else // USE_STANDARD_ADD
  return bf_add(a, b);
#endif // USE_STANDARD_ADD
}

/*=================================================================
 * subtractions
 *=================================================================*/
__host__ __device__
gts_real negative(const gts_real &a)
{
	return make_ts(-a.x, -a.y, -a.z);
}

__host__ __device__
gts_real operator-(const gts_real &a, float b) { return a + (-b); }

__host__ __device__
gts_real operator-(float a, const gts_real &b) { return a + negative(b); }

__host__ __device__
gts_real operator-(const gts_real &a, const gts_real &b) { return a + negative(b); }

/*=================================================================
 * multiplications
 *=================================================================*/
__host__ __device__
gts_real mul_pwr2(const gts_real &a, float b)
{
	return make_ts(a.x * b, a.y * b, a.z * b);
}

/* td * float */
__device__
gts_real operator*(const gts_real &a, float b)
{
	float p0, p1, p2;
	float q0, q1;
	float s0, s1, s2, s3;

	p0 = two_prod(a.x, b, q0);
	p1 = two_prod(a.y, b, q1);
	p2 = a.z * b;

	s0 = p0;
	s1 = two_sum(q0, p1, s2);     /* s2 receives the error */
	three_sum(s2, q1, p2);        /* combine the lower column */
	s3 = q1 + p2;                 /* tail */

	return make_ts_renorm(s0, s1, s2, s3);
}

__device__
gts_real operator*(float a, const gts_real &b) { return b * a; }

/* td * td  (CAMPARY truncated TD multiplication, 3-limb output)
 *
 * Cross terms by order:
 *   2^0   :  a0*b0
 *   2^-53 :  a0*b1, a1*b0,  err(a0*b0)
 *   2^-106:  a0*b2, a1*b1, a2*b0,  err(a0*b1), err(a1*b0)
 *   below : truncated
 */
//__device__
//gts_real operator*(const gts_real &a, const gts_real &b)
__device__
gts_real standard_mul(const gts_real &a, const gts_real &b)
{
	float p00, e00;
	float p01, e01;
	float p10, e10;
	float p02, p11, p20;
	float s0, s1, s2, s3;

	p00 = two_prod(a.x, b.x, e00);          /* 2^0  + 2^-53 */
	p01 = two_prod(a.x, b.y, e01);          /* 2^-53 */
	p10 = two_prod(a.y, b.x, e10);          /* 2^-53 */
	p02 = a.x * b.z;                        /* 2^-106 */
	p11 = a.y * b.y;                        /* 2^-106 */
	p20 = a.z * b.x;                        /* 2^-106 */

	s0 = p00;

	/* 2^-53 column: e00, p01, p10  -> s1 plus carry */
	s1 = two_sum(e00, p01, s2);             /* s1 = e00+p01 hi, s2 = err */
	float t;
	s1 = two_sum(s1, p10, t);               /* fold p10 in */
	s2 = s2 + t;

	/* 2^-106 column: p02 + p11 + p20 + (err's e01, e10) plus s2 carry */
	s3 = p02 + p11 + p20 + e01 + e10;

	/* combine s2 and s3 */
	s2 = two_sum(s2, s3, t);
	s3 = t;

	return make_ts_renorm(s0, s1, s2, s3);
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
// void Mul3(const float x[3], const float y[3], float z[3]) {
//static inline void c_td_mul_bf(float *a, float *b, float *c)
__device__
gts_real bf_mul(const gts_real &a, const gts_real &b)
{
  float a0, b0, c0, d0, e0, f0, g0, h0, i0;
  float c1, d1, e1, f1, g1, h1, i1;
  float b2, c2, g2;
  float a3, b3, c3, d3, e3, g3;
  float c4, e4;
  float b5, c5;
  float a6, b6;
  float b7, c7;

  a0 = two_prod(a.x, b.x, b0);
  c0 = two_prod(a.x, b.y, e0);
  d0 = two_prod(a.y, b.x, f0);
  g0 = a.x * b.z;
  h0 = a.y * b.y;
  i0 = a.z * b.x;
  c1 = two_sum(c0, d0, d1);
  e1 = two_sum(e0, f0, f1);
  g1 = two_sum(g0, i0, i1);
  b2 = two_sum(b0, c1, c2);
  g2 = two_sum(g1, h0, h1);
  a3 = quick_two_sum(a0, b2, b3);
  c3 = two_sum(c2, d1, d3);
  e3 = two_sum(e1, g2, g3);
  c4 = two_sum(c3, e3, e4);
  b5 = quick_two_sum(b3, c4, c5);
  a6 = quick_two_sum(a3, b5, b6);
  b7 = quick_two_sum(b6, c5, c7);

  return make_ts(a6, b7, c7);
}

__device__
gts_real operator*(const gts_real &a, const gts_real &b)
{
#ifdef GQD_STANDARD_MUL
  return standard_mul(a, b);
#else // GQD_STANDARD_MUL
  return bf_mul(a, b);
#endif // GQD_STANDARD_MUL
}

__device__
gts_real sqr(const gts_real &a)
{
	float p00, e00, p01, e01, p11;
	float s0, s1, s2, s3, t;

	p00 = two_sqr(a.x, e00);                /* a0^2     */
	p01 = two_prod(a.x, a.y, e01);          /* a0*a1    */
	p11 = a.y * a.y;                        /* 2^-106   */
	float p02 = a.x * a.z;                 /* 2^-106   */

	s0 = p00;
	/* 2^-53 column: e00 + 2*p01 */
	float two_p01 = 2.0 * p01;
	s1 = two_sum(e00, two_p01, s2);

	/* 2^-106 column: 2*e01 + p11 + 2*p02 */
	s3 = 2.0 * e01 + p11 + 2.0 * p02;

	s2 = two_sum(s2, s3, t);
	s3 = t;

	return make_ts_renorm(s0, s1, s2, s3);
}

/*=================================================================
 * divisions
 *=================================================================*/

/* Newton refinement: q = a / b
 *   q0 = a.x / b.x
 *   r  = a - q0 * b
 *   q1 = r.x / b.x
 *   r  = r - q1 * b
 *   q2 = r.x / b.x
 *   q  = renormalize3(q0, q1, q2, r.x/b.x)
 */
__device__
gts_real operator/(const gts_real &a, const gts_real &b)
{
	float q0, q1, q2, q3;
	gts_real r;

	q0 = a.x / b.x;
	r  = a - (b * q0);

	q1 = r.x / b.x;
	r  = r - (b * q1);

	q2 = r.x / b.x;
	r  = r - (b * q2);

	q3 = r.x / b.x;

	return make_ts_renorm(q0, q1, q2, q3);
}

__device__
gts_real operator/(const gts_real &a, float b)
{
	float q0, q1, q2, q3;
	gts_real r;

	q0 = a.x / b;
	r  = a - (q0 * b);

	q1 = r.x / b;
	r  = r - (q1 * b);

	q2 = r.x / b;
	r  = r - (q2 * b);

	q3 = r.x / b;

	return make_ts_renorm(q0, q1, q2, q3);
}

__device__
gts_real operator/(float a, const gts_real &b)
{
	return make_ts(a) / b;
}

__device__
gts_real inv(const gts_real &a)
{
	return 1.0 / a;
}

/*=================================================================
 * miscellaneous
 *=================================================================*/
__host__ __device__
gts_real abs(const gts_real &a)
{
	return (a.x < 0.0) ? negative(a) : a;
}

__host__ __device__
gts_real fabs(const gts_real &a) { return abs(a); }

__host__ __device__
float to_float(const gts_real &a) { return a.x; }

__host__ __device__
gts_real ldexp(const gts_real &a, int n)
{
	return make_ts(::ldexp(a.x, n), ::ldexp(a.y, n), ::ldexp(a.z, n));
}

__device__
gts_real nint(const gts_real &a)
{
	float x0 = nint(a.x);
	float x1 = 0.0, x2 = 0.0;

	if (x0 == a.x) {
		/* x0 is integer; round y, z next */
		x1 = nint(a.y);
		if (x1 == a.y)
			x2 = nint(a.z);
		else if (fabs(x1 - a.y) == 0.5 && a.z < 0.0)
			x1 -= 1.0;
	} else if (fabs(x0 - a.x) == 0.5 && a.y < 0.0) {
		x0 -= 1.0;
	}
	return make_ts_renorm(x0, x1, x2, 0.0);
}

/*=================================================================
 * comparisons
 *=================================================================*/
__host__ __device__ bool is_zero    (const gts_real &a) { return a.x == 0.0; }
__host__ __device__ bool is_one     (const gts_real &a) { return a.x == 1.0 && a.y == 0.0 && a.z == 0.0; }
__host__ __device__ bool is_positive(const gts_real &a) { return a.x >  0.0; }
__host__ __device__ bool is_negative(const gts_real &a) { return a.x <  0.0; }

__host__ __device__ bool operator==(const gts_real &a, const gts_real &b)
{ return a.x == b.x && a.y == b.y && a.z == b.z; }
__host__ __device__ bool operator==(const gts_real &a, float b)
{ return a.x == b && a.y == 0.0 && a.z == 0.0; }
__host__ __device__ bool operator==(float a, const gts_real &b)
{ return b == a; }

__host__ __device__ bool operator!=(const gts_real &a, const gts_real &b) { return !(a == b); }
__host__ __device__ bool operator!=(const gts_real &a, float b)          { return !(a == b); }
__host__ __device__ bool operator!=(float a, const gts_real &b)          { return !(a == b); }

__host__ __device__ bool operator<(const gts_real &a, const gts_real &b)
{
	if (a.x != b.x) return a.x < b.x;
	if (a.y != b.y) return a.y < b.y;
	return a.z < b.z;
}
__host__ __device__ bool operator<(const gts_real &a, float b)
{
	if (a.x != b) return a.x < b;
	if (a.y != 0.0) return a.y < 0.0;
	return a.z < 0.0;
}
__host__ __device__ bool operator<(float a, const gts_real &b) { return b > a; }

__host__ __device__ bool operator>(const gts_real &a, const gts_real &b)
{
	if (a.x != b.x) return a.x > b.x;
	if (a.y != b.y) return a.y > b.y;
	return a.z > b.z;
}
__host__ __device__ bool operator>(const gts_real &a, float b)
{
	if (a.x != b) return a.x > b;
	if (a.y != 0.0) return a.y > 0.0;
	return a.z > 0.0;
}
__host__ __device__ bool operator>(float a, const gts_real &b) { return b < a; }

__host__ __device__ bool operator<=(const gts_real &a, const gts_real &b) { return !(a > b); }
__host__ __device__ bool operator<=(const gts_real &a, float b)          { return !(a > b); }
__host__ __device__ bool operator<=(float a, const gts_real &b)          { return !(a > b); }

__host__ __device__ bool operator>=(const gts_real &a, const gts_real &b) { return !(a < b); }
__host__ __device__ bool operator>=(const gts_real &a, float b)          { return !(a < b); }
__host__ __device__ bool operator>=(float a, const gts_real &b)          { return !(a < b); }

#endif /* __GTS_BASIC_CU__ */
