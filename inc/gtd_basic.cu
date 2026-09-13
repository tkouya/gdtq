/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL).
 *
 * Triple-double basic operators (gtd_real = (x, y, z) with
 * |x| > |y| > |z|, ulp-nonoverlapping).  Algorithms follow CAMPARY
 * (Joldes-Muller-Popescu, "Tight and rigorous error bounds for basic
 * building blocks of double-word arithmetic", ACM TOMS 2017) and the
 * standard Bailey/Hida QD truncation pattern collapsed from 4 to 3
 * components.
 */

#ifndef __GTD_BASIC_CU__
#define __GTD_BASIC_CU__

#include "gqd.cuh"

/*=================================================================
 * renormalization
 *=================================================================*/
__host__ __device__
void renorm(gtd_real &x)
{
	double r0, r1, r2;
	renormalize3(x.x, x.y, x.z, 0.0, r0, r1, r2);
	x.x = r0;  x.y = r1;  x.z = r2;
}

__host__ __device__
gtd_real make_td_renorm(double x0, double x1, double x2, double x3)
{
	double r0, r1, r2;
	renormalize3(x0, x1, x2, x3, r0, r1, r2);
	return make_td(r0, r1, r2);
}

/*=================================================================
 * type construction helpers (forward to make_td defined in common.cu)
 *=================================================================*/

/*=================================================================
 * additions
 *=================================================================*/

/* td + double */
__host__ __device__
gtd_real operator+(const gtd_real &a, double b)
{
	double c0, c1, c2, e;
	c0 = two_sum(a.x, b, e);
	c1 = two_sum(a.y, e, e);
	c2 = a.z + e;
	return make_td_renorm(c0, c1, c2, 0.0);
}

__host__ __device__
gtd_real operator+(double a, const gtd_real &b) { return b + a; }

/* td + td  (CAMPARY Algorithm 6 specialised to 3 limbs) */
//__host__ __device__
//gtd_real operator+(const gtd_real &a, const gtd_real &b)
__host__ __device__
gtd_real standard_add(const gtd_real &a, const gtd_real &b)
{
	double s0, s1, s2;
	double e0, e1, t1;

	s0 = two_sum(a.x, b.x, e0);
	s1 = two_sum(a.y, b.y, e1);
	/* fold e0 into the second column */
	s1 = two_sum(s1, e0, t1);
	/* third column: a.z + b.z + e1 + t1 */
	s2 = a.z + b.z + e1 + t1;

	return make_td_renorm(s0, s1, s2, 0.0);
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
//void Add3(const double x[3], const double y[3], double z[3]) {
//static inline void c_td_add_bf(double *a, double *b, double *c)
__host__ __device__
gtd_real bf_add(const gtd_real &a, const gtd_real &b)
{
  double a0, b0, c0, d0, e0, f0;
  double a1, b1, c1, d1, e1, f1;
  double a2, b2, c2, d2, e2;
  double a3, b3, c3, d3;
  double c4;
  double c5, d5;
  double b6, c6;
  double a7, b7, c7;
  double b8, c8;

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

  return make_td(a7, b8, c8);
}

/* triple-double + triple-double
 *
 * NOTE: not `inline`.  With nvcc -rdc=true, an inline function is
 * emitted as a weak symbol only when it is actually used in the TU,
 * which means a TU that only declares it (e.g. matmul_strassen_general_gtd.cu
 * via the gqd.cuh header chain) cannot resolve the call at device-link
 * time when the only TU that compiles gtd_basic.cu (gddlinear.cu) does
 * not itself call gtd_real + gtd_real from its own kernels.  Marking it
 * non-inline forces a strong external symbol from gddlinear's TU, which
 * is what nvcc -dlink needs.  All sibling operators (-, *, /, ...) are
 * already non-inline. */
__host__ __device__
gtd_real operator+(const gtd_real &a, const gtd_real &b)
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
gtd_real negative(const gtd_real &a)
{
	return make_td(-a.x, -a.y, -a.z);
}

__host__ __device__
gtd_real operator-(const gtd_real &a, double b) { return a + (-b); }

__host__ __device__
gtd_real operator-(double a, const gtd_real &b) { return a + negative(b); }

__host__ __device__
gtd_real operator-(const gtd_real &a, const gtd_real &b) { return a + negative(b); }

/*=================================================================
 * multiplications
 *=================================================================*/
__host__ __device__
gtd_real mul_pwr2(const gtd_real &a, double b)
{
	return make_td(a.x * b, a.y * b, a.z * b);
}

/* td * double */
__device__
gtd_real operator*(const gtd_real &a, double b)
{
	double p0, p1, p2;
	double q0, q1;
	double s0, s1, s2, s3;

	p0 = two_prod(a.x, b, q0);
	p1 = two_prod(a.y, b, q1);
	p2 = a.z * b;

	s0 = p0;
	s1 = two_sum(q0, p1, s2);     /* s2 receives the error */
	three_sum(s2, q1, p2);        /* combine the lower column */
	s3 = q1 + p2;                 /* tail */

	return make_td_renorm(s0, s1, s2, s3);
}

__device__
gtd_real operator*(double a, const gtd_real &b) { return b * a; }

/* td * td  (CAMPARY truncated TD multiplication, 3-limb output)
 *
 * Cross terms by order:
 *   2^0   :  a0*b0
 *   2^-53 :  a0*b1, a1*b0,  err(a0*b0)
 *   2^-106:  a0*b2, a1*b1, a2*b0,  err(a0*b1), err(a1*b0)
 *   below : truncated
 */
//__device__
//gtd_real operator*(const gtd_real &a, const gtd_real &b)
__device__
gtd_real standard_mul(const gtd_real &a, const gtd_real &b)
{
	double p00, e00;
	double p01, e01;
	double p10, e10;
	double p02, p11, p20;
	double s0, s1, s2, s3;

	p00 = two_prod(a.x, b.x, e00);          /* 2^0  + 2^-53 */
	p01 = two_prod(a.x, b.y, e01);          /* 2^-53 */
	p10 = two_prod(a.y, b.x, e10);          /* 2^-53 */
	p02 = a.x * b.z;                        /* 2^-106 */
	p11 = a.y * b.y;                        /* 2^-106 */
	p20 = a.z * b.x;                        /* 2^-106 */

	s0 = p00;

	/* 2^-53 column: e00, p01, p10  -> s1 plus carry */
	s1 = two_sum(e00, p01, s2);             /* s1 = e00+p01 hi, s2 = err */
	double t;
	s1 = two_sum(s1, p10, t);               /* fold p10 in */
	s2 = s2 + t;

	/* 2^-106 column: p02 + p11 + p20 + (err's e01, e10) plus s2 carry */
	s3 = p02 + p11 + p20 + e01 + e10;

	/* combine s2 and s3 */
	s2 = two_sum(s2, s3, t);
	s3 = t;

	return make_td_renorm(s0, s1, s2, s3);
}

// 2025-12-24(Wed) T.Kouya
// Branch free algorithm
// void Mul3(const double x[3], const double y[3], double z[3]) {
//static inline void c_td_mul_bf(double *a, double *b, double *c)
__device__
gtd_real bf_mul(const gtd_real &a, const gtd_real &b)
{
  double a0, b0, c0, d0, e0, f0, g0, h0, i0;
  double c1, d1, e1, f1, g1, h1, i1;
  double b2, c2, g2;
  double a3, b3, c3, d3, e3, g3;
  double c4, e4;
  double b5, c5;
  double a6, b6;
  double b7, c7;

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

  return make_td(a6, b7, c7);
}

__device__
gtd_real operator*(const gtd_real &a, const gtd_real &b)
{
#ifdef GQD_STANDARD_MUL
  return standard_mul(a, b);
#else // GQD_STANDARD_MUL
  return bf_mul(a, b);
#endif // GQD_STANDARD_MUL
}

__device__
gtd_real sqr(const gtd_real &a)
{
	double p00, e00, p01, e01, p11;
	double s0, s1, s2, s3, t;

	p00 = two_sqr(a.x, e00);                /* a0^2     */
	p01 = two_prod(a.x, a.y, e01);          /* a0*a1    */
	p11 = a.y * a.y;                        /* 2^-106   */
	double p02 = a.x * a.z;                 /* 2^-106   */

	s0 = p00;
	/* 2^-53 column: e00 + 2*p01 */
	double two_p01 = 2.0 * p01;
	s1 = two_sum(e00, two_p01, s2);

	/* 2^-106 column: 2*e01 + p11 + 2*p02 */
	s3 = 2.0 * e01 + p11 + 2.0 * p02;

	s2 = two_sum(s2, s3, t);
	s3 = t;

	return make_td_renorm(s0, s1, s2, s3);
}

/*=================================================================
 * fused multiply-add
 *=================================================================*/

// 2026-08-04 T.Kouya
// Branch free algorithm: triple-word FMA,  z = a * b + c.
//
// Every term of the exact product a*b and every word of c is assigned to
// the magnitude level it belongs to (level 0 ~ 1, level 1 ~ eps,
// level 2 ~ eps^2).  The levels are accumulated with two_sum, the errors
// spilling one level down, and the three level accumulators are
// renormalized by the same quick_two_sum network used by bf_mul above.
// No branch anywhere, and a*b is never renormalized on its own.
/* TW-FMA  z = a * b + c   (72 flops)
   Machine-proved with FPANVerifier + z3 5.0.0 (ACS2026 formulation):
     error bound      |z-(ab+c)| <= 187 u^3 (|ab|+|c|)
                      (FPANVerifier anchor-relative 184 u^3 plus the
                       analytic input-relative conversion)
     every FastTwoSum precondition  exp(x) >= exp(y)
     non-overlapping output         z0 |> z1 |> z2   (strongly_dominates)
   Normalization repeats a cascade over adjacent pairs; the pass count is the
   smallest for which the non-overlap is provable (DW 1 / TW 3 / QW 5). */
__device__
gtd_real tw_fma(const gtd_real &a, const gtd_real &b, const gtd_real &c)
{
  double P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
  double A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;
  P00 = two_prod(a.x, b.x, E00);
  P01 = two_prod(a.x, b.y, E01);
  P10 = two_prod(a.y, b.x, E10);
  P02 = a.x * b.z;
  P11 = a.y * b.y;
  P20 = a.z * b.x;
  sg  = (P02 + P20) + P11;
  G   = ((E01 + E10) + sg) + c.z;
  A   = two_sum(P01, P10, q1);
  A   = two_sum(A, E00, q2);
  A   = two_sum(A, c.y, q3);
  G   = G + ((q1 + q2) + q3);
  B   = two_sum(P00, c.x, r);
  m1  = two_sum(r, A, m2);
  m2  = m2 + G;
  double z0, z1, z2;
  /* renormalization pass 1/3 */
  w0 = quick_two_sum(B, m1, w1);
  w1 = two_sum(w1, m2, w2);
  /* renormalization pass 2/3 */
  w0 = two_sum(w0, w1, w1);
  w1 = quick_two_sum(w1, w2, w2);
  /* renormalization pass 3/3 */
  z0 = quick_two_sum(w0, w1, w1);
  z1 = quick_two_sum(w1, w2, z2);
  return make_td(z0, z1, z2);
}

/* TW-FMA  z = a * b + c   (72 flops, scalar multiplier)
   Machine-proved with FPANVerifier + z3 5.0.0 (ACS2026 formulation):
     error bound      |z-(ab+c)| <= 187 u^3 (|ab|+|c|)
                      (FPANVerifier anchor-relative 184 u^3 plus the
                       analytic input-relative conversion)
     every FastTwoSum precondition  exp(x) >= exp(y)
     non-overlapping output         z0 |> z1 |> z2   (strongly_dominates)
   Normalization repeats a cascade over adjacent pairs; the pass count is the
   smallest for which the non-overlap is provable (DW 1 / TW 3 / QW 5). */
/* div/sqrt-safe variant (84 flops): the Newton iterations of division and
   square root receive residuals that are not known to be non-overlapping, so
   no FastTwoSum precondition can be claimed.  A FastTwoSum whose precondition
   fails does not even satisfy s+e=a+b, so this variant uses TwoSum everywhere
   with the same pass count as the standard one. */
__device__
gtd_real tw_fma_safe(const gtd_real &a, double b, const gtd_real &c)
{
  double P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
  double A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;
  P00 = two_prod(a.x, b, E00);
  P01 = 0.0; E01 = 0.0;
  P10 = two_prod(a.y, b, E10);
  P02 = 0.0;
  P11 = 0.0;
  P20 = a.z * b;
  sg  = (P02 + P20) + P11;
  G   = ((E01 + E10) + sg) + c.z;
  A   = two_sum(P01, P10, q1);
  A   = two_sum(A, E00, q2);
  A   = two_sum(A, c.y, q3);
  G   = G + ((q1 + q2) + q3);
  B   = two_sum(P00, c.x, r);
  m1  = two_sum(r, A, m2);
  m2  = m2 + G;
  double z0, z1, z2;
  /* renormalization pass 1/3 */
  w0 = two_sum(B, m1, w1);
  w1 = two_sum(w1, m2, w2);
  /* renormalization pass 2/3 */
  w0 = two_sum(w0, w1, w1);
  w1 = two_sum(w1, w2, w2);
  /* renormalization pass 3/3 */
  z0 = two_sum(w0, w1, w1);
  z1 = two_sum(w1, w2, z2);
  return make_td(z0, z1, z2);
}

__device__
gtd_real tw_fma(const gtd_real &a, double b, const gtd_real &c)
{
  double P00, E00, P01, E01, P10, E10, P02, P11, P20, sg, G;
  double A, q1, q2, q3, B, r, m1, m2, w0, w1, w2;
  P00 = two_prod(a.x, b, E00);
  P01 = 0.0; E01 = 0.0;
  P10 = two_prod(a.y, b, E10);
  P02 = 0.0;
  P11 = 0.0;
  P20 = a.z * b;
  sg  = (P02 + P20) + P11;
  G   = ((E01 + E10) + sg) + c.z;
  A   = two_sum(P01, P10, q1);
  A   = two_sum(A, E00, q2);
  A   = two_sum(A, c.y, q3);
  G   = G + ((q1 + q2) + q3);
  B   = two_sum(P00, c.x, r);
  m1  = two_sum(r, A, m2);
  m2  = m2 + G;
  double z0, z1, z2;
  /* renormalization pass 1/3 */
  w0 = quick_two_sum(B, m1, w1);
  w1 = two_sum(w1, m2, w2);
  /* renormalization pass 2/3 */
  w0 = two_sum(w0, w1, w1);
  w1 = quick_two_sum(w1, w2, w2);
  /* renormalization pass 3/3 */
  z0 = quick_two_sum(w0, w1, w1);
  z1 = quick_two_sum(w1, w2, z2);
  return make_td(z0, z1, z2);
}

/* Generic spelling; same operation. */
__device__
gtd_real fma(const gtd_real &a, const gtd_real &b, const gtd_real &c)
{
	return tw_fma(a, b, c);
}

__device__
gtd_real fma(const gtd_real &a, double b, const gtd_real &c)
{
	return tw_fma(a, b, c);
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
gtd_real standard_div(const gtd_real &a, const gtd_real &b)
{
	double q0, q1, q2, q3;
	gtd_real r;

	q0 = a.x / b.x;
	r  = a - (b * q0);

	q1 = r.x / b.x;
	r  = r - (b * q1);

	q2 = r.x / b.x;
	r  = r - (b * q2);

	q3 = r.x / b.x;

	return make_td_renorm(q0, q1, q2, q3);
}

/* Same correction sequence, but every residual  r <- r - q*b  is one
   fused tw_fma instead of a multiply followed by a subtraction, which
   removes one renormalization per step. */
__device__
gtd_real fma_div(const gtd_real &a, const gtd_real &b)
{
	double q0, q1, q2, q3;
	gtd_real r;

	q0 = a.x / b.x;
	r  = tw_fma_safe(b, -q0, a);       /* r = a - q0 * b */

	q1 = r.x / b.x;
	r  = tw_fma_safe(b, -q1, r);

	q2 = r.x / b.x;
	r  = tw_fma_safe(b, -q2, r);

	q3 = r.x / b.x;

	return make_td_renorm(q0, q1, q2, q3);
}

__device__
gtd_real operator/(const gtd_real &a, const gtd_real &b)
{
#ifdef GQD_NO_FMA_DIV
	return standard_div(a, b);
#else
	return fma_div(a, b);
#endif
}

__device__
gtd_real operator/(const gtd_real &a, double b)
{
	double q0, q1, q2, q3;
	gtd_real r;

	q0 = a.x / b;
	r  = a - (q0 * b);

	q1 = r.x / b;
	r  = r - (q1 * b);

	q2 = r.x / b;
	r  = r - (q2 * b);

	q3 = r.x / b;

	return make_td_renorm(q0, q1, q2, q3);
}

__device__
gtd_real operator/(double a, const gtd_real &b)
{
	return make_td(a) / b;
}

__device__
gtd_real inv(const gtd_real &a)
{
	return 1.0 / a;
}

/*=================================================================
 * miscellaneous
 *=================================================================*/
__host__ __device__
gtd_real abs(const gtd_real &a)
{
	return (a.x < 0.0) ? negative(a) : a;
}

__host__ __device__
gtd_real fabs(const gtd_real &a) { return abs(a); }

__host__ __device__
double to_double(const gtd_real &a) { return a.x; }

__host__ __device__
gtd_real ldexp(const gtd_real &a, int n)
{
	return make_td(::ldexp(a.x, n), ::ldexp(a.y, n), ::ldexp(a.z, n));
}

__device__
gtd_real nint(const gtd_real &a)
{
	double x0 = nint(a.x);
	double x1 = 0.0, x2 = 0.0;

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
	return make_td_renorm(x0, x1, x2, 0.0);
}

/*=================================================================
 * comparisons
 *=================================================================*/
__host__ __device__ bool is_zero    (const gtd_real &a) { return a.x == 0.0; }
__host__ __device__ bool is_one     (const gtd_real &a) { return a.x == 1.0 && a.y == 0.0 && a.z == 0.0; }
__host__ __device__ bool is_positive(const gtd_real &a) { return a.x >  0.0; }
__host__ __device__ bool is_negative(const gtd_real &a) { return a.x <  0.0; }

__host__ __device__ bool operator==(const gtd_real &a, const gtd_real &b)
{ return a.x == b.x && a.y == b.y && a.z == b.z; }
__host__ __device__ bool operator==(const gtd_real &a, double b)
{ return a.x == b && a.y == 0.0 && a.z == 0.0; }
__host__ __device__ bool operator==(double a, const gtd_real &b)
{ return b == a; }

__host__ __device__ bool operator!=(const gtd_real &a, const gtd_real &b) { return !(a == b); }
__host__ __device__ bool operator!=(const gtd_real &a, double b)          { return !(a == b); }
__host__ __device__ bool operator!=(double a, const gtd_real &b)          { return !(a == b); }

__host__ __device__ bool operator<(const gtd_real &a, const gtd_real &b)
{
	if (a.x != b.x) return a.x < b.x;
	if (a.y != b.y) return a.y < b.y;
	return a.z < b.z;
}
__host__ __device__ bool operator<(const gtd_real &a, double b)
{
	if (a.x != b) return a.x < b;
	if (a.y != 0.0) return a.y < 0.0;
	return a.z < 0.0;
}
__host__ __device__ bool operator<(double a, const gtd_real &b) { return b > a; }

__host__ __device__ bool operator>(const gtd_real &a, const gtd_real &b)
{
	if (a.x != b.x) return a.x > b.x;
	if (a.y != b.y) return a.y > b.y;
	return a.z > b.z;
}
__host__ __device__ bool operator>(const gtd_real &a, double b)
{
	if (a.x != b) return a.x > b;
	if (a.y != 0.0) return a.y > 0.0;
	return a.z > 0.0;
}
__host__ __device__ bool operator>(double a, const gtd_real &b) { return b < a; }

__host__ __device__ bool operator<=(const gtd_real &a, const gtd_real &b) { return !(a > b); }
__host__ __device__ bool operator<=(const gtd_real &a, double b)          { return !(a > b); }
__host__ __device__ bool operator<=(double a, const gtd_real &b)          { return !(a > b); }

__host__ __device__ bool operator>=(const gtd_real &a, const gtd_real &b) { return !(a < b); }
__host__ __device__ bool operator>=(const gtd_real &a, double b)          { return !(a < b); }
__host__ __device__ bool operator>=(double a, const gtd_real &b)          { return !(a < b); }

/* polyeval(c, n, x)
   Evaluates the given n-th degree polynomial at x.
   The polynomial is given by the array of (n+1) coefficients. */
__device__
gtd_real polyeval(const gtd_real *c, int n, const gtd_real &x)
{
	/* Horner's method, one fused multiply-add per step.  The
	   machine-proved fma keeps its error bound even when a step
	   cancels almost completely (near a root), so no separate
	   cancellation-safe variant is needed. */
	gtd_real r = c[n];

	for (int i = n - 1; i >= 0; i--) {
		r = tw_fma(r, x, c[i]);
	}

	return r;
}

#endif /* __GTD_BASIC_CU__ */
