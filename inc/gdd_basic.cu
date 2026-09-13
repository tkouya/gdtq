/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */
#ifndef __GDD_BASIC_CU__
#define __GDD_BASIC_CU__

/**
 * arithmetic operators
 * comparison
 */

//#include "common.cuh"
//#include "gdd_basic.cuh"
#include "gqd.cuh"

///////////////////// Addition /////////////////////

__device__
gdd_real negative( const gdd_real &a )
{
	return make_dd( -a.x, -a.y );
}

/* double-double = double + double */
__device__
gdd_real dd_add(double a, double b) 
{
	double s, e;
	s = two_sum(a, b, e);
	return make_dd(s, e);
}

/* double-double + double */
__host__ __device__
gdd_real operator+(const gdd_real &a, double b) 
{
	double s1, s2;
	s1 = two_sum(a.x, b, s2);
	s2 += a.y;
	s1 = quick_two_sum(s1, s2, s2);
	return make_dd(s1, s2);
}

/* double + double-double */
__host__ __device__
gdd_real operator+(const double &a, gdd_real b) 
{
	return b + a;
}

/* inline functions has been moved to header files */
/*

__inline__ __host__ __device__
gdd_real sloppy_add(const gdd_real &a, const gdd_real &b) 
{
	double s, e;

	s = two_sum(a.x, b.x, e);
	e += (a.y + b.y);
	s = quick_two_sum(s, e, e);
	return make_dd(s, e);
}

__inline__ __host__ __device__
gdd_real operator+(const gdd_real &a, const gdd_real &b) 
{
	return sloppy_add(a, b);
}

*/

/*********** Subtractions *********/

#define GQD_IEEE_ADD

__device__
gdd_real operator-(const gdd_real &a, const gdd_real &b) 
{
#ifndef GQD_IEEE_ADD // appended by T.Kouya 2015-07-06
	double s, e;
	s = two_diff(a.x, b.x, e);
	//return make_dd(s, e);
	e += a.y;
	e -= b.y;
	s = quick_two_sum(s, e, e);
	return make_dd(s, e);
#else // GQD_IEEE_ADD
	double s1, s2, t1, t2;
	s1 = two_diff(a.x, b.x, s2);
	t1 = two_diff(a.y, b.y, t2);
	s2 += t1;
	s1 = quick_two_sum(s1, s2, s2);
	s2 += t2;
	s1 = quick_two_sum(s1, s2, s2);
	return make_dd(s1, s2);
#endif // GQD_IEEE_ADD
}

/* double-double - double */
__device__
gdd_real operator-(const gdd_real &a, double b) 
{
	double s1, s2;
	s1 = two_diff(a.x, b, s2);
	s2 += a.y;
	s1 = quick_two_sum(s1, s2, s2);
	return make_dd(s1, s2);
}

/* double - double-double */
__device__
gdd_real operator-(double a, const gdd_real &b) {
  double s1, s2;
  s1 = two_diff(a, b.x, s2);
  s2 -= b.y;
  s1 = quick_two_sum(s1, s2, s2);
  return make_dd(s1, s2);
}

/*********** Squaring **********/
__device__
gdd_real sqr(const gdd_real &a) 
{
	double p1, p2;
	double s1, s2;
	p1 = two_sqr(a.x, p2);
	//p2 += (2.0 * a.x * a.y);
        p2 = __dadd_rn(p2,__dmul_rn(__dmul_rn(2.0,a.x), a.y));
	//p2 += (a.y * a.y);
	p2 = __dadd_rn(p2, __dmul_rn(a.y,a.y));
	s1 = quick_two_sum(p1, p2, s2);
	return make_dd(s1, s2);
}

/* Canonical "square a double, return gdd_real".  Always defined.
 * gdd_sqrt.cu calls this directly so it does not depend on the
 * conditional sqr(double) alias below. */
__device__
gdd_real sqr_d(double a)
{
	double p1, p2;
	p1 = two_sqr(a, p2);
	return make_dd(p1, p2);
}

/* Convenience alias.  See gdd_basic.cuh for the QD coexistence note. */
#ifndef _QD_INLINE_H
__device__
gdd_real sqr(double a)
{
	return sqr_d(a);
}
#endif

/****************** Multiplication ********************/


/* double-double * (2.0 ^ exp) */
__device__
gdd_real ldexp(const gdd_real &a, int exp) 
{
	return make_dd(ldexp(a.x, exp), ldexp(a.y, exp));
}

/* double-double * double,  where double is a power of 2. */
__device__
gdd_real mul_pwr2(const gdd_real &a, double b)
{
	return make_dd(a.x * b, a.y * b);
}

/* double-double * double-double */
__device__
gdd_real operator*(const gdd_real &a, const gdd_real &b)
{
	double p1, p2;

	p1 = two_prod(a.x, b.x, p2);
	//p2 += (a.x * b.y + a.y * b.x);
        p2 = p2 + (__dmul_rn(a.x,b.y) + __dmul_rn(a.y,b.x));
	p1 = quick_two_sum(p1, p2, p2);
	return make_dd(p1, p2);
}

/* double-double * double */
__device__
gdd_real operator*(const gdd_real &a, double b) 
{
	double p1, p2;

	p1 = two_prod(a.x, b, p2);
	p2 = __dadd_rn(p2,(__dmul_rn(a.y,b)));
	p1 = quick_two_sum(p1, p2, p2);
	return make_dd(p1, p2);
}

/* double * double-double */
__device__
gdd_real operator*(double a, const gdd_real &b) 
{
	return (b * a);
}


/**************** Fused multiply-add ****************/

// 2026-08-04 T.Kouya
// Branch free algorithm: double-word FMA,  z = a * b + c.
//
// The inputs are spread over two "levels" of magnitude (level 0 is the
// leading word, level 1 the trailing word).  Every term of the exact
// product a*b and of c is dropped into the level it belongs to, the two
// levels are accumulated with two_sum, and one quick_two_sum
// renormalizes the result.  No branch, and a*b is never renormalized on
// its own, so this is cheaper than -- and no less accurate than -- a*b+c.
/* DW-FMA  z = a * b + c   (17 flops)
   Machine-proved with FPANVerifier + z3 5.0.0 (ACS2026 formulation):
     error bound      |z-(ab+c)| <= 35 u^2 (|ab|+|c|)
                      (FPANVerifier anchor-relative 34 u^2 plus the
                       analytic input-relative conversion)
     every FastTwoSum precondition  exp(x) >= exp(y)
     non-overlapping output         z0 |> z1   (strongly_dominates)
   Normalization repeats a cascade over adjacent pairs; the pass count is the
   smallest for which the non-overlap is provable (DW 1 / TW 3 / QW 5). */
__device__
gdd_real dw_fma(const gdd_real &a, const gdd_real &b, const gdd_real &c)
{
  double P00, E00, P01, P10, l, v, w, s, t, tp;
  P00 = two_prod(a.x, b.x, E00);
  P01 = a.x * b.y;
  P10 = a.y * b.x;
  l   = P01 + P10;
  v   = E00 + c.y;
  w   = v + l;
  s   = two_sum(P00, c.x, t);
  tp  = t + w;
  double z0, z1;
  z0 = quick_two_sum(s, tp, z1);
  return make_dd(z0, z1);
}

/* DW-FMA  z = a * b + c   (17 flops, scalar multiplier)
   Machine-proved with FPANVerifier + z3 5.0.0 (ACS2026 formulation):
     error bound      |z-(ab+c)| <= 35 u^2 (|ab|+|c|)
                      (FPANVerifier anchor-relative 34 u^2 plus the
                       analytic input-relative conversion)
     every FastTwoSum precondition  exp(x) >= exp(y)
     non-overlapping output         z0 |> z1   (strongly_dominates)
   Normalization repeats a cascade over adjacent pairs; the pass count is the
   smallest for which the non-overlap is provable (DW 1 / TW 3 / QW 5). */
/* div/sqrt-safe variant (20 flops): the Newton iterations of division and
   square root receive residuals that are not known to be non-overlapping, so
   no FastTwoSum precondition can be claimed.  A FastTwoSum whose precondition
   fails does not even satisfy s+e=a+b, so this variant uses TwoSum everywhere
   with the same pass count as the standard one. */
__device__
gdd_real dw_fma_safe(const gdd_real &a, double b, const gdd_real &c)
{
  double P00, E00, P01, P10, l, v, w, s, t, tp;
  P00 = two_prod(a.x, b, E00);
  P01 = 0.0;
  P10 = a.y * b;
  l   = P01 + P10;
  v   = E00 + c.y;
  w   = v + l;
  s   = two_sum(P00, c.x, t);
  tp  = t + w;
  double z0, z1;
  z0 = two_sum(s, tp, z1);
  return make_dd(z0, z1);
}

__device__
gdd_real dw_fma(const gdd_real &a, double b, const gdd_real &c)
{
  double P00, E00, P01, P10, l, v, w, s, t, tp;
  P00 = two_prod(a.x, b, E00);
  P01 = 0.0;
  P10 = a.y * b;
  l   = P01 + P10;
  v   = E00 + c.y;
  w   = v + l;
  s   = two_sum(P00, c.x, t);
  tp  = t + w;
  double z0, z1;
  z0 = quick_two_sum(s, tp, z1);
  return make_dd(z0, z1);
}

/* Generic spelling; same operation. */
__device__
gdd_real fma(const gdd_real &a, const gdd_real &b, const gdd_real &c)
{
	return dw_fma(a, b, c);
}

__device__
gdd_real fma(const gdd_real &a, double b, const gdd_real &c)
{
	return dw_fma(a, b, c);
}


/******************* Division *********************/

__device__
gdd_real sloppy_div(const gdd_real &a, const gdd_real &b)
{
	double s1, s2;
	double q1, q2;
	gdd_real r;

	q1 = a.x / b.x;  /* approximate quotient */

	/* compute  this - q1 * dd */
	r = b * q1;
	s1 = two_diff(a.x, r.x, s2);
	s2 -= r.y;
	s2 += a.y;

	/* get next approximation */
	q2 = (s1 + s2) / b.x;

	/* renormalize */
	r.x = quick_two_sum(q1, q2, r.y);
	return r;
}

/* Division built on the branch-free fused multiply-add: each residual
   r <- r - q*b  is one fused dw_fma instead of a multiply followed by
   a subtraction, which removes one renormalization per step.
   The FMA is the div/sqrt-safe variant: the residuals entering it are
   not guaranteed non-overlapping, so no FastTwoSum precondition can be
   asserted here (measured cost is the same as the plain dw_fma to
   within noise, and the safe variant is never less accurate). */
__device__
gdd_real fma_div(const gdd_real &a, const gdd_real &b)
{
	double q1, q2, q3;
	gdd_real r;

	q1 = a.x / b.x;                /* approximate quotient */

	r = dw_fma_safe(b, -q1, a);    /* r = a - q1 * b */

	q2 = r.x / b.x;
	r = dw_fma_safe(b, -q2, r);    /* r = r - q2 * b */

	q3 = r.x / b.x;

	q1 = quick_two_sum(q1, q2, q2);
	return make_dd(q1, q2) + q3;
}

/* double-double / double-double */
__device__
gdd_real operator/(const gdd_real &a, const gdd_real &b)
{
#ifdef GQD_NO_FMA_DIV
	return sloppy_div(a, b);
#else
	return fma_div(a, b);
#endif
}



/* double-double / double */
__device__
gdd_real operator/(const gdd_real &a, double b) {

	double q1, q2;
	double p1, p2;
	double s, e;
	gdd_real r;
 
	q1 = a.x / b;   /* approximate quotient. */

	/* Compute  this - q1 * d */
	p1 = two_prod(q1, b, p2);
	s = two_diff(a.x, p1, e);
	e = e + a.y;
	e = e - p2;
  
	/* get next approximation. */
	q2 = (s + e) / b;

	/* renormalize */
	r.x = quick_two_sum(q1, q2, r.y);

	return r;
}


__host__ __device__
bool is_zero( const gdd_real &a ) 
{
	return (a.x == 0.0);
}

__host__ __device__
bool is_one( const gdd_real &a ) 
{
	return (a.x == 1.0 && a.y == 0.0);
}


/*  this > 0 */
__device__ 
bool is_positive(const gdd_real &a) {
	return (a.x > 0.0);
}

/* this < 0 */
__device__ 
bool is_negative(const gdd_real &a) {
	return (a.x < 0.0);
}

/* Cast to double. */
__device__
double to_double(const gdd_real &a)
{
	return a.x;
}

/************* Comparison ***************/


/* double-double <= double-double */
__host__ __device__
bool operator<=(const gdd_real &a, const gdd_real &b) {
  return (a.x < b.x || (a.x == b.x && a.y <= b.y));
}



/*********** Equality Comparisons ************/
/* double-double == double */
__host__ __device__ bool operator==(const gdd_real &a, double b) {
  return (a.x == b && a.y == 0.0);
}

/* double-double == double-double */
__host__ __device__ bool operator==(const gdd_real &a, const gdd_real &b) {
  return (a.x == b.x && a.y == b.y);
}

/* double == double-double */
__host__ __device__ bool operator==(double a, const gdd_real &b) {
  return (a == b.x && b.y == 0.0);
}

/*********** Greater-Than Comparisons ************/
/* double-double > double */
__host__ __device__ bool operator>(const gdd_real &a, double b) {
  return (a.x > b || (a.x == b && a.y > 0.0));
}

/* double-double > double-double */
__host__ __device__ bool operator>(const gdd_real &a, const gdd_real &b) {
  return (a.x > b.x || (a.x == b.x && a.y > b.y));
}

/* double > double-double */
__host__ __device__ bool operator>(double a, const gdd_real &b) {
  return (a > b.x || (a == b.x && b.y < 0.0));
}

/*********** Less-Than Comparisons ************/
/* double-double < double */
__host__ __device__ bool operator<(const gdd_real &a, double b) {
  return (a.x < b || (a.x == b && a.y < 0.0));
}

/* double-double < double-double */
__host__ __device__ bool operator<(const gdd_real &a, const gdd_real &b) {
  return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

/* double < double-double */
__host__ __device__ bool operator<(double a, const gdd_real &b) {
  return (a < b.x || (a == b.x && b.y > 0.0));
}

/*********** Greater-Than-Or-Equal-To Comparisons ************/
/* double-double >= double */
__host__ __device__ bool operator>=(const gdd_real &a, double b) {
  return (a.x > b || (a.x == b && a.y >= 0.0));
}

/* double-double >= double-double */
__host__ __device__ bool operator>=(const gdd_real &a, const gdd_real &b) {
  return (a.x > b.x || (a.x == b.x && a.y >= b.y));
}

/* double >= double-double */
//__host__ __device__ bool operator>=(double a, const gdd_real &b) {
//  return (b <= a);
//}

/*********** Less-Than-Or-Equal-To Comparisons ************/
/* double-double <= double */
__host__ __device__ bool operator<=(const gdd_real &a, double b) {
  return (a.x < b || (a.x == b && a.y <= 0.0));
}

/* double >= double-double */
__host__ __device__ bool operator>=(double a, const gdd_real &b) {
  return (b <= a);
}


/* double-double <= double-double */
//__host__ __device__ bool operator<=(const gdd_real &a, const gdd_real &b) {
//  return (a.x[0] < b.x[0] || (a.x[0] == b.x[0] && a.x[1] <= b.x[1]));
//}

/* double <= double-double */
__host__ __device__ bool operator<=(double a, const gdd_real &b) {
  return (b >= a);
}

/*********** Not-Equal-To Comparisons ************/
/* double-double != double */
__host__ __device__ bool operator!=(const gdd_real &a, double b) {
  return (a.x != b || a.y != 0.0);
}

/* double-double != double-double */
__host__ __device__ bool operator!=(const gdd_real &a, const gdd_real &b) {
  return (a.x != b.x || a.y != b.y);
}

/* double != double-double */
__host__ __device__ bool operator!=(double a, const gdd_real &b) {
  return (a != b.x || b.y != 0.0);
}



__device__
gdd_real nint(const gdd_real &a) {
  double hi = nint(a.x);
  double lo;

  if (hi == a.x) {
    /* High word is an integer already.  Round the low word.*/
    lo = nint(a.y);
    
    /* Renormalize. This is needed if x[0] = some integer, x[1] = 1/2.*/
    hi = quick_two_sum(hi, lo, lo);
  } else {
    /* High word is not an integer. */
    lo = 0.0;
    if (fabs(hi-a.x) == 0.5 && a.y < 0.0) {
      /* There is a tie in the high word, consult the low word 
         to break the tie. */
      hi -= 1.0;      /* NOTE: This does not cause INEXACT. */
    }
  }

  return make_dd(hi, lo);
}


__device__
gdd_real abs(const gdd_real &a) {
        return (a.x < 0.0) ? negative(a) : a;
}


__device__
gdd_real fabs(const gdd_real &a) {
	return abs(a);
}


/* double / double-double */
__device__
gdd_real operator/(double a, const gdd_real &b) {
	return make_dd(a) / b;
}

__device__
gdd_real inv(const gdd_real &a) {
  return 1.0 / a;
}

/* polyeval(c, n, x)
   Evaluates the given n-th degree polynomial at x.
   The polynomial is given by the array of (n+1) coefficients. */
__device__
gdd_real polyeval(const gdd_real *c, int n, const gdd_real &x)
{
	/* Horner's method, one fused multiply-add per step.  The
	   machine-proved fma keeps its error bound even when a step
	   cancels almost completely (near a root), so no separate
	   cancellation-safe variant is needed. */
	gdd_real r = c[n];

	for (int i = n - 1; i >= 0; i--) {
		r = dw_fma(r, x, c[i]);
	}

	return r;
}

#endif /* __GDD_BASIC_CU__ */

