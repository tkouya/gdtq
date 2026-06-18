/* SPDX-License-Identifier: BSD-3-Clause */
/* Copyright (c) 2026 Tomonori Kouya. Based on GQD (Mian Lu) and QD (Bailey et al., LBNL). */
#ifndef __GDS_BASIC_CU__
#define __GDS_BASIC_CU__

/**
 * arithmetic operators
 * comparison
 */

//#include "common.cuh"
//#include "gdd_basic.cuh"
#include "gqs.cuh"

///////////////////// Addition /////////////////////

__device__
gds_real negative( const gds_real &a )
{
	return make_ds( -a.x, -a.y );
}

/* float-float = float + float */
__device__
gds_real ds_add(float a, float b) 
{
	float s, e;
	s = two_sum(a, b, e);
	return make_ds(s, e);
}

/* float-float + float */
__host__ __device__
gds_real operator+(const gds_real &a, float b) 
{
	float s1, s2;
	s1 = two_sum(a.x, b, s2);
	s2 += a.y;
	s1 = quick_two_sum(s1, s2, s2);
	return make_ds(s1, s2);
}

/* float + float-float */
__host__ __device__
gds_real operator+(const float &a, gds_real b) 
{
	return b + a;
}

/* inline functions has been moved to header files */
/*

__inline__ __host__ __device__
gds_real sloppy_add(const gds_real &a, const gds_real &b) 
{
	float s, e;

	s = two_sum(a.x, b.x, e);
	e += (a.y + b.y);
	s = quick_two_sum(s, e, e);
	return make_ds(s, e);
}

__inline__ __host__ __device__
gds_real operator+(const gds_real &a, const gds_real &b) 
{
	return sloppy_add(a, b);
}

*/

/*********** Subtractions *********/

#define GQD_IEEE_ADD

__device__
gds_real operator-(const gds_real &a, const gds_real &b) 
{
#ifndef GQD_IEEE_ADD // appended by T.Kouya 2015-07-06
	float s, e;
	s = two_diff(a.x, b.x, e);
	//return make_ds(s, e);
	e += a.y;
	e -= b.y;
	s = quick_two_sum(s, e, e);
	return make_ds(s, e);
#else // GQD_IEEE_ADD
	float s1, s2, t1, t2;
	s1 = two_diff(a.x, b.x, s2);
	t1 = two_diff(a.y, b.y, t2);
	s2 += t1;
	s1 = quick_two_sum(s1, s2, s2);
	s2 += t2;
	s1 = quick_two_sum(s1, s2, s2);
	return make_ds(s1, s2);
#endif // GQD_IEEE_ADD
}

/* float-float - float */
__device__
gds_real operator-(const gds_real &a, float b) 
{
	float s1, s2;
	s1 = two_diff(a.x, b, s2);
	s2 += a.y;
	s1 = quick_two_sum(s1, s2, s2);
	return make_ds(s1, s2);
}

/* float - float-float */
__device__
gds_real operator-(float a, const gds_real &b) {
  float s1, s2;
  s1 = two_diff(a, b.x, s2);
  s2 -= b.y;
  s1 = quick_two_sum(s1, s2, s2);
  return make_ds(s1, s2);
}

/*********** Squaring **********/
__device__
gds_real sqr(const gds_real &a) 
{
	float p1, p2;
	float s1, s2;
	p1 = two_sqr(a.x, p2);
	//p2 += (2.0 * a.x * a.y);
        p2 = __fadd_rn(p2,__fmul_rn(__fmul_rn(2.0,a.x), a.y));
	//p2 += (a.y * a.y);
	p2 = __fadd_rn(p2, __fmul_rn(a.y,a.y));
	s1 = quick_two_sum(p1, p2, s2);
	return make_ds(s1, s2);
}

__device__
gds_real sqr(float a) 
{
	float p1, p2;
	p1 = two_sqr(a, p2);
	return make_ds(p1, p2);
}

/****************** Multiplication ********************/


/* float-float * (2.0 ^ exp) */
__device__
gds_real ldexp(const gds_real &a, int exp) 
{
	return make_ds(ldexp(a.x, exp), ldexp(a.y, exp));
}

/* float-float * float,  where float is a power of 2. */
__device__
gds_real mul_pwr2(const gds_real &a, float b)
{
	return make_ds(a.x * b, a.y * b);
}

/* float-float * float-float */
__device__
gds_real operator*(const gds_real &a, const gds_real &b)
{
	float p1, p2;

	p1 = two_prod(a.x, b.x, p2);
	//p2 += (a.x * b.y + a.y * b.x);
        p2 = p2 + (__fmul_rn(a.x,b.y) + __fmul_rn(a.y,b.x));
	p1 = quick_two_sum(p1, p2, p2);
	return make_ds(p1, p2);
}

/* float-float * float */
__device__
gds_real operator*(const gds_real &a, float b) 
{
	float p1, p2;

	p1 = two_prod(a.x, b, p2);
	p2 = __fadd_rn(p2,(__fmul_rn(a.y,b)));
	p1 = quick_two_sum(p1, p2, p2);
	return make_ds(p1, p2);
}

/* float * float-float */
__device__
gds_real operator*(float a, const gds_real &b) 
{
	return (b * a);
}


/******************* Division *********************/

__device__
gds_real sloppy_div(const gds_real &a, const gds_real &b) 
{
	float s1, s2;
	float q1, q2;
	gds_real r;

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

/* float-float / float-float */
__device__
gds_real operator/(const gds_real &a, const gds_real &b) 
{
	return sloppy_div(a, b);
}



/* float-float / float */
__device__
gds_real operator/(const gds_real &a, float b) {

	float q1, q2;
	float p1, p2;
	float s, e;
	gds_real r;
 
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
bool is_zero( const gds_real &a ) 
{
	return (a.x == 0.0);
}

__host__ __device__
bool is_one( const gds_real &a ) 
{
	return (a.x == 1.0 && a.y == 0.0);
}


/*  this > 0 */
__device__ 
bool is_positive(const gds_real &a) {
	return (a.x > 0.0);
}

/* this < 0 */
__device__ 
bool is_negative(const gds_real &a) {
	return (a.x < 0.0);
}

/* Cast to float. */
__device__
float to_float(const gds_real &a)
{
	return a.x;
}

/************* Comparison ***************/


/* float-float <= float-float */
__host__ __device__
bool operator<=(const gds_real &a, const gds_real &b) {
  return (a.x < b.x || (a.x == b.x && a.y <= b.y));
}



/*********** Equality Comparisons ************/
/* float-float == float */
__host__ __device__ bool operator==(const gds_real &a, float b) {
  return (a.x == b && a.y == 0.0);
}

/* float-float == float-float */
__host__ __device__ bool operator==(const gds_real &a, const gds_real &b) {
  return (a.x == b.x && a.y == b.y);
}

/* float == float-float */
__host__ __device__ bool operator==(float a, const gds_real &b) {
  return (a == b.x && b.y == 0.0);
}

/*********** Greater-Than Comparisons ************/
/* float-float > float */
__host__ __device__ bool operator>(const gds_real &a, float b) {
  return (a.x > b || (a.x == b && a.y > 0.0));
}

/* float-float > float-float */
__host__ __device__ bool operator>(const gds_real &a, const gds_real &b) {
  return (a.x > b.x || (a.x == b.x && a.y > b.y));
}

/* float > float-float */
__host__ __device__ bool operator>(float a, const gds_real &b) {
  return (a > b.x || (a == b.x && b.y < 0.0));
}

/*********** Less-Than Comparisons ************/
/* float-float < float */
__host__ __device__ bool operator<(const gds_real &a, float b) {
  return (a.x < b || (a.x == b && a.y < 0.0));
}

/* float-float < float-float */
__host__ __device__ bool operator<(const gds_real &a, const gds_real &b) {
  return (a.x < b.x || (a.x == b.x && a.y < b.y));
}

/* float < float-float */
__host__ __device__ bool operator<(float a, const gds_real &b) {
  return (a < b.x || (a == b.x && b.y > 0.0));
}

/*********** Greater-Than-Or-Equal-To Comparisons ************/
/* float-float >= float */
__host__ __device__ bool operator>=(const gds_real &a, float b) {
  return (a.x > b || (a.x == b && a.y >= 0.0));
}

/* float-float >= float-float */
__host__ __device__ bool operator>=(const gds_real &a, const gds_real &b) {
  return (a.x > b.x || (a.x == b.x && a.y >= b.y));
}

/* float >= float-float */
//__host__ __device__ bool operator>=(float a, const gds_real &b) {
//  return (b <= a);
//}

/*********** Less-Than-Or-Equal-To Comparisons ************/
/* float-float <= float */
__host__ __device__ bool operator<=(const gds_real &a, float b) {
  return (a.x < b || (a.x == b && a.y <= 0.0));
}

/* float >= float-float */
__host__ __device__ bool operator>=(float a, const gds_real &b) {
  return (b <= a);
}


/* float-float <= float-float */
//__host__ __device__ bool operator<=(const gds_real &a, const gds_real &b) {
//  return (a.x[0] < b.x[0] || (a.x[0] == b.x[0] && a.x[1] <= b.x[1]));
//}

/* float <= float-float */
__host__ __device__ bool operator<=(float a, const gds_real &b) {
  return (b >= a);
}

/*********** Not-Equal-To Comparisons ************/
/* float-float != float */
__host__ __device__ bool operator!=(const gds_real &a, float b) {
  return (a.x != b || a.y != 0.0);
}

/* float-float != float-float */
__host__ __device__ bool operator!=(const gds_real &a, const gds_real &b) {
  return (a.x != b.x || a.y != b.y);
}

/* float != float-float */
__host__ __device__ bool operator!=(float a, const gds_real &b) {
  return (a != b.x || b.y != 0.0);
}



__device__
gds_real nint(const gds_real &a) {
  float hi = nint(a.x);
  float lo;

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

  return make_ds(hi, lo);
}


__device__
gds_real abs(const gds_real &a) {
        return (a.x < 0.0) ? negative(a) : a;
}


__device__
gds_real fabs(const gds_real &a) {
	return abs(a);
}


/* float / float-float */
__device__
gds_real operator/(float a, const gds_real &b) {
	return make_ds(a) / b;
}

__device__
gds_real inv(const gds_real &a) {
  return 1.0 / a;
}

#endif /* __GDS_BASIC_CU__ */

